#!/usr/bin/env python3
"""Recoverable four-source continuation, evaluation, and parent endpoint recovery."""
import argparse
import fcntl
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_jevany_balanced160 import ROOT, PY, PARENT, read, write, sha, trainer_command
from eval_jevany_estp128 import verify_sources
from jevany_balanced160_control import publish_checkpoint
from jevany_mixed_control import enqueue
from pace_clm.rl_anchor import verify_committed_checkpoint
from run_jevany_estp128 import archive_uncommitted


def document(run):
    if read(run/'latest.json')['step'] != 160:
        raise ValueError('training incomplete')
    names = ('baseline-step128','full-step144','full-step160')
    results = {name:read(run/f'evaluation/{name}/result.json') for name in names}
    reports = [read(run/f'batch-{i:06d}/report.json') for i in range(128,160)]
    seconds = sum(r['seconds'] for r in reports)
    lines = ['# Four-source balanced RLOO continuation, steps 129–160','',
        'Continued model, optimizer, seven learner RNG states, sampler cursor and supervised replay from step 128.',
        'Each update: 2 original + 2 WEB + 2 native EGO + 2 ESTP-IT episodes; four trajectories per episode.',
        'Each source has objective weight 0.25; decision loss retains constant-32 normalization.',
        'Raw learned argmax, response margin zero; no inference-time score modifications.','',
        f'Successful update time: {seconds/3600:.2f} node-hours / {seconds*8/3600:.2f} allocated GPU-hours; loading, gates, retries and evaluation excluded.',
        '64 native EGO episodes reused from parent training. Other phase episodes are fresh; heldouts remain excluded.',
        'Evaluation cohorts are reused development cohorts; full WEB/EGO include training examples.',
        'IT dev uses the PAUC-F proxy. IT test and ESTP-Bench remain in the original parent endpoint suite.','',
        '| Cohort | Step 128 | Step 144 | Step 160 | Change 128→160 |',
        '|---|---:|---:|---:|---:|']
    for cohort in results[names[0]]['cohorts']:
        values = [results[n]['cohorts'][cohort]['pauc_f'] for n in names]
        lines.append(f'| {cohort} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[2]-values[0]:+.3f} |')
    lines += ['', 'Paired bootstrap intervals, useful/wasted fires, precision and recall are in each evaluation result.json.',
        'This one-arm continuation changes both episode mix and configured source weights; it does not isolate them or compare trajectory counts.','']
    target=ROOT/'docs/experiments/jevany_2b_balanced_to160_results.md'
    target.write_text('\n'.join(lines))
    write(run/'completion-documentation.json',dict(document=str(target),step=160,
        checkpoint_commit_sha256=sha(run/'step-000160/commit.json'),
        additional_updates=32,evaluation_steps=[128,144,160]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    if socket.gethostname()!='ip-10-3-120-167' or os.environ.get('SLURM_JOB_ID')!='38177':
        raise RuntimeError('authorized training allocation only')
    lock=(run/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources(run)
    import torch
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=read(run/'runtime-contract.json')[k] for k in actual):
        raise RuntimeError('pinned runtime mismatch')
    cache_keys=('TMPDIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR')
    caches={k:os.environ.get(k) for k in cache_keys}
    if any(not v or not Path(v).resolve().is_relative_to('/opt/dlami/nvme') for v in caches.values()):
        raise RuntimeError('temporary caches must use local NVMe')
    write(run/'pinned-training-runtime.json',dict(**actual,caches=caches))
    def status(state,**fields):
        write(run/'workflow-status.json',dict(state=state,utc=datetime.now(timezone.utc).isoformat(),
            target_step=160,quality_stopping=False,**fields))
    def command(argv,log):
        verify_sources(run)
        with (run/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    def restore(step):
        # Correctness restores run only on the training node and own its GPUs.
        with (ROOT/'outputs/jevany-2b-rl/gpu.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('restore requires unused training GPUs')
            command([PY,ROOT/'scripts/verify_jevany_mixed_resume.py','--run',run],f'restore-step{step}.log')
        result=read(run/'resume_verification.json')
        if not result['verified'] or result['step']!=step or result['additional_optimizer_steps']!=0:
            raise RuntimeError(f'exact restore failed at {step}')
        write(run/f'resume-step{step}-verification.json',result)
    while read(run/'latest.json')['step']<160:
        try:
            latest=read(run/'latest.json')
            # Adopt a fully committed update if publication was interrupted.
            next_step=latest['step']+1
            candidate=run/f'step-{next_step:06d}'
            report=run/f'batch-{next_step-1:06d}/report.json'
            if (candidate/'commit.json').exists() and report.exists():
                verify_committed_checkpoint(candidate,next_step)
                write(run/'latest.json',dict(step=next_step,checkpoint=str(candidate)))
                latest=read(run/'latest.json')
            verify_committed_checkpoint(latest['checkpoint'],latest['step'])
            for step in (129,144,160):
                path=run/f'batch-{step-1:06d}/report.json'
                if step<=latest['step'] and path.exists():
                    publish_checkpoint(run,step,run/f'step-{step:06d}',read(path))
            archive_uncommitted(run)
            if not (run/'resume-step128-verification.json').exists():
                status('restoring_step128');restore(128)
            if not (run/'continuation-gate-passed.json').exists():
                if read(run/'latest.json')['step']==128:
                    status('first_continuation_update')
                    command(trainer_command(run)+['--updates','129','--resume'],'train-step129.log')
                restore(129)
                report=read(run/'batch-000128/report.json')
                if any(v['head_grad_norm']<=0 or v['lora_grad_norm']<=0 or v['reserved_gib']>=115
                       for v in report['learner'].values()):
                    raise RuntimeError('step129 gradient/memory gate failed')
                exposure=report['episode_schedule']['exposure_after']
                enqueue(run,'parity-step129',run/'step-000129','parity',129,exposure,0)
                status('awaiting_step129_cross_node_parity')
                while not (run/'evaluation/parity-step129/result.json').exists():
                    verify_sources(run);time.sleep(10)
                if not read(run/'cross-node-parity.json')['verified']:
                    raise RuntimeError('cross-node observation replay parity failed')
                write(run/'continuation-gate-passed.json',dict(verified=True,restored_steps=[128,129],
                    cross_node_parity=True,additional_restore_optimizer_steps=0))
                enqueue(run,'baseline-step128',PARENT/'step-000128','baseline',128,
                        read(run/'data/schedule.json')[127]['exposure_after'],1)
            status('training')
            command(trainer_command(run)+['--updates','160','--resume'],'train-to160.log')
            if read(run/'latest.json')['step']!=160:
                raise RuntimeError('trainer exited before step160')
        except Exception as exc:
            status('recovering',error=f'{type(exc).__name__}: {exc}',latest=read(run/'latest.json'))
            with (run/'recovery.log').open('a') as handle:traceback.print_exc(file=handle)
            time.sleep(30)
    verify_committed_checkpoint(run/'step-000160',160)
    for step in (144,160):
        publish_checkpoint(run,step,run/f'step-{step:06d}',read(run/f'batch-{step-1:06d}/report.json'))
    write(run/'training-complete.json',dict(step=160,additional_updates=32,utc=datetime.now(timezone.utc).isoformat()))
    # Restore the original endpoint work on its originally authorized training node.
    # Its existing queues, shards, judgments and immutable sources remain untouched.
    status('recovering_parent_final_evaluation')
    while read(PARENT/'workflow-status.json')['state']!='complete':
        try:
            command([PY,ROOT/'scripts/run_jevany_estp128.py','--run',PARENT],'parent-final-recovery.log')
        except Exception as exc:
            status('retrying_parent_final_evaluation',error=f'{type(exc).__name__}: {exc}')
            time.sleep(30)
    status('awaiting_evaluation')
    while any(not (run/'evaluation'/p.stem/'result.json').exists() for p in (run/'queue').glob('*.json')):
        time.sleep(10)
    document(run)
    status('complete',checkpoint=str(run/'step-000160'))


if __name__=='__main__':
    main()
