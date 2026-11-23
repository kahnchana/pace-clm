#!/usr/bin/env python3
"""Durable step-128 supervisor: correctness gates, checkpoint recovery, no quality stops."""
import argparse
import fcntl
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_jevany_estp128 import ROOT, SG, PARENT, read, write, sha, trainer_command
from jevany_mixed_control import enqueue
from pace_clm.rl_anchor import verify_committed_checkpoint
from eval_jevany_estp128 import verify_sources

PY=SG/'.venv/bin/python'


def document_completion(run):
    if read(run/'latest.json')['step']!=128:
        raise ValueError('cannot document completion before step128')
    pending=[p.stem for p in (run/'queue').glob('*.json') if not (run/'evaluation'/p.stem/'result.json').exists()]
    if pending:raise ValueError('cannot document completion with pending evaluations')
    result=read(run/'evaluation/full-step128/result.json')
    if result['job']['step']!=128 or 'estp_bench_official_estp_f1' not in result:
        raise ValueError('endpoint judging incomplete')
    reports=[read(run/f'batch-{i:06d}/report.json') for i in range(32,128)]
    seconds=sum(r['seconds'] for r in reports)
    counts=read(run/'data/audit.json')['counts']
    lines=['# Mixed WEB/EGO/ESTP-IT episode RLOO through step 128','',
        'Training completed all 128 optimizer steps, including 96 updates continued from step 32.',
        'Evaluation scores did not control stopping. Inference used raw learned argmax with response margin zero.','',
        '## Exposure','',f"Additional episode counts: `{counts['additional_episodes']}`.",
        f"Additional decisions: {counts['additional_decisions']:,}.",
        f"Successful continuation-update time: {seconds/3600:.2f} hours / {seconds*8/3600:.2f} GPU-hours; excludes loading, retries, idle time and evaluation.",'',
        '## Final PAUC-F','', '| Cohort | PAUC-F | Useful | Wasted |', '|---|---:|---:|---:|']
    lines += [f"| {name} | {value['pauc_f']:.3f} | {value['useful']} | {value['wasted']} |"
              for name,value in result['cohorts'].items()]
    lines += ['', '## Official ESTP-F1','',
        'ESTP-Bench and the fixed IT test probe use the official scorer and its separate answer-quality rubric.',
        f"Full endpoint artifacts: [{run.name}]({run}).", '',
        'Existing full WEB/EGO cohorts include training identities. Existing heldouts are reused historical evaluations.',
        'IT splits exclude source-video overlap within this phase; eligibility filtering and selected task types are recorded in data/audit.json.',
        'Training uses converted IT PAUC-F rewards; official ESTP-F1 is a separate endpoint metric.', '']
    for name in ('estp_bench_official_estp_f1','estp_it_test_official_estp_f1'):
        value=result[name]
        lines.append(f"- {name}: ESTP-F1 x100 = {value.get('estp_f1_x100', 'see saved result')}.")
    path=ROOT/'docs/experiments/jevany_2b_estp_episode_balanced_to128_results.md'
    path.write_text('\n'.join(lines)+'\n')
    write(run/'completion-documentation.json',dict(document=str(path),step=128,
        final_checkpoint_commit_sha256=sha(run/'step-000128/commit.json'),counts=counts))


def archive_uncommitted(run):
    latest=read(run/'latest.json')['step']
    names=[run/f'batch-{latest:06d}',run/f'step-{latest+1:06d}']
    existing=[p for p in names if p.exists()]
    if existing:
        archive=run/'audit'/datetime.now(timezone.utc).strftime('retry-%Y%m%dT%H%M%S')
        archive.mkdir(parents=True)
        for path in existing:
            if path.is_symlink():raise ValueError('refusing to move parent artifact')
            path.rename(archive/path.name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run.resolve()
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('authorized training node only')
    lock=(run/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources(run)
    runtime=read(run/'runtime-contract.json')
    import torch
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=runtime[k] for k in actual):raise RuntimeError('pinned runtime mismatch')
    write(run/'pinned-training-runtime.json',actual)
    def status(state,**fields):
        write(run/'workflow-status.json',dict(state=state,utc=datetime.now(timezone.utc).isoformat(),**fields))
    def command(argv,log):
        verify_sources(run)
        with (run/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    while read(run/'latest.json')['step']<128:
        try:
            latest=read(run/'latest.json')
            verify_committed_checkpoint(latest['checkpoint'],latest['step'])
            # Repair publication if a process exited after committing latest.json.
            from jevany_estp_control import publish_checkpoint
            for step in [33,48,64,80,96,112,128]:
                report_path=run/f'batch-{step-1:06d}/report.json'
                if step<=latest['step'] and report_path.exists():
                    publish_checkpoint(run,step,run/f'step-{step:06d}',read(report_path))
            archive_uncommitted(run)
            if not (run/'resume-step32-verification.json').exists():
                status('restoring_step32')
                command([PY,ROOT/'scripts/verify_jevany_mixed_resume.py','--run',run],'restore-step32.log')
                restored=read(run/'resume_verification.json')
                if not restored['verified'] or restored['step']!=32 or restored['additional_optimizer_steps']!=0:
                    raise RuntimeError('step32 exact restore failed')
                write(run/'resume-step32-verification.json',restored)
            if not (run/'continuation-gate-passed.json').exists():
                status('first_continuation_update')
                command(trainer_command(run)+['--updates','33','--resume'],'train-step33.log')
                command([PY,ROOT/'scripts/verify_jevany_mixed_resume.py','--run',run],'restore-step33.log')
                restored=read(run/'resume_verification.json')
                if not restored['verified'] or restored['step']!=33 or restored['additional_optimizer_steps']!=0:
                    raise RuntimeError('step33 exact restore failed')
                write(run/'resume-step33-verification.json',restored)
                exposure=read(run/'data/schedule.json')[32]['exposure_after']
                enqueue(run,'parity-step33',run/'step-000033','parity',33,exposure,0)
                status('awaiting_step33_cross_node_parity')
                while not (run/'evaluation/parity-step33/result.json').exists():
                    verify_sources(run);time.sleep(10)
                parity=read(run/'cross-node-parity.json')
                if not parity['verified']:raise RuntimeError('cross-node parity failed')
                report=read(run/'batch-000032/report.json')
                if any(v['head_grad_norm']<=0 or v['lora_grad_norm']<=0 or v['reserved_gib']>=115
                       for v in report['learner'].values()):raise RuntimeError('step33 gradient/memory gate failed')
                write(run/'continuation-gate-passed.json',dict(verified=True,resume_step32=True,resume_step33=True,
                    cross_node_parity=True,additional_restore_optimizer_steps=0))
                enqueue(run,'baseline-step32',PARENT/'step-000032','baseline',32,
                        read(run/'data/schedule.json')[31]['exposure_after'],1)
            status('training',target_step=128,quality_stopping=False)
            command(trainer_command(run)+['--updates','128','--resume'],'train-to128.log')
            if read(run/'latest.json')['step']!=128:raise RuntimeError('trainer exited before step128')
        except Exception as exc:
            status('recovering',error=f'{type(exc).__name__}: {exc}',latest=read(run/'latest.json'),target_step=128)
            with (run/'recovery.log').open('a') as handle:traceback.print_exc(file=handle)
            time.sleep(30)
    # No learners remain. The final evaluator obtains the original GPU lock.
    latest=read(run/'latest.json');verify_committed_checkpoint(latest['checkpoint'],128)
    enqueue(run,'final-reference-step32',PARENT/'step-000032','final_reference',32,
            read(run/'data/schedule.json')[31]['exposure_after'],1)
    enqueue(run,'full-step128',latest['checkpoint'],'final',128,read(run/'data/schedule.json')[-1]['exposure_after'],2)
    status('final_evaluation',checkpoint=latest['checkpoint'])
    while not (run/'evaluation/full-step128/result.json').exists():
        try:command([PY,ROOT/'scripts/eval_jevany_estp128.py','--run',run,'--final-local'],'final-evaluation.log')
        except Exception as exc:
            status('final_evaluation_retry',error=f'{type(exc).__name__}: {exc}');time.sleep(30)
    status('draining_async_evaluations')
    while any(not (run/'evaluation'/p.stem/'result.json').exists() for p in (run/'queue').glob('*.json')):
        time.sleep(10)
    command([PY,ROOT/'scripts/report_jevany_mixed_rl.py','--run',run],'report.log')
    document_completion(run)
    status('complete',checkpoint=latest['checkpoint'],target_step=128,quality_stopping=False)


if __name__=='__main__':
    main()
