#!/usr/bin/env python3
"""Wait for requested RLOO WEB evaluations, then gate and run the E2-G pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from jevany_rl_reward import ROOT, SG, contract, sha, write
from pace_clm.grpo import recover_progress
from run_jevany_e2_rloo import coverage_decision
from run_jevany_rl_full_current_node import checkpoint_view
from run_qwen35_control import verify_web
from run_jevany_e2_full_web import validate_judgments, rows, write_rows, score as score_full_web, PROTOCOL_KEYS, BASELINE
from zs_infer.pauc_f import score_pauc_f

PY=SG/'.venv/bin/python'
RLOO=ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1'
DEFAULT=ROOT/'outputs/jevany-2b-rl/reward-aligned-grpo-v2'
SFT=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
NODE='ip-10-3-120-167'
EVALUATION_ROUNDS=(8,16,24,32)


def read(path):return json.loads(Path(path).read_text())


def prerequisites_complete(directory):
    path=directory/'status.json'
    if not path.exists():return False
    state=read(path)['state']
    if state=='failed':raise RuntimeError('full-WEB prerequisite failed; do not start training')
    if state!='complete':return False
    comparison=read(directory/'comparison.json')
    if not comparison['complete'] or set(comparison['expected_rounds'])!={8,16}:
        raise ValueError('full-WEB comparison is incomplete')
    for step in (8,16):verify_web(directory/f'web-step{step}')
    return True


def training_args(run,rounds,stop_after=None):
    argv=[str(PY),'-u',str(ROOT/'scripts/run_jevany_grpo.py'),'--output',str(run),
          '--checkpoint',str(SFT),'--learners','7','--starts','32','--rollouts','4',
          '--physical-batch','24','--microbatch','24','--collection-batch','24',
          '--frame-cache','2048','--updates',str(rounds),'--full-run',
          '--reward-version','pauc-f-aligned-gpt-v1']
    if (run/'freeze.json').exists():argv.append('--resume')
    if stop_after is not None:argv+=['--stop-after-optimizer-steps',str(stop_after)]
    return argv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    development=run/'development'
    development.mkdir(exist_ok=True)
    for name,source in (('selection.json',RLOO/'development/selection.json'),
                        ('freeze.json',RLOO/'development/freeze.json'),
                        ('sft-baseline.json',RLOO/'development/sft-baseline.json')):
        target=development/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('matched development cohort/baseline changed')
        if not target.exists():shutil.copy2(source,target)
    audit=ROOT/'outputs/jevany-2b-rl/reward-audit-v1'
    if read(audit/'reward-spec.json')!=contract() or not read(audit/'summary.json')['gate_passed']:
        raise ValueError('audited reward changed or audit failed')
    original=read(RLOO/'freeze.json')
    inputs=[SFT/'controller_adapter/adapter_model.safetensors',SFT/'pointer_head.pt',
            ROOT/'outputs/jevany-2b-rl/data/spans.jsonl']
    for path in inputs:
        if sha(path)!=original['hashes'][str(path)]:raise ValueError('SFT or draw schedule differs from E2-R')
    sources=[Path(__file__), ROOT/'scripts/run_jevany_grpo.py', ROOT/'scripts/jevany_grpo_workers.py',
             ROOT/'scripts/verify_jevany_grpo_resume.py',ROOT/'scripts/report_jevany_grpo.py',
             ROOT/'src/pace_clm/grpo.py',ROOT/'src/pace_clm/grpo_vision.py',ROOT/'scripts/jevany_rl_workers.py',
             ROOT/'scripts/run_jevany_e2_rloo.py',ROOT/'scripts/run_jevany_e2_full_web.py',
             ROOT/'scripts/run_jevany_rl_full_current_node.py',ROOT/'scripts/jevany_rl_reward.py',
             ROOT/'scripts/eval_jevany_streaming.py',ROOT/'scripts/merge_jevany_benchmark.py',
             ROOT/'scripts/score_exp10_web_ego.py',ROOT/'scripts/judge_exp10_utility.py',
             ROOT/'scripts/run_qwen35_control.py',SG/'zs_infer/scoring.py',SG/'zs_infer/pauc_f.py']
    hashes={str(p):sha(p) for p in sources}
    plan={'node':NODE,'gpus':list(range(8)),'initial_checkpoint':str(SFT),
          'initial_checkpoint_and_draw_hashes':{str(p):sha(p) for p in inputs},
          'source_hashes':hashes,'reward':contract(),'max_collection_rounds':32,
          'max_optimizer_steps':128,'evaluation_rounds':list(EVALUATION_ROUNDS),'evaluation_web_videos':500,'coverage_subset_videos':128,
          'starts_per_round':32,'rollouts_per_start':4,'horizon':32,
          'actor_passes':2,'global_minibatch':64,'optimizer_steps_per_round':4,
          'clip_epsilon':.2,'vision_replay':'persist exact behavior features; retain original trajectory rank ownership','advantage_population_std':True,'advantage_epsilon':1e-6,
          'head_lr':1e-5,'lora_lr':1e-6,'weight_decay':.01,'kl_beta':.02,
          'entropy_beta':.001,'gradient_clip':1.,'dropout':0.,'supervised_ce':0.,
          'sampling_seed':42,'temperature':1.,'causal_frames':32,'visual_tokens_per_frame':144,
          'input_fps':1,'max_generator_tokens':96,'physical_batch':24,'frame_cache':2048,
          'prerequisite':str(RLOO/'full_web_v1'),'development_selection_sha256':sha(development/'selection.json'),
          'development_baseline_sha256':sha(development/'sft-baseline.json'),
          'authorization':'User requested next planned training experiment after both full-WEB evaluations complete'}
    plan_path=run/'workflow-freeze.json'
    if plan_path.exists() and read(plan_path)!=plan:raise ValueError('GRPO workflow changed since preparation')
    write(plan_path,plan)
    snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in hashes.items():
        target=snapshot/(digest+Path(name).suffix)
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    if args.prepare_only:
        print(json.dumps(plan),flush=True)
        return
    if socket.gethostname()!=NODE:raise RuntimeError('only use the original requested node')
    logs=run/'logs';logs.mkdir(exist_ok=True)

    def status(state,**fields):
        record={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'node':NODE,'pid':os.getpid(),**fields}
        write(run/'workflow-status.json',record)
        with (run/'events.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        TORCHINDUCTOR_COMPILE_THREADS='4',PACE_DISTRIBUTED_BACKEND='gloo',
        TRITON_CACHE_DIR='/tmp/pace-rl-triton',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))

    def command(argv,log):
        for name,expected in hashes.items():
            if sha(name)!=expected:raise ValueError(f'frozen workflow source changed: {name}')
        print(json.dumps({'command':list(map(str,argv)),'log':str(logs/log)}),flush=True)
        with (logs/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)

    def evaluate(step):
        full=run/f'evaluation/web-step{step}';full.mkdir(parents=True,exist_ok=True)
        output=run/f'evaluation/dev-step{step}';output.mkdir(parents=True,exist_ok=True)
        if not (full/'pauc_f_summary-gpt-4.1-v1.json').exists():
            view=checkpoint_view(run,run/f'step-{step:06d}')
            command([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_streaming.py','--controller',view,'--benchmark','web',
                '--output-dir',full,'--response-margin','0','--answer-model','promoted',
                '--max-new-tokens','96','--record-action-logits'],f'web-step{step}-rollout.log')
            command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',full],f'web-step{step}-merge.log')
            status('gpt_scoring_full_web',round=step,videos=500)
            score_full_web(full,SG/'datasets/proactive-videoqa-web/anno.json',RLOO/'full_web_v1/gpt-cache.sqlite')
        verify_web(full)
        protocol=read(full/'manifest.json')
        control=read(BASELINE/'manifest.json')
        if any(protocol[key]!=control[key] for key in PROTOCOL_KEYS):
            raise ValueError('full-WEB GRPO generation protocol differs from SFT control')
        selected=read(development/'selection.json');wanted=set(selected)
        predictions={r['question_id']:r for r in rows(full/'predictions.jsonl') if r['question_id'] in wanted}
        scored={r['question_id']:r for r in read(full/'pauc_llm_scores-gpt-4.1.json') if r['question_id'] in wanted}
        if len(predictions)!=128 or len(scored)!=128:raise ValueError('incomplete frozen development subset')
        write_rows(output/'predictions.jsonl',[predictions[q] for q in selected])
        write(output/'pauc_llm_scores-gpt-4.1.json',[scored[q] for q in selected])
        write(output/'manifest.json',{**protocol,'rows':128,'question_ids':selected})
        requests=[r for r in rows(full/'pauc_batch_input-gpt-4.1.jsonl') if '='.join(r['custom_id'].split('=')[:-2]) in wanted]
        responses=[r for r in rows(full/'pauc_batch_output-gpt-4.1.jsonl') if '='.join(r['custom_id'].split('=')[:-2]) in wanted]
        validate_judgments(requests,responses)
        write_rows(output/'pauc_batch_input-gpt-4.1.jsonl',requests)
        write_rows(output/'pauc_batch_output-gpt-4.1.jsonl',responses)
        score=score_pauc_f(output/'predictions.jsonl',output/'pauc_llm_scores-gpt-4.1.json',omega=.5,judge_model='gpt-4.1')
        write(output/'pauc_f_summary-gpt-4.1-v1.json',score)
        write(output/'provenance.json',{'source':str(full),'same_saved_outputs_and_judgments':True,
              'selected_ids_sha256':sha(development/'selection.json')})
        return score

    try:
        status('waiting_for_full_web_evaluations',prerequisite=str(RLOO/'full_web_v1/status.json'))
        while not prerequisites_complete(RLOO/'full_web_v1'):
            time.sleep(30)
        write(run/'prerequisite-verified.json',{'complete':True,'status_sha256':sha(RLOO/'full_web_v1/status.json'),
             'comparison_sha256':sha(RLOO/'full_web_v1/comparison.json'),'verified_utc':datetime.now(timezone.utc).isoformat()})
        if len(os.sched_getaffinity(0))<8:raise RuntimeError('requires --cpu-bind=none')
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            status('waiting_for_node_gpu_cleanup');time.sleep(10)
        names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
        if len(names)!=8 or any('H200' not in x for x in names):raise RuntimeError('requires eight local H200s')
        progress=recover_progress(run)
        if not (run/'gate-passed.json').exists():
            if progress['optimizer_steps']<1:
                status('gate_first_minibatch',optimizer_step_target=1)
                command(training_args(run,1,stop_after=1),'gate-first-update.log')
            progress=recover_progress(run)
            if progress['optimizer_steps']==1 and not (run/'midbatch-resume-verified.json').exists():
                status('gate_midbatch_restore',round=0,next_update=1)
                command([PY,ROOT/'scripts/verify_jevany_grpo_resume.py','--run',run],'gate-midbatch-restore.log')
                restored=read(run/'resume_verification.json')
                if (not restored['verified'] or restored['optimizer_steps']!=1 or
                        restored['additional_optimizer_steps']!=0 or restored['cursor']['next_update']!=1):
                    raise ValueError('mid-batch restore verification failed')
                write(run/'midbatch-resume-verified.json',restored)
            if not (run/'midbatch-resume-verified.json').exists():raise RuntimeError('missing controlled mid-batch restore gate')
            if progress['step']<1:
                status('gate_complete_round',optimizer_step_target=4)
                command(training_args(run,1),'gate-finish-round.log')
            report=read(run/'batch-000000/report.json')
            if report['optimizer_steps']!=4 or report['actor_replayed_decisions']!=8192:
                raise ValueError('first round did not use exactly two passes/four optimizer steps')
            first=list(report['updates'][0]['learner'].values())
            for value in first:
                if (abs(value['ratio_min']-1)>.05 or abs(value['ratio_max']-1)>.05 or
                        abs(value['kl_sum'])>1e-4 or
                        (value['decisions'] and (value['head_grad_norm']<=0 or value['lora_grad_norm']<=0))):
                    raise ValueError('initial ratios, SFT KL or gradients failed the GPU gate')
            status('gate_completed_round_restore')
            command([PY,ROOT/'scripts/verify_jevany_grpo_resume.py','--run',run],'gate-round-restore.log')
            restored=read(run/'resume_verification.json')
            if restored['step']!=1 or restored['optimizer_steps']!=4 or restored['cursor'] is not None:
                raise ValueError('completed-round restore counter mismatch')
            write(run/'gate-passed.json',{'verified':True,'collection_rounds':1,'optimizer_steps':4,
                'midbatch_restore':read(run/'midbatch-resume-verified.json'),
                'round_restore':restored,'profile_seconds':report['seconds'],
                'profile_end_to_end_seconds':report['end_to_end_seconds']})
            status('gate_passed',core_round_seconds=report['seconds'],optimizer_steps=4)
        baseline=read(development/'sft-baseline.json')
        checks=[]
        for step in EVALUATION_ROUNDS:
            progress=recover_progress(run)
            if progress['step']<step:
                status('training',completed_rounds=progress['step'],optimizer_steps=progress['optimizer_steps'],target_round=step)
                command(training_args(run,step),f'train-to-{step}.log')
            status('full_web_evaluation',round=step,videos=500)
            score=evaluate(step)
            checks.append((step,score));write(development/'checks.json',checks)
            command([PY,ROOT/'scripts/report_jevany_grpo.py','--run',run],f'report-step{step}.log')
            pause=coverage_decision(baseline,checks)
            if pause:
                status('stopped_by_coverage_gate',reason=pause[0],round=pause[1])
                command([PY,ROOT/'scripts/report_jevany_grpo.py','--run',run],'report.log')
                return
        status('complete',collection_rounds=32,optimizer_steps=128,automatic_extension=False)
        command([PY,ROOT/'scripts/report_jevany_grpo.py','--run',run],'report.log')
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
