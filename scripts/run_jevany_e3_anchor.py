#!/usr/bin/env python3
"""Train-only calibrated supervised anchor plus RLOO; raw learned inference."""
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

from run_jevany_e2_rloo import coverage_decision
from run_jevany_rl_full_current_node import checkpoint_view
from run_qwen35_control import verify_web
from run_jevany_e2_full_web import validate_judgments, rows, write_rows, score as score_full_web, PROTOCOL_KEYS, BASELINE
from zs_infer.pauc_f import score_pauc_f

PY=SG/'.venv/bin/python'
RLOO=ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1'
DEFAULT=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1'
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


def training_args(run, rounds):
    argv=[PY,'-u',ROOT/'scripts/run_jevany_rl_anchor.py','--output',run,
          '--checkpoint',SFT,'--learners','7','--starts','32','--rollouts','4',
          '--physical-batch','24','--microbatch','24','--collection-batch','24',
          '--frame-cache','2048','--updates',str(rounds),'--full-run',
          '--reward-version','pauc-f-aligned-gpt-v1','--anchor-manifest',run/'anchor/replay.jsonl',
          '--anchor-target-fraction','.15','--anchor-per-round','126']
    if (run/'freeze.json').exists():argv.append('--resume')
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
    decision=read(ROOT/'outputs/jevany-2b-rl/coverage-audit-v1/decision.json')
    if decision['next_experiment']!='rloo_sft_anchor' or not decision['proceed']:raise ValueError('coverage audit did not authorize this recipe')
    anchor=read(run/'anchor/freeze.json')
    if not anchor['training_only'] or anchor['benchmark_overlap'] or sha(run/'anchor/replay.jsonl')!=anchor['replay_sha256']:
        raise ValueError('supervised training manifest changed or overlaps evaluation')
    audit=ROOT/'outputs/jevany-2b-rl/reward-audit-v1'
    if read(audit/'reward-spec.json')!=contract() or not read(audit/'summary.json')['gate_passed']:
        raise ValueError('audited reward changed or audit failed')
    original=read(RLOO/'freeze.json')
    inputs=[SFT/'controller_adapter/adapter_model.safetensors',SFT/'pointer_head.pt',
            ROOT/'outputs/jevany-2b-rl/data/spans.jsonl']
    for path in inputs:
        if sha(path)!=original['hashes'][str(path)]:raise ValueError('SFT or draw schedule differs from E2-R')
    sources=[Path(__file__), ROOT/'scripts/run_jevany_rl_anchor.py', ROOT/'scripts/jevany_rl_anchor_workers.py',
             ROOT/'scripts/verify_jevany_rl_anchor_resume.py',ROOT/'scripts/report_jevany_rl_anchor.py',
             ROOT/'src/pace_clm/rl.py',ROOT/'src/pace_clm/rl_anchor.py',ROOT/'scripts/jevany_rl_workers.py',
             ROOT/'scripts/prepare_jevany_rl_anchor.py',ROOT/'scripts/audit_jevany_rl_coverage.py',
             run/'anchor/freeze.json',run/'anchor/replay.jsonl',
             ROOT/'outputs/jevany-2b-rl/coverage-audit-v1/decision.json',
             ROOT/'outputs/jevany-2b-rl/coverage-audit-v1/audit.json',
             ROOT/'scripts/report_jevany_e2.py',
             ROOT/'scripts/run_jevany_e2_rloo.py',ROOT/'scripts/run_jevany_e2_full_web.py',
             ROOT/'scripts/run_jevany_rl_full_current_node.py',ROOT/'scripts/jevany_rl_reward.py',
             ROOT/'scripts/eval_jevany_streaming.py',ROOT/'scripts/merge_jevany_benchmark.py',
             ROOT/'scripts/score_exp10_web_ego.py',ROOT/'scripts/judge_exp10_utility.py',
             ROOT/'scripts/run_qwen35_control.py',SG/'zs_infer/scoring.py',SG/'zs_infer/pauc_f.py']
    hashes={str(p):sha(p) for p in sources}
    plan={'node':NODE,'gpus':list(range(8)),'initial_checkpoint':str(SFT),
          'initial_checkpoint_and_draw_hashes':{str(p):sha(p) for p in inputs},
          'source_hashes':hashes,'reward':contract(),'max_collection_rounds':32,
          'max_optimizer_steps':32,'evaluation_rounds':list(EVALUATION_ROUNDS),'evaluation_web_videos':500,'coverage_subset_videos':128,
          'starts_per_round':32,'rollouts_per_start':4,'horizon':32,
          'algorithm':'rloo-with-fixed-supervised-anchor-v1','actor_passes':1,'global_minibatch':128,'optimizer_steps_per_round':1,
          'supervised_replay':{'per_round':126,'per_class':42,'target_gradient_fraction':.15,
                               'coefficient':'first training round, then fixed; no WEB tuning'},
          'head_lr':1e-5,'lora_lr':1e-6,'weight_decay':.01,'kl_beta':.02,
          'entropy_beta':.001,'gradient_clip':1.,'dropout':0.,'supervised_ce':'fixed training-only calibrated coefficient',
          'sampling_seed':42,'temperature':1.,'causal_frames':32,'visual_tokens_per_frame':144,
          'input_fps':1,'max_generator_tokens':96,'physical_batch':24,'frame_cache':2048,
          'prerequisite':str(RLOO/'full_web_v1'),'development_selection_sha256':sha(development/'selection.json'),
          'development_baseline_sha256':sha(development/'sft-baseline.json'),
          'authorization':'User requested coverage audit and supervised-anchor RLOO pilot; no runtime heuristics',
          'inference':'raw learned argmax; response_margin=0; no offsets, thresholds or cooldowns'}
    plan_path=run/'workflow-freeze.json'
    if plan_path.exists() and read(plan_path)!=plan:raise ValueError('RLOO anchor workflow changed since preparation')
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
            raise ValueError('full-WEB RLOO anchor generation protocol differs from SFT control')
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
        status('prerequisite_checks')
        if not prerequisites_complete(RLOO/'full_web_v1'):raise RuntimeError('RLOO evaluations incomplete')
        previous=ROOT/'outputs/jevany-2b-rl/reward-aligned-grpo-v2'
        if read(previous/'workflow-status.json')['state']!='stopped_by_coverage_gate':
            raise RuntimeError('prior GRPO workflow not complete')
        for step in (8,16):verify_web(previous/f'evaluation/web-step{step}')
        if len(os.sched_getaffinity(0))<8:raise RuntimeError('requires --cpu-bind=none')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('original node GPUs are busy')
        names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
        if len(names)!=8 or any('H200' not in x for x in names):raise RuntimeError('requires eight local H200s')
        if not (run/'latest.json').exists():
            status('one_round_correctness_and_anchor_calibration')
            command(training_args(run,1),'gate-first-round.log')
        if not (run/'gate-passed.json').exists():
            status('gate_restore')
            command([PY,ROOT/'scripts/verify_jevany_rl_anchor_resume.py','--run',run],'gate-restore.log')
            restored=read(run/'resume_verification.json')
            report=read(run/'batch-000000/report.json')
            calibration=read(run/'anchor-calibration.json')
            if not restored['verified'] or restored['additional_optimizer_steps']!=0 or restored['step']!=1:
                raise ValueError('restore verification failed')
            if report['optimizer_steps']!=1 or report['decisions']!=4096:
                raise ValueError('incorrect first-round exposure')
            if sum(v['anchor_examples'] for v in report['learner'].values())!=126:
                raise ValueError('incorrect supervised replay exposure')
            if abs(calibration['coefficient']*calibration['unweighted_ce_grad_norm']/calibration['rl_objective_grad_norm']-.15)>1e-6:
                raise ValueError('incorrect train-only anchor calibration')
            for value in report['learner'].values():
                if (value['head_grad_norm']<=0 or value['lora_grad_norm']<=0 or abs(value['kl_sum'])>1e-4 or
                        value['parity_probability_max']>.005):raise ValueError('initial SFT KL, gradients or replay parity failed')
            write(run/'gate-passed.json',{'verified':True,'collection_rounds':1,'optimizer_steps':1,
                  'anchor_calibration':calibration,'round_restore':restored})
            status('gate_passed',core_round_seconds=report['seconds'],optimizer_steps=1)
        baseline=read(development/'sft-baseline.json');checks=[];full_checks=[]
        full_baseline=read(BASELINE/'pauc_f_summary-gpt-4.1-v1.json')
        for step in EVALUATION_ROUNDS:
            progress=read(run/'latest.json')
            if progress['step']<step:
                status('training',completed_rounds=progress['step'],optimizer_steps=progress['step'],target_round=step)
                command(training_args(run,step),f'train-to-{step}.log')
            status('full_web_evaluation',round=step,videos=500)
            score=evaluate(step)
            checks.append((step,score));write(development/'checks.json',checks)
            command([PY,ROOT/'scripts/report_jevany_rl_anchor.py','--run',run],f'report-step{step}.log')
            full_checks.append((step,read(run/f'evaluation/web-step{step}/pauc_f_summary-gpt-4.1-v1.json')))
            write(development/'full-web-checks.json',full_checks)
            pause=coverage_decision(baseline,checks) or coverage_decision(full_baseline,full_checks)
            if pause:
                status('stopped_by_coverage_gate',reason=pause[0],round=pause[1])
                command([PY,ROOT/'scripts/report_jevany_rl_anchor.py','--run',run],'report.log')
                return
        status('complete',collection_rounds=32,optimizer_steps=32,automatic_extension=False)
        command([PY,ROOT/'scripts/report_jevany_rl_anchor.py','--run',run],'report.log')
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
