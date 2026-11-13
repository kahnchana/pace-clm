#!/usr/bin/env python3
"""Run the frozen mixed-domain pilot with a remote evaluation gate."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time

from jevany_rl_reward import ROOT,SG,sha,write
from jevany_mixed_control import enqueue,read
from prepare_jevany_mixed_rl import DEFAULT,OLD

PY=SG/'.venv/bin/python'
SFT=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
ANCHOR=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1/anchor/replay.jsonl'


def freeze(run):
    schedule=read(run/'data/schedule.json')
    audit=read(run/'data/audit.json');sampling_unit=audit.get('sampling_unit','decision')
    full=sorted({next((i+1 for i,s in enumerate(schedule) if s['exposure_after']>=target),len(schedule))
                 for target in (25000,50000,75000,100000)})
    fast=sorted({next((i+1 for i,s in enumerate(schedule) if s['exposure_after']>=target),len(schedule))
                 for target in range(10000,100000,10000)}-set(full))
    sources=list((ROOT/'scripts').glob('*mixed*.py'))+[ROOT/'src/pace_clm/rl_mixed.py',
        ROOT/'src/pace_clm/rl_episode.py',ROOT/'src/pace_clm/rl.py',ROOT/'src/pace_clm/rl_anchor.py',
        ROOT/'scripts/jevany_rl_reward.py',ROOT/'scripts/eval_jevany_streaming.py',
        ROOT/'scripts/train_jevany_streaming.py',ROOT/'scripts/benchmark_qwen35_2b_training.py',
        ROOT/'scripts/merge_jevany_benchmark.py',ROOT/'src/pace_clm/jevany_streaming.py',
        ROOT/'src/pace_clm/qwen_controller_runtime.py',ROOT/'src/pace_clm/evaluation_policy.py',
        SG/'zs_infer/pauc_f.py',SG/'third_party/proactive-videoqa/pauc/judge_llm_scores_openai.py']
    sources+=list((run/'data').glob('*'))
    sources += [ROOT/'scripts/prepare_jevany_episode_balanced_mixed.py',
        ROOT/'scripts/run_jevany_episode_balanced_mixed.sh',
        SG/'src/stream_gemma/v24_memory.py',SG/'src/stream_gemma/config.py']
    hashes={str(p.resolve()):sha(p) for p in sources if p.is_file()}
    plan={'version':audit.get('version','mixed-web-ego-rloo-v1'),'sampling_unit':sampling_unit,
        'initial_checkpoint':str(SFT),'anchor_manifest':str(ANCHOR),
        'training_node':'ip-10-3-120-167','training_allocation':38177,'evaluation_node':'ip-10-3-70-177',
        'evaluation_allocation':38226,'gpu_count_per_node':8,'decision_budget':100000,
        'source_weights':{'ours':.5,'web':.25,'ego':.25},'seed':42,
        'full_steps':full,'fast_steps':fast,'schedule':schedule,'source_hashes':hashes,
        'coverage_gate':{'useful_fraction_of_sft':.9,'consecutive_full_evals':2,
                         'near_silence_fire_fraction':.1,'max_decisions_lead':10000},
        'inference':'raw learned argmax, response margin 0; no score modifications',
        'selection':'WEB/EGO train cohorts intentionally seen; heldout excluded from this RL run; no untouched-test claim'}
    path=run/'workflow-freeze.json'
    if path.exists() and read(path)!=plan:raise ValueError('immutable workflow configuration changed')
    if not path.exists():write(path,plan)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();run=a.run.resolve()
    run.mkdir(parents=True,exist_ok=True);(run/'cache').mkdir(exist_ok=True)
    plan=freeze(run)
    # Use separate caches for the two concurrent judge processes.
    cache=run/'cache/training.sqlite'
    if not cache.exists():
        source_path=(ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1/cache/training.sqlite'
            if plan['sampling_unit']=='episode' else ROOT/'outputs/jevany-2b-rl/reward-audit-v1/gpt-cache.sqlite')
        with sqlite3.connect(source_path) as source:
            with sqlite3.connect(cache) as dest:source.backup(dest)
    eval_cache=run/'cache/evaluation.sqlite'
    if plan['sampling_unit']=='episode' and not eval_cache.exists():
        with sqlite3.connect(ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1/cache/evaluation.sqlite') as source:
            with sqlite3.connect(eval_cache) as dest:source.backup(dest)
    enqueue(run,'baseline-sft-smoke',SFT,'baseline_smoke',priority=0)
    enqueue(run,'baseline-sft',SFT,'baseline',priority=2)
    enqueue(run,'baseline-prior',OLD/'step-000023','baseline',priority=4)
    if plan['sampling_unit']=='episode':
        prior=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1'
        enqueue(run,'reference-mixed-v1-step16',prior/'step-000016','reference',16,89680,priority=6)
    if a.prepare_only:print(json.dumps({'full_steps':plan['full_steps'],'fast_steps':plan['fast_steps']}));return
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('training allocation only')
    lock=(run/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        TRITON_CACHE_DIR='/tmp/pace-mixed-train-triton',TORCHINDUCTOR_COMPILE_THREADS='4',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    def status(state,**fields):write(run/'workflow-status.json',{'state':state,'utc':datetime.now(timezone.utc).isoformat(),**fields})
    def check():
        for path,expected in plan['source_hashes'].items():
            if sha(path)!=expected:raise ValueError('frozen source changed: '+path)
        ep=run/'evaluation-status.json'
        if ep.exists() and read(ep)['state']=='failed':raise RuntimeError('evaluation worker failed: '+read(ep)['error'])
    def command(argv,log):
        check()
        with (run/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    trainer=[PY,ROOT/'scripts/run_jevany_mixed_rloo.py','--output',run,'--spans',run/'data/episodes.jsonl',
        '--checkpoint',SFT,'--anchor-manifest',ANCHOR,'--learners','7','--starts','2','--rollouts','4',
        '--microbatch','24','--physical-batch','24','--collection-batch','24','--generator-batch','8',
        '--frame-cache','2048','--decision-budget','100000','--reward-version','pauc-f-aligned-gpt-v1','--full-run']
    trainer+=['--sampling-unit',plan['sampling_unit']]
    try:
        if not (run/'gate-passed.json').exists():
            status('first_update_gate')
            if not (run/'latest.json').exists():
                command(trainer+['--updates','1']+(['--resume'] if (run/'freeze.json').exists() else []),'train-gate.log')
            status('restoring_first_update')
            if not (run/'resume_verification.json').exists():
                command([PY,ROOT/'scripts/verify_jevany_mixed_resume.py','--run',run],'restore-gate.log')
            status('awaiting_remote_gate_and_sft_baseline')
            while not all((run/p).exists() for p in ('evaluation/gate-step1/result.json','evaluation/baseline-sft/result.json','cross-node-parity.json')):
                check();time.sleep(10)
            restore=read(run/'resume_verification.json');parity=read(run/'cross-node-parity.json')
            report=read(run/'batch-000000/report.json')
            if not restore['verified'] or not parity['verified']:raise ValueError('restore/parity gate failed')
            if set(report['domain_trajectory_counts'])!={'ours','web','ego'}:raise ValueError('first update missed a source')
            for rank in report['learner'].values():
                if rank['reserved_gib']>=115 or rank['head_grad_norm']<=0 or rank['lora_grad_norm']<=0:
                    raise ValueError('first update memory/gradient gate failed')
            write(run/'gate-passed.json',{'verified':True,'first_checkpoint':restore['checkpoint'],
                'cross_node_replay':True,'restore_optimizer_steps':0,'baseline':'evaluation/baseline-sft/result.json'})
        status('training')
        command(trainer+['--updates',str(len(plan['schedule'])),'--resume'],'train-pilot.log')
        latest=read(run/'latest.json')
        if latest['step'] not in plan['full_steps']:
            exposure=plan['schedule'][latest['step']-1]['exposure_after']
            enqueue(run,'final-step'+str(latest['step']),latest['checkpoint'],'full',latest['step'],exposure,2)
        status('evaluating_final')
        while any(not (run/'evaluation'/p.stem/'result.json').exists() for p in (run/'queue').glob('*.json')):
            check();time.sleep(10)
        command([PY,ROOT/'scripts/report_jevany_mixed_rl.py','--run',run],'report.log')
        latest=read(run/'latest.json');exposure=plan['schedule'][latest['step']-1]['exposure_after']
        status('stopped_by_coverage_gate' if (run/'stop-request.json').exists() else 'complete',
               checkpoint=latest['checkpoint'],decisions=exposure)
    except BaseException as exc:
        # Publish the last durable checkpoint even when the learner fails.
        latest_path=run/'latest.json'
        if latest_path.exists():
            latest=read(latest_path);step=latest['step'];exposure=plan['schedule'][step-1]['exposure_after']
            if step not in plan['full_steps']:
                enqueue(run,'final-step'+str(step),latest['checkpoint'],'full',step,exposure,2)
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
