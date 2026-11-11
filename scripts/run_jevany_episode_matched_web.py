#!/usr/bin/env python3
"""Evaluate nearest historical exposure/cost controls on the original node."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from jevany_rl_reward import ROOT, SG, sha, write
from report_jevany_e2 import paired_bootstrap
from run_jevany_e2_full_web import BASELINE, PROTOCOL_KEYS, score as score_full_web
from run_jevany_rl_full_current_node import checkpoint_view
from run_qwen35_control import verify_web

PY=SG/'.venv/bin/python'
ANCHOR=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1'
NODE='ip-10-3-120-167'


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args();run=args.run.resolve()
    if socket.gethostname()!=NODE:raise RuntimeError('authorized node only')
    controls=[];cost=0.;decisions=0
    for step in range(1,33):
        report=ANCHOR/f'batch-{step-1:06d}/report.json';r=read(report)
        cost+=r['gpu_hours'];decisions+=r['decisions']
        checkpoint=ANCHOR/f'step-{step:06d}'
        files=[checkpoint/name for name in ('controller_adapter/adapter_model.safetensors',
                                            'pointer_head.pt','model_identity.json')]
        if all(p.exists() for p in files):
            controls.append({'step':step,'checkpoint':str(checkpoint),'sampled_decisions':decisions,
                             'core_gpu_hours':cost,'hashes':{str(p):sha(p) for p in [report,*files]}})
    if not controls:raise ValueError('no historical checkpoints available')
    spec={'workflow_freeze_sha256':sha(run/'workflow-freeze.json'),'script_sha256':sha(Path(__file__)),
          'controls':controls,'selection':'nearest available historical sampled decisions and separately nearest core GPU-hours; ties favor earlier step; never use quality to select controls',
          'hardware':{'node':NODE,'allocation':38177,'gpus':list(range(8))},
          'protocol':'full 500-video WEB, promoted generator, raw learned argmax, response margin zero, GPT-4.1, unchanged pauc-f-v1',
          'budget':'no additional training; evaluate missing historical controls after the bounded pilot; never allocate another node'}
    target=run/'matched-reference-freeze.json'
    if target.exists() and read(target)!=spec:raise ValueError('matched comparison contract changed')
    if not target.exists():write(target,spec)
    if args.watch:
        while True:
            state=read(run/'workflow-status.json')['state'];result=run/'result.json'
            if state=='failed':
                write(run/'matched-comparison-status.json',{'state':'not_run_main_workflow_failed'});return
            if state in ('complete','stopped_by_coverage_gate') and result.exists() and read(result)['workflow']['state']==state:break
            time.sleep(30)
    curves=read(run/'learning-curves.json');plan=read(run/'workflow-freeze.json')
    jobs=[]
    for step in plan['evaluation_steps']:
        candidate=run/f'evaluation/web-step{step}'
        if not (candidate/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
        curve=next(r for r in curves if r['optimizer_step']==step)
        for criterion,field in (('sampled_decisions','sampled_decisions'),('core_gpu_hours','core_gpu_hours')):
            control=min(controls,key=lambda c:(abs(c[field]-curve[field]),c['step']))
            jobs.append((step,candidate,curve,criterion,control))
    logs=run/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        TORCHINDUCTOR_COMPILE_THREADS='4',TRITON_CACHE_DIR='/tmp/pace-rl-triton',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    baseline=read(BASELINE/'manifest.json');references={};extra_gpu_hours=0.
    try:
        lock=(ROOT/'outputs/jevany-2b-rl/gpu.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for control in {job[4]['step']:job[4] for job in jobs}.values():
            for name,digest in control['hashes'].items():
                if sha(name)!=digest:raise ValueError('historical control changed: '+name)
            step=control['step'];existing=ANCHOR/f'evaluation/web-step{step}'
            output=existing if (existing/'pauc_f_summary-gpt-4.1-v1.json').exists() else run/f'reference-evaluations/anchor-step{step}'
            if not (output/'pauc_f_summary-gpt-4.1-v1.json').exists():
                allocation=subprocess.check_output(['squeue','-h','-j','38177','-o','%T %N'],text=True).strip()
                if allocation!='RUNNING '+NODE:raise RuntimeError('original allocation unavailable')
                if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    raise RuntimeError('authorized node GPUs are busy; do not interrupt another task')
                output.mkdir(parents=True,exist_ok=True)
                write(run/'matched-comparison-status.json',{'state':'evaluating_historical_control','step':step})
                started=time.monotonic();view=checkpoint_view(run/'reference-models',Path(control['checkpoint']))
                commands=[
                    [PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                     ROOT/'scripts/eval_jevany_streaming.py','--controller',view,'--benchmark','web',
                     '--output-dir',output,'--response-margin','0','--answer-model','promoted',
                     '--max-new-tokens','96','--record-action-logits'],
                    [PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',output]]
                with (logs/f'matched-anchor-step{step}.log').open('a') as handle:
                    for command in commands:subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
                score_full_web(output,SG/'datasets/proactive-videoqa-web/anno.json',ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1/full_web_v1/gpt-cache.sqlite')
                extra_gpu_hours+=8*(time.monotonic()-started)/3600
            verify_web(output)
            protocol=read(output/'manifest.json')
            if any(protocol[key]!=baseline[key] for key in PROTOCOL_KEYS):raise ValueError('historical control protocol differs from SFT')
            references[step]=output
        comparisons=[]
        for step,candidate,curve,criterion,control in jobs:
            reference=references[control['step']]
            comparisons.append({'candidate_step':step,'criterion':criterion,
                'candidate_sampled_decisions':curve['sampled_decisions'],
                'candidate_core_gpu_hours':curve['core_gpu_hours'],
                'reference_step':control['step'],'reference_source':str(reference),
                'reference_sampled_decisions':control['sampled_decisions'],
                'reference_core_gpu_hours':control['core_gpu_hours'],
                'decision_difference':curve['sampled_decisions']-control['sampled_decisions'],
                'gpu_hour_difference':curve['core_gpu_hours']-control['core_gpu_hours'],
                'cost_outside_reference_range':curve['core_gpu_hours']>max(c['core_gpu_hours'] for c in controls),
                'paired_bootstrap':paired_bootstrap(candidate,reference)})
        write(run/'matched-comparisons.json',{'spec_sha256':sha(target),'comparisons':comparisons,
              'additional_reference_evaluation_gpu_hours':extra_gpu_hours,
              'limits':'Nearest available checkpoints, not exact equality; signed exposure/cost differences disclosed. Core costs exclude initialization, failed preparation, evaluation and restarts. One seed; reused WEB development.'})
        write(run/'matched-comparison-status.json',{'state':'complete','comparisons':len(comparisons)})
    except BaseException as exc:
        write(run/'matched-comparison-status.json',{'state':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':main()
