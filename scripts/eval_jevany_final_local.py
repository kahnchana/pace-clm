#!/usr/bin/env python3
"""Final immutable-checkpoint evaluation on training allocation 38177 after release."""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from jevany_rl_reward import ROOT,SG,write,sha
from jevany_mixed_reward import score_predictions
from run_jevany_rl_full_current_node import checkpoint_view
from run_jevany_e2_full_web import seed_judge_cache
from pace_clm.rl_anchor import verify_committed_checkpoint

PY=SG/'.venv/bin/python'
NODE='ip-10-3-120-167'


def read(p):return json.loads(Path(p).read_text())


def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def stats(metrics,ids):
    m=[metrics[q] for q in ids];u=sum(x['num_value_adding_fires'] for x in m);f=sum(x['num_fires'] for x in m)
    g=sum(x['num_gold_turns'] for x in m);area=sum(x['pauc_recall_component']/100*x['num_gold_turns'] for x in m)
    p=u/f if f else 0.;r=area/g if g else 0.
    return {'episodes':len(m),'pauc_f':100*2*p*r/(p+r) if p+r else 0.,
            'mean_episode_pauc_f':float(np.mean([x['pauc_f'] for x in m])),
            'fire_precision':100*p,'temporal_recall':100*r,'useful':u,'wasted':f-u,'fires':f,'gold_events':g}


def bootstrap(candidate,baseline,ids):
    def arrays(metrics):return np.array([[metrics[q]['num_value_adding_fires'],metrics[q]['num_fires'],
        metrics[q]['pauc_recall_component']/100*metrics[q]['num_gold_turns'],metrics[q]['num_gold_turns']] for q in ids],float)
    def metric(x):
        u,f,a,g=x.T;p=np.divide(u,f,out=np.zeros_like(u),where=f>0);r=np.divide(a,g,out=np.zeros_like(a),where=g>0)
        return 100*np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)
    a,b=arrays(candidate),arrays(baseline);draws=np.random.default_rng(42).integers(0,len(ids),(5000,len(ids)))
    delta=metric(a[draws].sum(1))-metric(b[draws].sum(1))
    return {'difference_points':float(metric(a.sum(0)[None])[0]-metric(b.sum(0)[None])[0]),
            'paired_95_ci':np.quantile(delta,[.025,.975]).tolist(),'paired_episodes':len(ids),'replicates':5000,
            'scope':'paired episode sampling, conditional on saved judgments; selected/reused data'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    if socket.gethostname()!=NODE:raise RuntimeError('evaluation allocation only')
    allocation=subprocess.check_output(['squeue','-h','-j','38177','-o','%T %N'],text=True).strip()
    if allocation!='RUNNING '+NODE:raise RuntimeError('evaluation allocation unavailable')
    # Reserve evaluation ownership now; wait for the trainer process to exit.
    lock=(run/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(run/'evaluation-status.json',{'state':'waiting_for_training_release',
        'node':NODE,'allocation':38177,'utc':datetime.now(timezone.utc).isoformat()})
    while True:
        state=read(run/'workflow-status.json')['state']
        if state in ('failed','stopped_by_coverage_gate'):
            write(run/'evaluation-status.json',{'state':'training_failed','node':NODE});return
        if state=='complete':return
        if state=='evaluating_final':break
        time.sleep(10)
    routing=read(run/'final-evaluation-routing.json')
    if routing['node']!=NODE or routing['evaluator_sha256']!=sha(Path(__file__)):
        raise ValueError('final evaluation routing/source changed')
    gpu_lock=(ROOT/'outputs/jevany-2b-rl/gpu.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(names)!=8 or any('H200' not in x for x in names):raise RuntimeError('eight assigned H200s required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('evaluation GPUs busy')
    plan=read(run/'workflow-freeze.json');specs={s['span_id']:s for s in rows(run/'data/evaluation-specs.jsonl')}
    cohorts=read(run/'data/cohorts.json');fast=read(run/'data/fast-cohorts.json')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',TORCHINDUCTOR_COMPILE_THREADS='4',
        TRITON_CACHE_DIR='/tmp/pace-mixed-eval-triton',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    def status(state,**fields):write(run/'evaluation-status.json',{'state':state,'utc':datetime.now(timezone.utc).isoformat(),'node':NODE,**fields})
    def command(argv,log):
        for path,digest in plan['source_hashes'].items():
            if sha(path)!=digest:raise ValueError('frozen source changed: '+path)
        with log.open('a') as f:subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        status('waiting')
        while True:
            jobs=[read(p) for p in (run/'queue').glob('*.json') if not (run/'evaluation'/p.stem/'result.json').exists()]
            # Stale fast probes can be skipped; full checkpoints and baselines are retained.
            fast_jobs=[j for j in jobs if j['kind']=='fast']
            for j in sorted(fast_jobs,key=lambda j:j['step'])[:-1]:
                d=run/'evaluation'/j['name'];d.mkdir(parents=True,exist_ok=True);write(d/'result.json',{'skipped':'superseded fast checkpoint','job':j})
            jobs=[j for j in jobs if not (run/'evaluation'/j['name']/'result.json').exists()]
            if not jobs:
                state=read(run/'workflow-status.json')['state'] if (run/'workflow-status.json').exists() else None
                if state in ('complete','stopped_by_coverage_gate','failed'):
                    status('complete' if state!='failed' else 'training_failed');return
                time.sleep(10);continue
            job=min(jobs,key=lambda j:(j['priority'],j['step'],j['name']))
            if job['kind']!='full' or job['step']!=len(plan['schedule']):raise ValueError('local evaluator is reserved for the final checkpoint')
            directory=run/'evaluation'/job['name'];directory.mkdir(parents=True,exist_ok=True)
            chosen={'gate':cohorts['gate']} if job['kind'] in ('gate','baseline_smoke') else fast if job['kind']=='fast' else cohorts
            ids=sorted({q for values in chosen.values() for q in values})
            write(directory/'selection.json',ids);checkpoint=Path(job['checkpoint'])
            if job['step']:verify_committed_checkpoint(checkpoint,job['step'])
            if job['kind']=='gate':
                status('cross_node_parity',job=job['name'])
                command([PY,ROOT/'scripts/probe_jevany_mixed_parity.py','--run',run],directory/'parity.log')
            rollout=directory/'rollout';rollout.mkdir(exist_ok=True)
            # Retain protocol-matched historical WEB trajectories for the two baselines.
            if job['name'] in ('baseline-sft','baseline-prior') and not (rollout/'predictions.jsonl').exists():
                source=(ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web' if job['name']=='baseline-sft'
                        else ROOT/'outputs/jevany-2b-rl/episode-rloo-v2/evaluation/web-step23')
                protocol=read(source/'manifest.json')
                required={'response_margin':0.,'answer_model':'promoted','history':'self-generated',
                    'max_new_tokens':96,'window_frames':32,'controller_history':'j1-original',
                    'memory_mode':'last-response','memory_select':'aged-out','memory_instruction':'v2',
                    'answer_prompt':'pace-forced-response','prefix':'direct'}
                if any(protocol.get(k)!=v for k,v in required.items()):raise ValueError('incompatible baseline cache')
                if protocol['source_sha256']!=sha(SG/'datasets/proactive-videoqa-web/anno.json'):raise ValueError('baseline annotation changed')
                if protocol['original_generator_adapter_sha256']!=sha(SG/'outputs/lora-qwen-w1-long-neg/adapter_model.safetensors'):raise ValueError('baseline generator changed')
                if protocol['controller_adapter_sha256']!=sha(checkpoint/'controller_adapter/adapter_model.safetensors') or protocol['pointer_head_sha256']!=sha(checkpoint/'pointer_head.pt'):raise ValueError('baseline checkpoint mismatch')
                cached={'web:'+p['question_id']:p for p in rows(source/'predictions.jsonl')}
                for rank in range(8):
                    path=rollout/f'predictions.rank{rank:02d}.jsonl'
                    if path.exists():continue
                    records=[{**cached[q],'question_id':q} for q in ids[rank::8] if q in cached]
                    path.write_text(''.join(json.dumps(p)+'\n' for p in records))
                seed_judge_cache(source,run/'cache/evaluation.sqlite')
            if not (rollout/'predictions.jsonl').exists():
                status('generating',job=job['name'],episodes=len(ids))
                view=checkpoint_view(directory,checkpoint)
                command([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',ROOT/'scripts/eval_jevany_streaming.py',
                         '--controller',view,'--validation-manifest',run/'data/evaluation-manifest.jsonl',
                         '--selection-ids',directory/'selection.json','--output-dir',rollout,'--response-margin','0',
                         '--answer-model','promoted','--max-new-tokens','96','--record-action-logits'],directory/'rollout.log')
                command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],directory/'merge.log')
            predictions=rows(rollout/'predictions.jsonl')
            if [p['question_id'] for p in predictions]!=ids:raise ValueError('incomplete evaluation')
            for pred in predictions:
                if 'jevany_action_trace' in pred:
                    trace=pred['jevany_action_trace']
                    if len(trace)!=specs[pred['question_id']]['rollout_frames']:raise ValueError('incomplete causal episode')
                    for item in trace:
                        if item['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:item['logits'][i])]:raise ValueError('non-argmax action')
            gold=[{'question_id':q,'conversation':[{'role':'user','content':specs[q]['question']}],'answer':specs[q]['gold_answers']} for q in ids]
            status('scoring',job=job['name'],episodes=len(ids))
            metrics,usage=score_predictions(predictions,gold,directory/'scoring',run/'cache/evaluation.sqlite')
            summary={name:stats(metrics,values) for name,values in chosen.items()}
            result={'execution_node':NODE,'allocation':38177,'job':job,'cohorts':summary,'per_episode':metrics,'judge_usage':usage,
                    'seen_data_label':'WEB/EGO training cohorts deliberately enter RL; heldout cohorts excluded from RL; historical reused data'}
            baseline_path=run/'evaluation/baseline-sft/result.json'
            if job['step']>1 and baseline_path.exists():
                base=read(baseline_path)['per_episode']
                result['paired_vs_sft']={name:bootstrap(metrics,base,values) for name,values in chosen.items()}
                if job['kind'] in ('full','fast'):
                    reasons=[]
                    for name in ('web_heldout','ego_heldout','ours_heldout'):
                        current=summary[name];reference=stats(base,chosen[name])
                        if current['fires']<.1*reference['fires']:reasons.append(name+':near_silence')
                        if job['kind']=='full' and current['useful']<.9*reference['useful']:
                            prior=[read(p) for p in (run/'evaluation').glob('full-step*/result.json')]
                            prior=[p for p in prior if 'cohorts' in p and p['job']['step']<job['step']]
                            if prior:
                                previous=max(prior,key=lambda p:p['job']['step'])
                                if previous['cohorts'][name]['useful']<.9*stats(base,cohorts[name])['useful']:
                                    reasons.append(name+':coverage_below_90_percent_twice')
                    if reasons:write(run/'stop-request.json',{'reasons':reasons,'evaluated_step':job['step'],'evaluated_exposure':job['exposure']})
            write(directory/'result.json',result)
            command([PY,ROOT/'scripts/report_jevany_mixed_rl.py','--run',run],directory/'report.log')
            status('waiting',last_completed=job['name'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
