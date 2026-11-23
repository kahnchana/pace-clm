#!/usr/bin/env python3
"""Persistent eight-GPU evaluation; fixed 200-case ESTP subset; no hard deadline."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/eval-speed-design-v1'
PARENT=DESIGN.with_name('base-50-50-b384-prefetch-v2-16h')
PY=ROOT.parent/'stream-gemma/.venv/bin/python'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (out/'run-start.json').exists():write(out/'run-start.json',dict(started_utc=now(),started_epoch=time.time()))
    start_epoch=read(out/'run-start.json')['started_epoch']
    job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID','')
    if socket.gethostname()!='ip-10-3-120-167' or job!='38177' or not step.isdigit():raise RuntimeError('assigned evaluation Slurm step required')
    from run_pace_estp_sft import verify_sources
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats
    from zs_infer import estp_pauc_f as estp
    from judge_exp10_utility import load_gateway_key
    verify_sources(PARENT)
    pilot=read(DESIGN/'pilot/result.json')
    if not pilot['passed']:raise RuntimeError('performance/correctness pilot must pass before a production evaluation')
    for path,digest in read(DESIGN/'pilot/source-freeze.json').items():
        if sha(Path(path))!=digest:raise ValueError('profiled code changed: '+path)
    for path,digest in read(DESIGN/'data/input-hashes.json').items():
        if sha(Path(path))!=digest:raise ValueError('fixed subset/input data changed')
    checkpoint=args.checkpoint.resolve();commit=read(checkpoint/'commit.json')
    for name in ['pointer_head.pt','controller_adapter/adapter_model.safetensors']:
        if sha(checkpoint/name)!=commit['files'][name]:raise ValueError('checkpoint checksum mismatch')
    freeze=dict(checkpoint=str(checkpoint),step=commit['step'],checkpoint_commit_sha256=sha(checkpoint/'commit.json'),
        population=read(DESIGN/'population.json'),input_hashes=read(DESIGN/'data/input-hashes.json'),
        runtime_policy='complete_all_inference_and_judging',hard_time_limit=False,single_model_load_per_gpu=True,controller_policy='unmodified singleton raw argmax',response_margin=0,
        generator='promoted frozen MoE greedy, 96 tokens',judge='gpt-4.1',estp_scope='Fixed 25-per-task diagnostic subset of 200, not the official full 851-case result')
    if (out/'freeze.json').exists() and read(out/'freeze.json')!=freeze:raise ValueError('existing run protocol differs')
    write(out/'freeze.json',freeze);write(out/'status.json',dict(phase='combined_inference',utc=now(),step=commit['step']))
    rollout=out/'rollout';rollout.mkdir(exist_ok=True)
    if not (out/'inference-start.json').exists():write(out/'inference-start.json',dict(started_utc=now(),started_epoch=time.time()))
    if not (rollout/'predictions.jsonl').exists():
        argv=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',ROOT/'scripts/eval_pace_sft_fast_native.py',
            '--controller',checkpoint,'--validation-manifest',DESIGN/'data/manifest.jsonl','--selection-ids',DESIGN/'data/selection-web-first.json',
            '--output-dir',rollout,'--response-margin','0','--answer-model','promoted','--max-new-tokens','96','--record-action-logits','--direct-gpu-load','--cached-controller-inputs']
        with (out/'inference.log').open('a') as handle:subprocess.run(list(map(str,argv)),stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        with (out/'merge.log').open('a') as handle:subprocess.run([str(PY),str(ROOT/'scripts/merge_jevany_benchmark.py'),'--input-dir',str(rollout)],stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    if not (out/'inference-timing.json').exists():write(out/'inference-timing.json',dict(seconds=time.time()-read(out/'inference-start.json')['started_epoch'],finished_utc=now()))
    inference_seconds=read(out/'inference-timing.json')['seconds']
    specs={s['span_id']:s for s in rows(DESIGN/'data/specs.jsonl')};predictions=rows(rollout/'predictions.jsonl')
    if len(predictions)!=1026 or {p['question_id'] for p in predictions}!=set(specs):raise ValueError('incomplete fixed population')
    for pred in predictions:
        trace=pred['jevany_action_trace']
        if len(trace)!=specs[pred['question_id']]['rollout_frames']:raise ValueError('incomplete episode')
        if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):raise ValueError('non-argmax action')
    write(out/'status.json',dict(phase='gpt_scoring',utc=now()));cohorts=read(PARENT/'data/cohorts.json');results={};score_start=time.monotonic()
    for suite in ['web','ego']:
        directory=out/suite;directory.mkdir(exist_ok=True)
        if (directory/'result.json').exists():
            results[suite]=read(directory/'result.json');continue
        ids=set(cohorts[suite+'_all']);selected=[p for p in predictions if p['question_id'] in ids]
        gold=[dict(question_id=p['question_id'],conversation=[dict(role='user',content=specs[p['question_id']]['question'])],answer=specs[p['question_id']]['gold_answers']) for p in selected]
        tick=time.monotonic();metrics,usage=score_predictions(selected,gold,directory/'scoring',out/'cache/evaluation.sqlite')
        results[suite]=dict(historical_cohort_note='Train/heldout names refer to prior RL identities; this fresh SFT trained only PACE and ESTP-IT.',cohorts={k:stats(metrics,v) for k,v in cohorts.items() if k.startswith(suite+'_')},per_episode=metrics,judge_usage=usage,scoring_seconds=time.monotonic()-tick)
        write(directory/'result.json',results[suite])
    directory=out/'estp-bench-explicit-200';scoring=directory/'scoring';scoring.mkdir(parents=True,exist_ok=True)
    records=rows(DESIGN/'data/estp-bench-explicit-200.jsonl');manifest=scoring/'explicit_manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    by_qid={f"{specs[p['question_id']]['video_id']}/{specs[p['question_id']]['clip_id']}/{specs[p['question_id']]['qa_index']}":(p,specs[p['question_id']]) for p in predictions if specs[p['question_id']].get('data_source')=='estp_bench'}
    converted=[]
    for record in records:
        pred,spec=by_qid[record['question_id']]
        fires=[dict(role='assistant',content=t['content'],absolute_time=float(t['time'])+spec['clip_start_time'],time=float(t['time'])+spec['clip_start_time']-record['eval_start_time'],fire_id=i) for i,t in enumerate(pred['model_response_list'])]
        converted.append(dict(question_id=record['question_id'],task_type=record['task_type'],eval_start_time=record['eval_start_time'],eval_end_time=record['eval_end_time'],model_response_list=fires))
    (scoring/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in converted));tick=time.monotonic();load_gateway_key()
    estp.cmd_judge(SimpleNamespace(manifest=manifest,predictions=scoring/'predictions.jsonl',official_scorer=estp.OFFICIAL_SCORER,pred_dir=scoring,judge_model='gpt-4.1',preflight_only=False,concurrency=16,max_attempts=5,request_timeout=120.))
    summary=read(scoring/'pauc_f_summary-gpt-4.1.json')
    if not summary['headline_eligible'] or len(summary['per_task'])!=8:raise ValueError('Incomplete judged ESTP subset')
    results['estp-bench-explicit-200']=dict(summary=summary,scoring_seconds=time.monotonic()-tick,scope='Fixed 200-case task-stratified subset, not official full benchmark')
    write(directory/'result.json',results['estp-bench-explicit-200'])
    write(out/'metrics-summary.json',dict(step=commit['step'],utc=now(),inference_seconds=inference_seconds,scoring_seconds=sum(r['scoring_seconds'] for r in results.values()),total_seconds=time.time()-start_epoch,
        population=read(DESIGN/'population.json'),scores={'web':results['web']['cohorts']['web_all']['pauc_f'],'ego':results['ego']['cohorts']['ego_all']['pauc_f'],'estp_bench_explicit_200':summary['all']['omega_0.5']}))
    from report_pace_sft_fast_eval import report
    report(out)
    write(out/'status.json',dict(phase='complete',utc=now()))


if __name__=='__main__':main()
