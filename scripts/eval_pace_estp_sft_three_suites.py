#!/usr/bin/env python3
"""Evaluate one pinned SFT checkpoint on WEB, EGO and ESTP-Bench Explicit."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

from prepare_pace_estp_sft import ROOT,read,write,sha,dump
from run_pace_estp_sft import verify_sources
from jevany_mixed_reward import score_predictions
from eval_jevany_mixed_queue import stats

PY=ROOT.parent/'stream-gemma/.venv/bin/python'
PARENT=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h'
SOURCE=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128'
OUT=None
CHECKPOINT=None
STEP=None


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def now():return datetime.now(timezone.utc).isoformat()
def state(suite,phase,**kw):write(OUT/suite/'status.json',dict(phase=phase,utc=now(),node=socket.gethostname(),step=STEP,**kw))


def checked(argv,log):
    with log.open('a') as handle:subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)


def counts(specs):
    return dict(episodes=len(specs),unique_video_ids=len({s['video_id'] for s in specs}),
        unique_media_files=len({s['video_path'] for s in specs}),decisions=sum(s['rollout_frames'] for s in specs))


def validate(predictions,specs):
    by_id={s['span_id']:s for s in specs}
    if len(predictions)!=len(by_id) or {p['question_id'] for p in predictions}!=set(by_id):raise ValueError('incomplete or duplicate prediction population')
    for pred in predictions:
        trace=pred.get('jevany_action_trace',[])
        if len(trace)!=by_id[pred['question_id']]['rollout_frames']:raise ValueError('incomplete saved trace')
        if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):raise ValueError('non-argmax action')


def infer(suite,specs,manifest):
    directory=OUT/suite;directory.mkdir(exist_ok=True);rollout=directory/'rollout';rollout.mkdir(exist_ok=True)
    selection=sorted(s['span_id'] for s in specs);write(directory/'selection.json',selection)
    timing=directory/'inference-timing.json'
    if not (rollout/'predictions.jsonl').exists():
        state(suite,'inference',**counts(specs));started=time.monotonic();utc=now()
        checked([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',
            ROOT/'scripts/eval_jevany_streaming.py','--controller',CHECKPOINT,'--validation-manifest',manifest,
            '--selection-ids',directory/'selection.json','--output-dir',rollout,'--response-margin','0',
            '--answer-model','promoted','--max-new-tokens','96','--record-action-logits'],directory/'inference.log')
        checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],directory/'merge.log')
        write(timing,dict(seconds=time.monotonic()-started,started_utc=utc,finished_utc=now(),scope='external wall time including model loading and merge; excludes GPT scoring'))
    predictions=rows(rollout/'predictions.jsonl');validate(predictions,specs)
    return predictions,read(timing)


def native_score(suite,predictions,specs,cohorts,timing):
    directory=OUT/suite;state(suite,'gpt_scoring',**counts(specs));start=time.monotonic();utc=now()
    by_id={s['span_id']:s for s in specs}
    gold=[dict(question_id=p['question_id'],conversation=[dict(role='user',content=by_id[p['question_id']]['question'])],answer=by_id[p['question_id']]['gold_answers']) for p in predictions]
    metrics,usage=score_predictions(predictions,gold,directory/'scoring',OUT/'cache/evaluation.sqlite')
    result=dict(step=STEP,checkpoint=str(CHECKPOINT),suite=suite,**counts(specs),
        cohorts={name:stats(metrics,ids) for name,ids in cohorts.items()},per_episode=metrics,judge_usage=usage,
        inference_timing=timing,scoring_timing=dict(seconds=time.monotonic()-start,started_utc=utc,finished_utc=now()),
        raw_argmax_verified=True,judge='gpt-4.1',generator='frozen promoted MoE, singleton greedy, 96 tokens',
        training_sources='Fresh mixed SFT used PACE and ESTP-IT only; train/heldout cohort labels below refer to historical RL identities, not this SFT run')
    write(directory/'result.json',result);state(suite,'complete',cohorts=result['cohorts'])
    return result


def ego():
    directory=OUT/'ego';directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return read(directory/'result.json')
    cohorts=read(PARENT/'data/cohorts.json');ids=set(cohorts['ego_all'])
    specs=[s for s in rows(PARENT/'data/evaluation-specs.jsonl') if s['span_id'] in ids]
    predictions,timing=infer('ego',specs,PARENT/'data/evaluation-manifest.jsonl')
    return native_score('ego',predictions,specs,{k:v for k,v in cohorts.items() if k.startswith('ego_')},timing)


def web():
    directory=OUT/'web';directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return read(directory/'result.json')
    cohorts=read(PARENT/'data/cohorts.json');ids=set(cohorts['web_all']);specs=[s for s in rows(PARENT/'data/evaluation-specs.jsonl') if s['span_id'] in ids]
    predictions,timing=infer('web',specs,PARENT/'data/evaluation-manifest.jsonl')
    return native_score('web',predictions,specs,{k:v for k,v in cohorts.items() if k.startswith('web_')},timing)


def estp_bench():
    from zs_infer import estp_pauc_f as estp
    from judge_exp10_utility import load_gateway_key
    suite='estp-bench-explicit';directory=OUT/suite;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return read(directory/'result.json')
    manifest=estp.build_manifest(estp.DEFAULT_ANNOTATION,estp.DEFAULT_VIDEO_ROOT,horizon='clip');records=manifest['records']
    all_specs={f"{s['video_id']}/{s['clip_id']}/{s['qa_index']}":s for s in rows(SOURCE/'data/evaluation-specs.jsonl') if s.get('data_source')=='estp_bench'}
    chosen={r['question_id']:all_specs[r['question_id']] for r in records};specs=list(chosen.values())
    if len(specs)!=851:raise ValueError('expected unchanged 851-case ESTP-Bench Explicit split')
    scoring=directory/'estp_pauc_explicit';estp.write_manifest(manifest,scoring)
    write(directory/'protocol.json',dict(scorer=estp.SCORER_VERSION,horizon='clip',match_window='strict',omega=.5,
        annotation_sha256=manifest['summary']['annotation_sha256'],scorer_sha256=sha(estp.__file__),
        official_scorer_sha256=sha(estp.OFFICIAL_SCORER),source_specs_sha256=sha(SOURCE/'data/evaluation-specs.jsonl'),**counts(specs)))
    predictions,timing=infer(suite,specs,SOURCE/'data/evaluation-manifest.jsonl');by_span={s['span_id']:s for s in specs}
    by_qid={f"{by_span[p['question_id']]['video_id']}/{by_span[p['question_id']]['clip_id']}/{by_span[p['question_id']]['qa_index']}":(p,by_span[p['question_id']]) for p in predictions}
    converted=[]
    for record in records:
        pred,spec=by_qid[record['question_id']]
        fires=[dict(role='assistant',content=t['content'],absolute_time=float(t['time'])+spec['clip_start_time'],
            time=float(t['time'])+spec['clip_start_time']-record['eval_start_time'],fire_id=i) for i,t in enumerate(pred['model_response_list'])]
        if any(not f['content'].strip() for f in fires):raise ValueError('empty generated answer')
        converted.append(dict(question_id=record['question_id'],task_type=record['task_type'],eval_start_time=record['eval_start_time'],eval_end_time=record['eval_end_time'],model_response_list=fires))
    dump(scoring/'predictions.jsonl',converted);state(suite,'gpt_scoring',**counts(specs));start=time.monotonic();utc=now();load_gateway_key()
    estp.cmd_judge(SimpleNamespace(manifest=scoring/'explicit_manifest.jsonl',predictions=scoring/'predictions.jsonl',official_scorer=estp.OFFICIAL_SCORER,
        pred_dir=scoring,judge_model='gpt-4.1',preflight_only=False,concurrency=16,max_attempts=5,request_timeout=120.))
    summary=read(scoring/'pauc_f_summary-gpt-4.1.json')
    if not summary['headline_eligible'] or len(summary['per_task'])!=8:raise ValueError('incomplete ESTP-Bench Explicit GPT scores')
    result=dict(step=STEP,checkpoint=str(CHECKPOINT),suite=suite,**counts(specs),summary=summary,raw_argmax_verified=True,
        inference_timing=timing,scoring_timing=dict(seconds=time.monotonic()-start,started_utc=utc,finished_utc=now()))
    write(directory/'result.json',result);state(suite,'complete',pauc_f=summary['all']['omega_0.5']);return result


def retry(function,suite):
    for attempt in range(1,6):
        try:return function()
        except Exception as exc:
            state(suite,'retrying',attempt=attempt,error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
            if attempt==5:raise
            time.sleep(15)


def report():
    result={s:read(OUT/s/'result.json') for s in ['web','ego','estp-bench-explicit']}
    lines=[f'# Checkpoint-{STEP} evaluation: WEB, EGO and ESTP-Bench Explicit','',
        'Same SFT checkpoint, raw argmax/margin zero, promoted MoE generator and authorized GPT-4.1 judging. ESTP-IT is excluded.','',
        '| Evaluation set | Episodes | Unique video IDs | Distinct media files | Inference hours | GPT/scoring minutes | PAUC-F |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for suite in ['web','ego','estp-bench-explicit']:
        r=result[suite];score=r['summary']['all']['omega_0.5'] if suite=='estp-bench-explicit' else r['cohorts'][suite+'_all']['pauc_f']
        lines.append(f"| {suite} | {r['episodes']} | {r['unique_video_ids']} | {r['unique_media_files']} | {r['inference_timing']['seconds']/3600:.3f} | {r['scoring_timing']['seconds']/60:.2f} | {score:.3f} |")
    lines+=['','All inference times include model loading and merging. GPT/scoring time includes request construction, judging and deterministic metric calculation. Historical WEB/EGO train/heldout labels refer to earlier RL identities; this fresh SFT trained only PACE and ESTP-IT.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');write(OUT/'complete.json',dict(utc=now(),step=STEP,report=str(OUT/'report.md')))


def main():
    global OUT,CHECKPOINT,STEP
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();OUT=args.output_dir.resolve();CHECKPOINT=args.checkpoint.resolve();STEP=args.step
    OUT.mkdir(parents=True,exist_ok=True)
    for suite in ['ego','web','estp-bench-explicit','cache']:(OUT/suite).mkdir(exist_ok=True)
    lock=(OUT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if socket.gethostname()!='ip-10-3-120-167' or os.environ.get('SLURM_JOB_ID')!='38177':raise RuntimeError('authorized evaluation node/allocation only')
    verify_sources(PARENT)
    commit=read(CHECKPOINT/'commit.json')
    if commit['step']!=STEP:raise ValueError('checkpoint identity mismatch')
    for name in ['pointer_head.pt','controller_adapter/adapter_model.safetensors']:
        if sha(CHECKPOINT/name)!=commit['files'][name]:raise ValueError('checkpoint checksum mismatch')
    freeze=dict(checkpoint=str(CHECKPOINT),step=STEP,checkpoint_commit_sha256=sha(CHECKPOINT/'commit.json'),source_sha256=sha(Path(__file__)),
        order=['web','ego','estp-bench-explicit'],ego='fresh inference from the pinned checkpoint',response_margin=0,
        generator='promoted MoE singleton greedy, 96 tokens',judge='gpt-4.1',node=socket.gethostname(),allocation=38177)
    if (OUT/'freeze.json').exists() and read(OUT/'freeze.json')!=freeze:raise ValueError('corrected workflow changed')
    write(OUT/'freeze.json',freeze)
    retry(web,'web')
    retry(ego,'ego')
    retry(estp_bench,'estp-bench-explicit')
    report()


if __name__=='__main__':main()
