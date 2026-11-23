#!/usr/bin/env python3
"""Durable two-node SFT supervision, evaluation, scoring and final reporting."""
import argparse
import copy
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_pace_estp_sft import ROOT, read, write, dump, sha

SG=ROOT.parent/'stream-gemma';PY=SG/'.venv/bin/python'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def status(run,kind,state,**fields):
    write(run/f'{kind}-supervisor-status.json',dict(state=state,node=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(),**fields))


def checked(argv,log):
    with Path(log).open('a') as handle:
        subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)


def verify_sources(run):
    for path,digest in read(run/'freeze.json')['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen executable changed: '+path)


def initialize(run):
    run.mkdir(parents=True,exist_ok=True)
    for name in ('logs','queue','evaluation','cache','code'): (run/name).mkdir(exist_ok=True)
    if (run/'freeze.json').exists():
        verify_sources(run);return
    files=[ROOT/'scripts'/n for n in ('prepare_pace_estp_sft.py','train_pace_estp_sft.py','run_pace_estp_sft.py',
        'eval_jevany_streaming.py','train_jevany_streaming.py','benchmark_qwen35_2b_training.py','jevany_mixed_reward.py','jevany_rl_reward.py',
        'eval_pace_estp_sft_batched.py')]
    files += [ROOT/'src/pace_clm'/n for n in ('sft_pace_estp.py','jevany_streaming.py','qwen_controller_runtime.py','sft_video.py','rl_inputs.py')]
    for path in files:
        (run/'code'/path.name).write_bytes(path.read_bytes())
    # Keep the exact standard evaluator; the readiness variant only samples raw
    # probabilities instead of argmax. It never applies an action offset.
    source=(ROOT/'scripts/eval_jevany_streaming.py').read_text()
    source=source.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
    needle='            if args.record_action_logits:\n                action_trace.append'
    assert source.count(needle)==1
    source=source.replace(needle,"            _sampled_action = ACTIONS[int(torch.distributions.Categorical(logits=logits.float()).sample().item())]\n"+needle)
    assert source.count('ACTIONS[int(logits.argmax().item())]')==2
    source=source.replace('ACTIONS[int(logits.argmax().item())]','_sampled_action')
    source=source.replace('    torch.cuda.set_device(local_rank)','    torch.cuda.set_device(local_rank)\n    torch.manual_seed(42 + rank)')
    source=source.replace('    protocol = {', '    protocol = {\n        "controller_policy": "sampled raw categorical probabilities, temperature one",')
    (run/'code/eval_readiness.py').write_text(source)
    files.append(run/'code/eval_readiness.py')
    write(run/'freeze.json',dict(training_node='ip-10-3-70-177',training_allocation=38226,
        evaluation_node='ip-10-3-120-167',evaluation_allocation=38177,training_seconds=57600,
        evaluation_hours=list(range(2,17,2)),source_weights={'pace':.5,'estp_it':.5},
        global_batch=32,initialization='fresh pinned Qwen3.5-2B with fresh rank64 alpha128 LoRA and pointer',
        objective='action cross entropy',head_lr=1e-4,lora_lr=2e-6,seed=42,
        generator='frozen promoted Qwen MoE; greedy;96 tokens',judge='gpt-4.1 authorized gateway',
        eval_policy='raw learned argmax; margin zero; self-generated history',readiness_hours=[8,16],
        source_hashes={str(p):sha(p) for p in files},prepared_sha256=sha(run/'prepared.json')))


def parity(run):
    import torch
    from pace_clm.qwen_controller_runtime import load_2b_controller
    from pace_clm.jevany_streaming import PointerHead,controller_logits
    from train_jevany_streaming import EpisodeFrames
    gate=read(run/'first-update-gate.json');checkpoint=Path(gate['checkpoint'])
    runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True));head.eval()
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=128)
    results=[]
    for rank in range(8):
        sample=read(run/f'parity.rank{rank}.json');inputs,vision,options,decision=frames.prepare(sample['row'])
        with torch.no_grad(),runtime._cached_image_features(vision):
            logits=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False)
        old=torch.tensor(sample['logits'],device=logits.device,dtype=torch.float32)
        error=float((logits.float().softmax(-1)-old.softmax(-1)).abs().max())
        results.append(dict(training_rank=rank,probability_max_error=error))
        if error>.005:raise RuntimeError(f'cross-node probability mismatch: {results}')
    write(run/'cross-node-parity.json',dict(passed=True,samples=results,node=socket.gethostname()))


def infer_and_score(run,job,probe=False):
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats,bootstrap
    verify_sources(run)
    suffix='-readiness' if probe else ''
    directory=run/'evaluation'/(job['name']+suffix);directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return
    checkpoint=Path(job['checkpoint']);commit=read(checkpoint/'commit.json')
    if commit['step']!=job['step']:raise ValueError('checkpoint identity mismatch')
    for name,digest in commit['files'].items():
        if name in ('pointer_head.pt','controller_adapter/adapter_model.safetensors') and sha(checkpoint/name)!=digest:
            raise ValueError('evaluation checkpoint changed')
    if probe:
        specs=[]
        for spec in rows(run/'data/readiness-specs.jsonl'):
            for trajectory in range(4):
                clone=copy.deepcopy(spec);clone['probe_parent']=spec['span_id'];clone['trajectory']=trajectory
                clone['span_id']=spec['span_id']+f':trajectory-{trajectory}';specs.append(clone)
        manifest=directory/'manifest.jsonl'
        dump(manifest,[dict(question_id=s['span_id'],video_id=s['video_id'],question=s['question'],
            resolved_video_path=s['video_path'],inference_source_indices=s['source_indices'],
            inference_timestamps=s['timestamps']) for s in specs])
        cohorts={'readiness':[s['span_id'] for s in specs]}
    else:
        specs=rows(run/'data/evaluation-specs.jsonl');manifest=run/'data/evaluation-manifest.jsonl';cohorts=read(run/'data/cohorts.json')
    by_id={s['span_id']:s for s in specs};ids=sorted(by_id)
    write(directory/'selection.json',ids);rollout=directory/'rollout';rollout.mkdir(exist_ok=True)
    if not (rollout/'predictions.jsonl').exists():
        script=run/'code/eval_readiness.py' if probe else ROOT/'scripts/eval_jevany_streaming.py'
        canonical=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',script,
            '--controller',checkpoint,'--validation-manifest',manifest,'--selection-ids',directory/'selection.json',
            '--output-dir',rollout,'--response-margin','0','--answer-model','promoted','--max-new-tokens','96','--record-action-logits']
        config=read(run/'evaluator-config.json') if (run/'evaluator-config.json').exists() else {}
        if not probe and config.get('use_validated_batching'):
            dump(directory/'specs.jsonl',specs)
            command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',
                ROOT/'scripts/eval_pace_estp_sft_batched.py','--checkpoint',checkpoint,'--specs',directory/'specs.jsonl',
                '--output-dir',rollout,'--batch-size','8']
            try:
                checked(command,directory/'inference.log')
            except subprocess.CalledProcessError as exc:
                # Technical failures retain every partial artifact and fall back
                # to the already validated original evaluator on identical cases.
                archive=directory/'batched-recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                archive.parent.mkdir(exist_ok=True);rollout.rename(archive);rollout.mkdir()
                write(directory/'singleton-fallback.json',dict(returncode=exc.returncode,preserved_attempt=str(archive),
                    reason='batched evaluator correctness/runtime failure; identical checkpoint, inputs, raw argmax and generator retained'))
                checked(canonical,directory/'inference.log')
        else:
            checked(canonical,directory/'inference.log')
        checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],directory/'merge.log')
    predictions=rows(rollout/'predictions.jsonl')
    if len(predictions)!=len(ids) or {p['question_id'] for p in predictions}!=set(ids):raise ValueError('incomplete predictions')
    for pred in predictions:
        trace=pred.get('jevany_action_trace',[])
        if len(trace)!=by_id[pred['question_id']]['rollout_frames']:raise ValueError('incomplete trace')
        if not probe and any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):
            raise ValueError('non-argmax action')
    gold=[dict(question_id=p['question_id'],conversation=[dict(role='user',content=by_id[p['question_id']]['question'])],
               answer=by_id[p['question_id']]['gold_answers']) for p in predictions]
    metrics,usage=score_predictions(predictions,gold,directory/'scoring',run/'cache/evaluation.sqlite')
    result=dict(job=job,protocol='native-window PAUC-F proxy omega .5',per_episode=metrics,judge_usage=usage,
        cohorts={name:stats(metrics,values) for name,values in cohorts.items()},
        evaluation_node=socket.gethostname(),new_sft_training_sources=['PACE','ESTP-IT'],
        historical_cohort_note='WEB/EGO train and heldout labels refer to prior RL exposure, not this new SFT run')
    result['evaluation_backend']='batched controller / singleton MoE' if read(rollout/'manifest.json').get('batch_size') else 'original singleton evaluator'
    if probe:
        groups={}
        for spec in specs:
            groups.setdefault(spec['probe_parent'],[]).append(metrics[spec['span_id']]['pauc_f']/100)
        result['reward_groups']=groups
        result['all_zero_groups']=sum(all(v==0 for v in values) for values in groups.values())
        result['flat_groups']=sum(len(set(values))==1 for values in groups.values())
        result['groups']=len(groups)
    else:
        task_metrics={task:stats(metrics,[s['span_id'] for s in specs if s.get('task_type')==task])
                      for task in sorted({s.get('task_type') for s in specs if s.get('task_type')})}
        result['estp_it_explicit_tasks']=task_metrics
        result['estp_it_explicit_task_macro']=sum(v['pauc_f'] for v in task_metrics.values())/len(task_metrics)
        first=run/'evaluation/hour-02/result.json'
        if first.exists() and job['hour']!=2:
            baseline=read(first)['per_episode']
            result['paired_vs_hour02']={name:bootstrap(metrics,baseline,values) for name,values in cohorts.items()}
    write(directory/'result.json',result)


def report(run):
    results=[read(run/'evaluation'/f'hour-{h:02d}'/'result.json') for h in range(2,17,2)]
    lines=['# Fresh Qwen3.5-2B PACE/ESTP-IT supervised training','',
        'Training: allocation 38226. Evaluation: allocation 38177. Fresh base initialization; equal source decision presentations.',
        'Full WEB/EGO; frozen explicit ESTP-IT development cohort; raw argmax; frozen promoted MoE answers; GPT-4.1 judging.',
        'ESTP-IT PAUC-F is a development proxy. No ESTP-Bench or IT test examples were used.', '',
        '| Training hours | Step | WEB | EGO | IT Explicit pooled | IT Explicit task macro |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        c=r['cohorts'];lines.append(f"| {r['job']['hour']} | {r['job']['step']} | {c['web_all']['pauc_f']:.3f} | {c['ego_all']['pauc_f']:.3f} | {c['estp_it_explicit_dev']['pauc_f']:.3f} | {r['estp_it_explicit_task_macro']:.3f} |")
    lines += ['', '## RL readiness', '']
    for hour in (8,16):
        r=read(run/'evaluation'/f'hour-{hour:02d}-readiness/result.json')
        lines.append(f"Hour {hour}: {r['all_zero_groups']}/{r['groups']} all-zero reward groups; {r['flat_groups']}/{r['groups']} tied groups. Four sampled trajectories per episode; this small probe is diagnostic.")
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    write(run/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),training=read(run/'training-complete.json'),
        evaluation_hours=list(range(2,17,2)),readiness_hours=[8,16],report=str(run/'report.md')))


def supervise(run,kind):
    expected='ip-10-3-70-177' if kind=='train' else 'ip-10-3-120-167'
    if socket.gethostname()!=expected:raise RuntimeError('wrong authorized node')
    lock=(run/f'{kind}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if kind=='train':
        for attempt in range(1,11):
            if (run/'training-complete.json').exists():break
            verify_sources(run);status(run,kind,'training',attempt=attempt)
            command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',ROOT/'scripts/train_pace_estp_sft.py','--run',run]
            if (run/'train/latest.json').exists():
                latest=read(run/'train/latest.json')
                if not (Path(latest['checkpoint'])/'commit.json').exists():
                    committed=sorted((run/'train').glob('step-*/commit.json'))
                    if not committed:raise RuntimeError('no committed recovery checkpoint')
                    latest=dict(step=read(committed[-1])['step'],checkpoint=str(committed[-1].parent))
                    write(run/'train/latest.json',latest)
                command+=['--resume']
            try:checked(command,run/'logs/train.log')
            except subprocess.CalledProcessError:
                status(run,kind,'retrying',attempt=attempt);time.sleep(30)
        if not (run/'training-complete.json').exists():raise RuntimeError('training retries exhausted')
        status(run,kind,'complete');return
    # Isolate the parity process so its CUDA allocations disappear before evaluation.
    while not (run/'first-update-gate.json').exists():time.sleep(10)
    if not (run/'cross-node-parity.json').exists():
        try:checked([PY,__file__,'parity','--run',run],run/'logs/parity.log')
        except Exception:
            write(run/'cross-node-parity-failed.json',dict(error=traceback.format_exc()));raise
    failures=Counter()
    while True:
        jobs=sorted([read(p) for p in (run/'queue').glob('hour-*.json')],key=lambda j:j['hour'])
        work=[]
        for job in jobs:
            for probe in ([False,True] if job['hour'] in (8,16) else [False]):
                name=job['name']+('-readiness' if probe else '')
                if not (run/'evaluation'/name/'result.json').exists():work.append((job,probe,name))
        if not work:
            if (run/'training-complete.json').exists() and len(jobs)==8:
                report(run);status(run,kind,'complete');return
            status(run,kind,'waiting',completed=len(jobs));time.sleep(15);continue
        job,probe,name=work[0];status(run,kind,'evaluating',job=name,pending=len(work))
        try:
            # Separate process bounds CUDA and scorer memory across checkpoints.
            checked([PY,__file__,'evaluate','--run',run,'--job',job['name']]+(['--probe'] if probe else []),run/'logs/evaluation.log')
        except Exception:
            failures[name]+=1;status(run,kind,'retrying',job=name,attempt=failures[name])
            if failures[name]>=10:raise
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['initialize','train','eval','parity','evaluate'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job');parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    if args.mode=='initialize':initialize(run)
    elif args.mode=='parity':parity(run)
    elif args.mode=='evaluate':infer_and_score(run,read(run/'queue'/f'{args.job}.json'),args.probe)
    else:supervise(run,args.mode)
