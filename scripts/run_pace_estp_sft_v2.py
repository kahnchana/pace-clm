#!/usr/bin/env python3
"""Fresh, efficient 50:50 SFT with profiled fixed-step evaluation milestones."""
import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

from prepare_pace_estp_sft import ROOT,read,write,sha
from run_pace_estp_sft import checked,parity,infer_and_score,rows

PY=ROOT.parent/'stream-gemma/.venv/bin/python'
OLD=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-seed42-16h'


def status(run,kind,state,**kw):
    write(run/f'{kind}-supervisor-status.json',dict(state=state,node=socket.gethostname(),utc=datetime.now(timezone.utc).isoformat(),**kw))


def prepare(run):
    if (run/'freeze.json').exists():
        verify(run);return
    for name in ['logs','queue','evaluation','cache','code','profiles']:(run/name).mkdir(parents=True,exist_ok=True)
    for path,digest in read(OLD/'prepared.json')['hashes'].items():
        if sha(path)!=digest:raise ValueError('prepared data changed: '+path)
    (run/'data').symlink_to(OLD/'data',target_is_directory=True)
    shutil.copyfile(OLD/'prepared.json',run/'prepared.json')
    shutil.copyfile(OLD/'code/eval_readiness.py',run/'code/eval_readiness.py')
    shutil.copyfile(OLD/'evaluator-config.json',run/'evaluator-config.json')
    config=read(run/'evaluator-config.json');config['validation']=str(OLD/'smoke/batched-validation.json');write(run/'evaluator-config.json',config)
    previous=read(OLD/'freeze.json');files=dict(previous['source_hashes'])
    # The stopped run remains immutable; new executable names describe the new recipe.
    for path in [ROOT/'scripts/train_pace_estp_sft_batched.py',Path(__file__).resolve(),ROOT/'scripts/run_pace_estp_sft_v2.sh',run/'code/eval_readiness.py',run/'evaluator-config.json']:
        files[str(path)]=sha(path)
        if path.suffix=='.py' and path.parent!=run/'code':shutil.copyfile(path,run/'code'/path.name)
    write(run/'freeze.json',dict(source_hashes=files,prepared_sha256=sha(run/'prepared.json'),data_origin=str(OLD/'data'),
        training_node='ip-10-3-70-177',training_allocation=38226,evaluation_node='ip-10-3-120-167',evaluation_allocation=38177,
        global_batch=384,source_weights={'pace':.5,'estp_it':.5},physical_batch_candidates=[48,24],
        head_lr=.00035,lora_lr=.000007,lr_warmup_steps=50,optimizer='AdamW',weight_decay=.01,gradient_clip=1.,
        seed=42,profile_updates=10,profile_warm_updates=3,initialization='fresh pinned Qwen3.5-2B; fresh decoder LoRA and pointer head',
        evaluation_hours=list(range(2,17,2)),initialization_evaluation='small smoke only; no full checkpoint-zero suites',
        readiness_hours=[8,16],training_budget='8 times profiled two-hour step interval; approximately 16 hours',
        generator='unchanged frozen promoted MoE singleton greedy, 96 tokens',judge='gpt-4.1 authorized gateway',
        evaluation_policy='raw learned argmax, response margin zero, causal self-generated history',backend='packed CPU/Gloo gradient average'))


def verify(run):
    for path,digest in read(run/'freeze.json')['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen executable or environment changed: '+path)


def profile(run):
    if socket.gethostname()!='ip-10-3-70-177':raise RuntimeError('training node only')
    lock=(run/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'step-schedule.json').exists():return
    results=[]
    for physical in [48,24]:
        directory=run/'profiles'/f'batch-{physical}';directory.mkdir(exist_ok=True)
        if not (directory/'data').exists():(directory/'data').symlink_to(run/'data',target_is_directory=True)
        shutil.copyfile(run/'freeze.json',directory/'freeze.json')
        existing=sorted(directory.glob('profile.rank*.json'))
        if len(existing)!=8:
            if (directory/'train/latest.json').exists():
                archive=directory.with_name(directory.name+'-attempt-'+datetime.now(timezone.utc).strftime('%H%M%S'))
                directory.rename(archive);directory.mkdir();(directory/'data').symlink_to(run/'data',target_is_directory=True)
                shutil.copyfile(run/'freeze.json',directory/'freeze.json')
            status(run,'train','profiling',physical_batch=physical)
            try:
                checked([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',
                    ROOT/'scripts/train_pace_estp_sft_batched.py','--run',directory,'--profile','--profile-updates','10',
                    '--physical-batch',str(physical)],directory/'train.log')
            except subprocess.CalledProcessError as exc:
                write(directory/'failed.json',dict(returncode=exc.returncode));continue
        ranks=[read(directory/f'profile.rank{i}.json') for i in range(8)]
        durations=ranks[0]['measured_step_seconds']
        mean=sum(durations)/len(durations)
        results.append(dict(physical_batch=physical,mean_step_seconds=mean,examples_per_second=384/mean,
            max_reserved_gib=max(r['peak_reserved_gib'] for r in ranks),profile=str(directory)))
    if not results:raise RuntimeError('no validated physical batch profile completed')
    best=min(results,key=lambda r:r['mean_step_seconds']);interval=max(1,round(7200/best['mean_step_seconds']))
    write(run/'step-schedule.json',dict(interval_steps=interval,total_steps=8*interval,
        checkpoints=[i*interval for i in range(1,9)],evaluation_hours=list(range(2,17,2)),
        physical_batch=best['physical_batch'],profile_seconds_per_update=best['mean_step_seconds'],
        estimated_training_seconds=8*interval*best['mean_step_seconds'],profiles=results,
        rule='fixed step schedule frozen before fresh production initialization; profiling weights discarded'))
    status(run,'train','profile_complete',schedule=read(run/'step-schedule.json'))


def train(run):
    if socket.gethostname()!='ip-10-3-70-177':raise RuntimeError('training node only')
    profile(run)
    lock=(run/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    physical=read(run/'step-schedule.json')['physical_batch']
    for attempt in range(1,11):
        if (run/'user-stopped.json').exists():return
        if (run/'training-complete.json').exists():status(run,'train','complete');return
        verify(run);status(run,'train','training',attempt=attempt,physical_batch=physical)
        command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--',
            ROOT/'scripts/train_pace_estp_sft_batched.py','--run',run,'--physical-batch',str(physical)]
        if (run/'train/latest.json').exists():
            commits=sorted((run/'train').glob('step-*/commit.json'))
            if not commits:raise RuntimeError('no committed checkpoint for recovery')
            latest=commits[-1];write(run/'train/latest.json',dict(step=read(latest)['step'],checkpoint=str(latest.parent)))
            command+=['--resume']
        try:checked(command,run/'logs/train.log')
        except subprocess.CalledProcessError:
            status(run,'train','retrying',attempt=attempt);time.sleep(30)
    if not (run/'training-complete.json').exists():raise RuntimeError('training retries exhausted')
    status(run,'train','complete')


def report(run):
    values=[read(run/'evaluation'/f'hour-{hour:02d}/result.json') for hour in range(2,17,2)]
    lines=['# Prefetched PACE / ESTP-IT supervised controller','',
        'Fresh Qwen3.5-2B, global batch 384, equal PACE/ESTP-IT decisions. Profiling weights discarded.',
        'Full WEB/EGO and frozen IT Explicit development; native windows, raw argmax, frozen singleton MoE generator, GPT-4.1 judging.',
        'Checkpoint zero was saved and smoke-tested only; no full random-head baseline evaluation.', '',
        '| Estimated training hour | Step | Actual training hours | WEB | EGO | IT Explicit pooled | IT Explicit task macro |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in values:
        c=r['cohorts'];j=r['job'];lines.append(f"| {j['hour']} | {j['step']} | {j['training_seconds']/3600:.2f} | {c['web_all']['pauc_f']:.3f} | {c['ego_all']['pauc_f']:.3f} | {c['estp_it_explicit_dev']['pauc_f']:.3f} | {r['estp_it_explicit_task_macro']:.3f} |")
    lines+=['','## RL readiness','']
    for hour in [8,16]:
        r=read(run/'evaluation'/f'hour-{hour:02d}-readiness/result.json')
        lines.append(f"Hour {hour}: {r['all_zero_groups']}/{r['groups']} all-failure groups; {r['flat_groups']}/{r['groups']} tied groups. Four sampled trajectories per fixed dev episode; diagnostic only.")
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    write(run/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),training=read(run/'training-complete.json'),
        schedule=read(run/'step-schedule.json'),evaluation_hours=list(range(2,17,2)),report=str(run/'report.md')))


def evaluate(run):
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('evaluation node only')
    lock=(run/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(run,'eval','waiting_for_initialization')
    while not (run/'first-update-gate.json').exists():
        if (run/'user-stopped.json').exists():return
        time.sleep(10)
    if not (run/'cross-node-parity.json').exists():
        try:checked([PY,__file__,'parity','--run',run],run/'logs/parity.log')
        except Exception:
            write(run/'cross-node-parity-failed.json',dict(failed=True));raise
    # Only three complete cases for the saved random-head initialization.
    if not (run/'initialization-smoke/result.json').exists():
        checked([PY,__file__,'smoke','--run',run],run/'logs/initialization-smoke.log')
    failures={}
    while True:
        if (run/'user-stopped.json').exists():return
        # Reconcile a crash between checkpoint commit and queue publication.
        for path in (run/'train').glob('step-*/commit.json'):
            commit=read(path);hour=commit.get('evaluation_hour')
            if hour is not None and not (run/'queue'/f'hour-{hour:02d}.json').exists():
                write(run/'queue'/f'hour-{hour:02d}.json',dict(name=f'hour-{hour:02d}',hour=hour,step=commit['step'],checkpoint=str(path.parent),training_seconds=commit['training_seconds']))
        jobs=sorted([read(p) for p in (run/'queue').glob('hour-*.json')],key=lambda j:j['hour'])
        work=[]
        for job in jobs:
            for probe in ([False,True] if job['hour'] in [8,16] else [False]):
                name=job['name']+('-readiness' if probe else '')
                if not (run/'evaluation'/name/'result.json').exists():work.append((job,probe,name))
        if not work:
            if (run/'training-complete.json').exists() and len(jobs)==8:
                report(run);status(run,'eval','complete');return
            status(run,'eval','waiting',completed=len(jobs));time.sleep(15);continue
        job,probe,name=work[0];status(run,'eval','evaluating',job=name,pending=len(work))
        try:checked([PY,__file__,'score','--run',run,'--job',job['name']]+(['--probe'] if probe else []),run/'logs/evaluation.log')
        except Exception:
            failures[name]=failures.get(name,0)+1;status(run,'eval','retrying',job=name,attempt=failures[name])
            if failures[name]>=10:raise
            time.sleep(30)


def smoke(run):
    directory=run/'initialization-smoke';directory.mkdir(exist_ok=True)
    cohorts=read(run/'data/cohorts.json');specs=rows(run/'data/evaluation-specs.jsonl')
    selected=[]
    for cohort in ['web_all','ego_all','estp_it_explicit_dev']:
        members=set(cohorts[cohort]);selected.append(min((s for s in specs if s['span_id'] in members),key=lambda s:s['rollout_frames'])['span_id'])
    write(directory/'selection.json',selected)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    with (directory/'inference.log').open('a') as handle:
        subprocess.run(list(map(str,[PY,ROOT/'scripts/eval_jevany_streaming.py','--controller',run/'train/step-000000',
            '--validation-manifest',run/'data/evaluation-manifest.jsonl','--selection-ids',directory/'selection.json',
            '--output-dir',directory/'rollout','--response-margin','0','--answer-model','promoted','--max-new-tokens','96','--record-action-logits'])),
            env=env,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',directory/'rollout','--world-size','1'],directory/'merge.log')
    predictions=rows(directory/'rollout/predictions.jsonl')
    lookup={s['span_id']:s for s in specs}
    for p in predictions:
        assert len(p['jevany_action_trace'])==lookup[p['question_id']]['rollout_frames']
        assert all(t['action']==('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in p['jevany_action_trace'])
    write(directory/'result.json',dict(passed=True,episodes=len(predictions),purpose='initialization correctness only; not quality baseline'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','eval','parity','smoke','score'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--job');p.add_argument('--probe',action='store_true')
    a=p.parse_args();run=a.run.resolve()
    if a.mode=='prepare':prepare(run)
    elif a.mode=='train':train(run)
    elif a.mode=='eval':evaluate(run)
    elif a.mode=='parity':parity(run)
    elif a.mode=='smoke':smoke(run)
    elif a.mode=='score':infer_and_score(run,read(run/'queue'/f'{a.job}.json'),a.probe)
