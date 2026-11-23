#!/usr/bin/env python3
"""Dispatch exactly one latest-checkpoint evaluation after the current suite."""
import argparse
import fcntl
import hashlib
import json
import sqlite3
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h'
TRAIN=PARENT.with_name('base-50-50-b384-to-estp-epoch1')
CURRENT=PARENT/'evaluation/corrected-step143'
QUEUE=PARENT/'evaluation/latest-after-step143'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def latest():
    candidates=[]
    for path in (TRAIN/'train').glob('step-*/commit.json'):
        commit=read(path)
        if int(path.parent.name.split('-')[-1])!=commit['step']:raise ValueError('checkpoint directory/commit mismatch')
        if not {'pointer_head.pt','controller_adapter/adapter_model.safetensors','training_state.pt'}<=commit['files'].keys():raise ValueError('incomplete commit contract')
        candidates.append((commit['step'],path))
    if not candidates:raise RuntimeError('no committed continuation checkpoint available')
    step,path=max(candidates,key=lambda pair:pair[0]);commit=read(path)
    for name in ['pointer_head.pt','controller_adapter/adapter_model.safetensors']:
        if sha(path.parent/name)!=commit['files'][name]:raise ValueError('checkpoint checksum mismatch')
    return dict(step=step,checkpoint=str(path.parent.resolve()),commit_sha256=sha(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    if args.inspect:
        print(json.dumps(latest()));return
    lock=(QUEUE/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(QUEUE/'plan.json');restarts=0
    while not (QUEUE/'user-stopped.json').exists():
        for path,digest in plan['source_hashes'].items():
            if sha(Path(path))!=digest:raise ValueError('queued evaluation source changed: '+path)
        if not (CURRENT/'complete.json').exists():
            write(QUEUE/'status.json',dict(state='waiting_for_current_evaluation',utc=now(),depends_on=str(CURRENT/'complete.json')));time.sleep(30);continue
        old=read(CURRENT/'launch.json')
        if alive(old['pid']):
            write(QUEUE/'status.json',dict(state='waiting_for_current_gpu_release',utc=now()));time.sleep(5);continue
        selection=QUEUE/'selection.json'
        if selection.exists():chosen=read(selection)
        else:
            chosen=dict(**latest(),selected_utc=now(),rule='Newest fully committed continuation checkpoint at dispatch, after checkpoint-143 suites complete; pinned for all three suites')
            write(selection,chosen)
        checkpoint=Path(chosen['checkpoint'])
        if sha(checkpoint/'commit.json')!=chosen['commit_sha256']:raise ValueError('pinned checkpoint commit changed')
        output=QUEUE/f"step-{chosen['step']:06d}";output.mkdir(exist_ok=True)
        if (output/'complete.json').exists():
            write(QUEUE/'complete.json',dict(utc=now(),selection=chosen,output=str(output),report=str(output/'report.md')))
            write(QUEUE/'status.json',dict(state='complete',utc=now(),**chosen,output=str(output)));return
        launch_path=QUEUE/'launch.json';launch=read(launch_path) if launch_path.exists() else {};pid=launch.get('pid',0)
        if not pid or not alive(pid):
            if restarts>=3:
                write(QUEUE/'needs-attention.json',dict(utc=now(),selection=chosen,launch_attempts=restarts));return
            # Reuse only exact request/contract-keyed native judgments; current
            # workflow is complete, so its SQLite connection has been closed.
            source=CURRENT/'cache/evaluation.sqlite';cache=output/'cache/evaluation.sqlite'
            if source.exists() and not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                with sqlite3.connect(source) as src,sqlite3.connect(cache) as dst:src.backup(dst)
                write(output/'cache-seed.json',dict(source=str(source),source_sha256=sha(source),protocol='Reuse exact request/unchanged scoring-contract hashes only'))
            with (QUEUE/'launcher.log').open('ab',buffering=0) as handle:
                child=subprocess.Popen(['srun','--jobid=38177','--overlap','--cpu-bind=none','--nodes=1','--ntasks=1',
                    'bash','scripts/run_pace_estp_sft_latest_eval.sh','--checkpoint',chosen['checkpoint'],'--step',str(chosen['step']),'--output-dir',str(output)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;write(launch_path,dict(pid=pid,allocation=38177,selection=chosen,output=str(output),utc=now()));restarts+=1
        write(QUEUE/'status.json',dict(state='evaluating_pinned_latest_checkpoint',utc=now(),selection=chosen,output=str(output),launcher_pid=pid,launch_attempts=restarts))
        time.sleep(30)


if __name__=='__main__':main()
