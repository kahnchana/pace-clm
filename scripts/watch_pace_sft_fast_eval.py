#!/usr/bin/env python3
"""Supervise one pinned faster evaluation, without a wall-time cutoff."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();out=args.output_dir.resolve()
    lock=(out/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dispatch=read(out/'dispatch.json');restarts=0
    while not (out/'user-stopped.json').exists():
        if (out/'complete.json').exists():
            write(out/'watcher-status.json',dict(state='complete',utc=now(),step=dispatch['step'],report=str(out/'report.md'),restarts=restarts));return
        launch=read(out/'launch.json');pid=launch['pid']
        if not alive(pid):
            if restarts>=3:
                write(out/'needs-attention.json',dict(utc=now(),reason='technical retries exhausted',step=dispatch['step']));return
            for path,digest in dispatch['source_hashes'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('frozen implementation changed: '+path)
            with (out/'launcher.log').open('ab',buffering=0) as handle:
                p=subprocess.Popen(['srun','--jobid=38177','--overlap','--cpu-bind=none','--nodes=1','--ntasks=1',
                    'bash','scripts/run_pace_sft_budget_eval.sh','--checkpoint',dispatch['checkpoint'],'--output-dir',str(out)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            pid=p.pid;write(out/'launch.json',dict(pid=pid,allocation=38177,utc=now()));restarts+=1
        phase=read(out/'status.json').get('phase') if (out/'status.json').exists() else 'starting'
        write(out/'watcher-status.json',dict(state='running',phase=phase,utc=now(),step=dispatch['step'],launcher_pid=pid,launcher_alive=alive(pid),restarts=restarts,hard_time_limit=False))
        time.sleep(30)


if __name__=='__main__':main()
