#!/usr/bin/env python3
"""Continue the committed mixed SFT sampler to a second prepared ESTP-IT pass."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from prepare_pace_estp_sft import ROOT,read,write
from run_pace_estp_sft_v2 import train,verify


def now():return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def followup(run):
    write(run/'next-action.json',dict(state='training_complete',utc=now(),
        checkpoint=read(run/'training-complete.json')['checkpoint'],
        instructions='Preserve completed training. No automatic evaluations, further training or RL. Step-2275 evaluation is a separate authorized job.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','monitor']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();verify(run)
    if args.mode=='worker':
        train(run)
        if (run/'training-complete.json').exists():followup(run)
        return
    lock=(run/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    retries=0
    while not (run/'user-stopped.json').exists():
        if (run/'training-complete.json').exists():
            followup(run);write(run/'monitor-status.json',dict(state='training_complete',utc=now(),restarts=retries));return
        verify(run);path=run/'launch.json';launch=read(path) if path.exists() else {};pid=launch.get('pid',0)
        if not pid or not alive(pid):
            if retries>=3:
                write(run/'monitor-needs-attention.json',dict(utc=now(),restarts=retries));return
            with (run/'logs/launcher.log').open('ab',buffering=0) as handle:
                child=subprocess.Popen(['srun','--jobid=38226','--overlap','--cpu-bind=none','--nodes=1','--ntasks=1',
                    'bash','scripts/run_pace_estp_sft_epoch2.sh'],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;write(path,dict(pid=pid,allocation=38226,utc=now()));retries+=1
        write(run/'monitor-status.json',dict(state='training',utc=now(),launcher_pid=pid,launcher_alive=alive(pid),launch_attempts=retries))
        time.sleep(30)


if __name__=='__main__':main()
