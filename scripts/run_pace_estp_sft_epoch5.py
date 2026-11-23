#!/usr/bin/env python3
"""Continue the committed mixed SFT sampler to five prepared ESTP-IT passes."""
import argparse
import socket
import hashlib
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
        instructions='Preserve completed training. No automatic evaluations, further training or RL. Five-epoch training budget is complete.'))


def await_parent(run):
    """Dispatch only after the committed epoch-two endpoint releases its GPUs."""
    parent=Path(read(run/'freeze.json')['parent_run'])
    while not (run/'user-stopped.json').exists():
        if (run/'train/latest.json').exists():
            return True
        marker=parent/'training-complete.json'
        reason='waiting_for_epoch2_completion'
        if marker.exists():
            final=read(marker)
            if final['step']!=4550:
                raise ValueError('Expected epoch-two final step 4550')
            launch=read(parent/'launch.json')
            if alive(launch['pid']):
                reason='waiting_for_epoch2_worker_exit'
            else:
                if socket.gethostname()!='ip-10-3-70-177':
                    raise RuntimeError('Continuation dispatch must run on training node')
                check=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                if check.stdout.strip():
                    reason='waiting_for_training_gpu_release'
                else:
                    verify(run)
                    checkpoint=Path(final['checkpoint']).resolve()
                    expected=(parent/'train/step-004550').resolve()
                    if checkpoint!=expected:
                        raise ValueError('Unexpected parent checkpoint path')
                    commit=read(checkpoint/'commit.json')
                    if commit['step']!=4550:
                        raise ValueError('Parent commit step differs')
                    for name,digest in commit['files'].items():
                        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=digest:
                            raise ValueError('Parent checkpoint checksum mismatch: '+name)
                    link=run/'train/step-004550'
                    if not link.exists():
                        link.symlink_to(checkpoint,target_is_directory=True)
                    write(run/'resume-audit.json',dict(state='parent_commit_verified_awaiting_first_update',parent_step=4550,checkpoint=str(checkpoint),all_commit_files_verified=True,utc=now()))
                    write(run/'train/latest.json',dict(step=4550,checkpoint=str(checkpoint)))
                    write(run/'dispatch-ready.json',dict(utc=now(),checkpoint=str(checkpoint),parent_gpus_released=True))
                    return True
        write(run/'monitor-status.json',dict(state=reason,utc=now(),resume_step=4550,target_step=11375))
        time.sleep(30)
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','monitor']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();verify(run)
    if args.mode=='worker':
        train(run)
        if (run/'training-complete.json').exists():followup(run)
        return
    lock=(run/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not await_parent(run):return
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
                    'bash','scripts/run_pace_estp_sft_epoch5.sh'],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;write(path,dict(pid=pid,allocation=38226,utc=now()));retries+=1
        write(run/'monitor-status.json',dict(state='training',utc=now(),launcher_pid=pid,launcher_alive=alive(pid),launch_attempts=retries))
        time.sleep(30)


if __name__=='__main__':main()
