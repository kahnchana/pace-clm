#!/usr/bin/env python3
"""Monitor this authorized run and reconcile interrupted milestone publication."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

from prepare_pace_estp_sft import read,write


def reconcile(run):
    commits=[]
    for path in (run/'train').glob('step-*/commit.json'):
        item=read(path);item['checkpoint']=str(path.parent);commits.append(item)
    recovered=[]
    for hour in range(2,17,2):
        target=run/'queue'/f'hour-{hour:02d}.json'
        eligible=[c for c in commits if c['training_seconds']>=hour*3600]
        if not target.exists() and eligible:
            item=min(eligible,key=lambda c:c['step'])
            write(target,dict(name=f'hour-{hour:02d}',hour=hour,step=item['step'],
                checkpoint=item['checkpoint'],training_seconds=item['training_seconds']))
            recovered.append(hour)
    return recovered


def main(run):
    import fcntl
    lock=(run/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restarts={'train':0,'eval':0}
    while True:
        recovered=reconcile(run)
        if recovered:
            write(run/'milestone-publication-recovery.json',dict(hours=recovered,utc=datetime.now(timezone.utc).isoformat()))
        status={}
        launches=read(run/'launch.json')
        for mode,allocation in [('train',38226),('eval',38177)]:
            done=(run/('training-complete.json' if mode=='train' else 'complete.json')).exists()
            pid=launches[mode]['pid']
            try:
                stat=Path(f'/proc/{pid}/stat').read_text().split()
                alive=stat[2]!='Z'
            except FileNotFoundError:alive=False
            if not done and not alive and restarts[mode]<3:
                log=(run/'logs'/f'{mode}-launcher.log').open('ab',buffering=0)
                p=subprocess.Popen(['srun',f'--jobid={allocation}','--overlap','--cpu-bind=none',
                    '--nodes=1','--ntasks=1','bash','scripts/run_pace_estp_sft.sh',mode],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                log.close();launches[mode]['pid']=p.pid;write(run/'launch.json',launches)
                restarts[mode]+=1;alive=True
            status[mode]=dict(done=done,launcher_alive=alive,restarts=restarts[mode],pid=launches[mode]['pid'])
        write(run/'monitor-status.json',dict(utc=datetime.now(timezone.utc).isoformat(),processes=status,
            completed_evaluations=[h for h in range(2,17,2) if (run/'evaluation'/f'hour-{h:02d}/result.json').exists()]))
        if (run/'complete.json').exists():return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
