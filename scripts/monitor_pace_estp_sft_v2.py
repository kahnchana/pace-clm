#!/usr/bin/env python3
"""Recover only this fixed-step SFT workflow; honor explicit stop markers."""
import argparse
import fcntl
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from prepare_pace_estp_sft import read,write
from run_pace_estp_sft_v2 import verify


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main(run):
    lock=(run/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restarts={'train':0,'eval':0}
    while not (run/'user-stopped.json').exists():
        verify(run);launches=read(run/'launch.json');states={}
        for mode,allocation in [('train',38226),('eval',38177)]:
            done=(run/('training-complete.json' if mode=='train' else 'complete.json')).exists()
            running=alive(launches[mode]['pid'])
            if not done and not running and restarts[mode]<3:
                with (run/'logs'/f'{mode}-launcher.log').open('ab',buffering=0) as handle:
                    child=subprocess.Popen(['srun',f'--jobid={allocation}','--overlap','--cpu-bind=none','--nodes=1','--ntasks=1',
                        'bash','scripts/run_pace_estp_sft_v2.sh',mode],stdin=subprocess.DEVNULL,stdout=handle,
                        stderr=subprocess.STDOUT,start_new_session=True)
                launches[mode]['pid']=child.pid;write(run/'launch.json',launches);restarts[mode]+=1;running=True
            states[mode]=dict(done=done,launcher_alive=running,restarts=restarts[mode],pid=launches[mode]['pid'])
        write(run/'monitor-status.json',dict(utc=datetime.now(timezone.utc).isoformat(),processes=states,
            completed_evaluations=[h for h in range(2,17,2) if (run/'evaluation'/f'hour-{h:02d}/result.json').exists()]))
        if (run/'complete.json').exists():return
        if any(not v['done'] and not v['launcher_alive'] for v in states.values()):
            write(run/'monitor-needs-attention.json',dict(utc=datetime.now(timezone.utc).isoformat(),processes=states));return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
