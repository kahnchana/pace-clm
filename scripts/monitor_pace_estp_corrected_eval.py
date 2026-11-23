#!/usr/bin/env python3
"""Bounded restart supervision for the corrected, explicitly selected suites."""
import fcntl
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h/evaluation/corrected-step143'


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restarts=0
    while not (OUT/'user-stopped.json').exists():
        launch=json.loads((OUT/'launch.json').read_text());pid=launch['pid'];done=(OUT/'complete.json').exists()
        if not done and not alive(pid) and restarts<3:
            with (OUT/'launcher.log').open('ab',buffering=0) as handle:
                child=subprocess.Popen(['srun','--jobid=38177','--overlap','--cpu-bind=none','--nodes=1','--ntasks=1','bash','scripts/run_pace_estp_sft_corrected_eval.sh'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid;launch['pid']=pid;write(OUT/'launch.json',launch);restarts+=1
        state=dict(utc=datetime.now(timezone.utc).isoformat(),complete=done,launcher_pid=pid,launcher_alive=alive(pid),restarts=restarts,
            completed_suites=[s for s in ['ego','web','estp-bench-explicit'] if (OUT/s/'result.json').exists()])
        write(OUT/'monitor-status.json',state)
        if done:return
        if not alive(pid) and restarts>=3:write(OUT/'monitor-needs-attention.json',state);return
        time.sleep(30)


if __name__=='__main__':main()
