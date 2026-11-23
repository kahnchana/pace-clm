#!/usr/bin/env python3
"""Resume a failed Flash suite alongside the original same-run supervisor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime,timezone
from run_jevany_flash_parallel import DEFAULT, NODE, ALLOCATION, verify, evaluate, write

def main():
    p=argparse.ArgumentParser();p.add_argument('domain',choices=('web','ego','tv','vad'));a=p.parse_args()
    directory=DEFAULT;target=directory/a.domain
    if socket.gethostname()!=NODE or subprocess.check_output(['squeue','-h','-j',str(ALLOCATION),'-o','%T %N'],text=True).strip()!='RUNNING '+NODE: raise RuntimeError('authorized current node/allocation only')
    lock=(target/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((target/'status.json').read_text())
    if state['state'] not in ('failed','complete'): raise RuntimeError('suite is not failed or complete; refuse duplicate workers')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if 'eval_jevany_flash_streaming.py' in command and str(target/'rollout') in command: raise RuntimeError('suite still has active workers')
    frozen=verify(directory)
    write(target/'recovery.json',dict(state='running',pid=os.getpid(),reason='Transient HTTP-200 empty Flash answer; resume committed episode shards with identical model, prompts, budgets and controller.',utc=datetime.now(timezone.utc).isoformat()))
    for attempt in range(1,4):
        try:
            result=evaluate(directory,a.domain,frozen)
            write(target/'recovery.json',dict(state='complete',attempt=attempt,utc=datetime.now(timezone.utc).isoformat(),cohorts=result.get('cohorts',{})))
            return
        except Exception as exc:
            write(target/'recovery.json',dict(state='retry_pending' if attempt<3 else 'failed',attempt=attempt,error=str(exc),utc=datetime.now(timezone.utc).isoformat()))
            if attempt==3: raise
            time.sleep(10)

if __name__=='__main__': main()
