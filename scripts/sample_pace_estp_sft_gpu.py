#!/usr/bin/env python3
"""Read-only GPU telemetry for the authorized training allocation."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def main(run):
    with (run/'logs/gpu-samples.jsonl').open('a',buffering=1) as handle:
        while not (run/'training-complete.json').exists() and not (run/'user-stopped.json').exists():
            path=run/'train-supervisor-status.json'
            status=json.loads(path.read_text()) if path.exists() else {}
            result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            handle.write(json.dumps(dict(epoch=time.time(),stage=status.get('state'),physical_batch=status.get('physical_batch'),
                gpus=[[float(v.strip()) for v in row] for row in csv.reader(result.stdout.splitlines())]))+'\n')
            time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
