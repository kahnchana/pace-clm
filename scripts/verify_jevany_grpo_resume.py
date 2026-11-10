#!/usr/bin/env python3
"""Verify GRPO model/optimizer/RNG and immutable within-batch behavior/pass cursors."""
import argparse
import hashlib
import json
import os
import queue
import socket
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SG=ROOT.parent/'stream-gemma'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(SG/'src'),str(SG)]

import torch.multiprocessing as mp
from jevany_grpo_workers import controller
from pace_clm.grpo import recover_progress


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    a=parser.parse_args()
    cfg=json.loads((a.run/'freeze.json').read_text())
    for path,expected in cfg['hashes'].items():
        if digest(path)!=expected:raise ValueError(f'code or artifact changed: {path}')
    latest=recover_progress(a.run)
    checkpoint=Path(latest['checkpoint'])
    state=json.loads((checkpoint/'rl_state.json').read_text())
    if state['policy_version']!=latest['step'] or state['sampler_cursor']!=latest['step']*cfg['starts']:
        raise ValueError('checkpoint policy/sampler mismatch')
    if state['optimizer_steps']!=latest['optimizer_steps'] or state['cursor']!=latest['cursor']:
        raise ValueError('optimizer/pass cursor metadata mismatch')
    cfg['resume']=str(checkpoint)
    cfg['resume_progress']=latest
    if latest['cursor'] is not None:
        batch=a.run/f"batch-{latest['cursor']['round']:06d}"
        for key,name in (('behavior_sha256','behavior.json'),('scored_sha256','scored.json'),('rollouts_sha256','rollouts.json'),('schedule_sha256','schedule.json'),('parity_sha256','parity.json'),('vision_sha256','vision.json')):
            if digest(batch/name)!=latest['cursor'][key]:raise ValueError('saved batch changed across resume')
        for record in json.loads((batch/'vision.json').read_text()).values():
            if digest(batch/record['file'])!=record['sha256']:
                raise ValueError('saved frozen vision features changed across resume')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        cfg['init_method']=f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    ctx=mp.get_context('spawn')
    events,requests=ctx.Queue(),ctx.Queue()
    commands=[ctx.Queue() for _ in range(cfg['learners'])]
    answers=[ctx.Queue() for _ in commands]
    workers=[]
    previous=os.environ.get('CUDA_VISIBLE_DEVICES')
    visible=previous.split(',') if previous else list(map(str,range(8)))
    try:
        for rank in range(cfg['learners']):
            os.environ['CUDA_VISIBLE_DEVICES']=visible[rank]
            worker=ctx.Process(target=controller,args=(rank,cfg,commands[rank],events,requests,answers[rank]))
            worker.start()
            workers.append(worker)
        restored={}
        deadline=time.monotonic()+900
        while len(restored)<cfg['learners']:
            try:event,rank,value=events.get(timeout=5)
            except queue.Empty:
                if any(w.exitcode is not None for w in workers) or time.monotonic()>deadline:
                    raise RuntimeError('resume verification worker exited or stalled')
                continue
            if (event!='ready' or value['restored_step']!=latest['step'] or
                    value.get('exact_reference_tensors')!=300 or
                    value['restored_optimizer_steps']!=latest['optimizer_steps'] or value['restored_cursor']!=latest['cursor']):
                raise RuntimeError(f'failed restore: {event}/{rank}/{value}')
            restored[rank]=value
        result={'verified':True,'checkpoint':str(checkpoint),'step':latest['step'],
                'sampler_cursor':state['sampler_cursor'],'optimizer_steps':latest['optimizer_steps'],
                'cursor':latest['cursor'],'additional_optimizer_steps':0,'ranks':restored}
        (a.run/'resume_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=previous
        for command in commands:command.put(('stop',None))
        for worker in workers:worker.join(timeout=5)
        for worker in workers:
            if worker.is_alive():worker.terminate()
        for worker in workers:worker.join(timeout=5)


if __name__=='__main__':main()
