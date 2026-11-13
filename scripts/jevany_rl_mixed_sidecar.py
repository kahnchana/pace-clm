#!/usr/bin/env python3
"""Frozen generator/judge process; queue payloads contain no Torch tensors."""
import argparse
import json
import os
import importlib.metadata
from pathlib import Path

from jevany_rl_rpc import connect


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    manager=connect(a.port,bytes.fromhex(os.environ.pop('PACE_RL_RPC_KEY')))
    cfg=json.loads(a.config.read_text())
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','triton')}
    if versions!=cfg['generator_versions']:raise RuntimeError('generator runtime differs from frozen versions')
    from jevany_rl_rpc import require_native_grouped_mm
    counts=require_native_grouped_mm()
    from jevany_rl_mixed_generator_workers import generator
    try:
        generator(a.gpu,cfg,manager.events(),manager.requests(),
                  [manager.answer(rank) for rank in range(cfg['learners'])])
    finally:
        (Path(cfg['output'])/f'generator_runtime.gpu{a.gpu}.json').write_text(
            json.dumps({'versions':versions,**counts},indent=2)+'\n')


if __name__=='__main__':main()
