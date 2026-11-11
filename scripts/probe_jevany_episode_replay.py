#!/usr/bin/env python3
"""Diagnose saved episode replay on SFT weights, without optimizer updates."""
import argparse
import copy
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_qwen35_2b_training import batched_logits, chunk_conv1d_batches
from jevany_rl_episode_workers import prepare_chunks
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl import parity_metrics
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl_reference import load_exact_reference
from train_jevany_streaming import EpisodeFrames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--rank',type=int,default=5)
    args=parser.parse_args()
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('authorized node only')
    torch.set_num_threads(1)
    cfg=json.loads((args.run/'freeze.json').read_text())
    traces=json.loads((args.run/'batch-000000/scored.json').read_text())
    steps=[s for t in traces if t['trajectory_id']%7==args.rank for s in t['steps']]
    steps.sort(key=lambda s:(s['row']['padded_length'],s['row']['physical_batch'],len(s['row']['turns'])))
    initial=Path(cfg['checkpoint'])
    runtime,_=load_2b_controller(checkpoint=str(initial/'controller_adapter'))
    install_cached_vision_inputs(runtime);load_exact_reference(runtime.model,initial/'controller_adapter')
    runtime.model.set_adapter('default')
    for name,p in runtime.model.named_parameters():p.requires_grad_('.default.' in name and ('lora_A' in name or 'lora_B' in name))
    for module in runtime.model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    runtime.model.gradient_checkpointing_enable();runtime.model.enable_input_require_grads()
    chunk_conv1d_batches(runtime.model,128)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(initial/'pointer_head.pt',map_location='cpu',weights_only=True))
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
    frames.physical_batch=24;frames.token_bucket=512;frames.token_budget=262144
    rows=[s['row'] for s in steps];results=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for start,samples,packed in prepare_chunks(rows,frames,runtime,24,pool):
            chunk=steps[start:start+len(samples)]
            old=torch.tensor([s['old_logp'] for s in chunk],device=runtime.device)
            actions=torch.tensor([s['action'] for s in chunk],device=runtime.device)
            values={}
            for mode,grad in (('eval',False),('train',False),('train',True)):
                runtime.model.train(mode=='train');runtime._conditional_model().model.visual.eval()
                with torch.set_grad_enabled(grad):
                    logp=F.log_softmax(batched_logits(runtime,head,samples,packed)[0].float(),-1)
                    values[mode+str(grad)]=logp.detach()
                del logp
            entry={'start':start,'frames':[s['row']['frame_index'] for s in chunk],
                   'episodes':[s['row']['episode_id'] for s in chunk],
                   'shape':list(packed[0]['input_ids'].shape),
                   'metrics':{name:parity_metrics(v,old,actions) for name,v in values.items()}}
            results.append(entry)
            print(json.dumps(entry),flush=True)
            if max(v['probability_max'] for v in entry['metrics'].values())>.005:
                entry['old']=old.cpu().tolist();entry['current']={k:v.cpu().tolist() for k,v in values.items()}
                args.output.write_text(json.dumps({'optimizer_steps':0,'results':results},indent=2)+'\n')
                break
        else:args.output.write_text(json.dumps({'optimizer_steps':0,'results':results},indent=2)+'\n')


if __name__=='__main__':main()
