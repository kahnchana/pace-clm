#!/usr/bin/env python3
"""Compare collected layout versus regrouped replay without policy updates."""
import argparse
import copy
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_qwen35_2b_training import batched_logits, chunk_conv1d_batches
from jevany_rl_episode_workers import prepare_chunks
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl import parity_metrics
from pace_clm.rl_episode import ordered_replay
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl_reference import load_exact_reference
from train_jevany_streaming import EpisodeFrames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('authorized node only')
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=json.loads((args.run/'freeze.json').read_text());initial=Path(cfg['checkpoint'])
    traces=json.loads((args.run/'batch-000000/scored.json').read_text())
    selected=[copy.deepcopy(s) for t in traces if t['trajectory_id']%7==5 for s in t['steps']
              if s['row']['frame_index']<=8 or s['row']['frame_index'] in (31,32,64,178)]
    selected.sort(key=lambda s:(s['row']['frame_index'],s['row']['episode_id']))
    runtime,_=load_2b_controller(checkpoint=str(initial/'controller_adapter'))
    install_cached_vision_inputs(runtime);load_exact_reference(runtime.model,initial/'controller_adapter')
    runtime.model.set_adapter('default')
    for name,p in runtime.model.named_parameters():p.requires_grad_('.default.' in name and ('lora_A' in name or 'lora_B' in name))
    for module in runtime.model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    runtime.model.gradient_checkpointing_enable();runtime.model.enable_input_require_grads();chunk_conv1d_batches(runtime.model,128)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(initial/'pointer_head.pt',map_location='cpu',weights_only=True))
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
    frames.physical_batch=24;frames.token_bucket=512;frames.token_budget=262144
    old={};results=[];batch_id=0
    with ThreadPoolExecutor(max_workers=1) as pool:
        runtime.model.eval()
        for _,group in groupby(selected,key=lambda s:s['row']['frame_index']):
            group=list(group)
            for s in group:
                s['row'].pop('padded_length');s['row'].pop('physical_batch')
                s['row']['probe_index']=len(old)
                old[len(old)]=None
            for start,samples,packed in prepare_chunks([s['row'] for s in group],frames,runtime,24,pool):
                with torch.no_grad():values=F.log_softmax(batched_logits(runtime,head,samples,packed)[0].float(),-1).cpu()
                for pos,(s,value) in enumerate(zip(group[start:start+len(samples)],values,strict=True)):
                    row=s['row'];old[row['probe_index']]=value
                    row.update(padded_length=packed[0]['input_ids'].shape[1],physical_batch=packed[0]['input_ids'].shape[0],
                               collection_batch=batch_id,collection_position=pos,collection_size=len(samples))
                batch_id+=1
        for variant in ('collected-layout','regrouped-layout'):
            steps=copy.deepcopy(selected)
            if variant=='collected-layout':steps=ordered_replay(steps)
            else:
                for s in steps:
                    for key in ('collection_batch','collection_position','collection_size'):s['row'].pop(key)
                steps.sort(key=lambda s:(s['row']['padded_length'],s['row']['physical_batch'],len(s['row']['turns'])))
            metrics=[]
            runtime.model.train();runtime._conditional_model().model.visual.eval()
            for start,samples,packed in prepare_chunks([s['row'] for s in steps],frames,runtime,24,pool):
                group=steps[start:start+len(samples)]
                previous=torch.stack([old[s['row']['probe_index']] for s in group]).to(runtime.device)
                actions=torch.tensor([s['action'] for s in group],device=runtime.device)
                logits,_=batched_logits(runtime,head,samples,packed)
                logp=F.log_softmax(logits.float(),-1)
                metrics.append(parity_metrics(logp,previous,actions))
                if variant=='collected-layout':
                    with runtime._cached_image_features(packed[1]):(-logp[:,2].sum()/len(steps)).backward()
                del logits,logp
            result={'variant':variant,'observations':len(steps),
                    'probability_max':max(m['probability_max'] for m in metrics),
                    'kl_max':max(m['kl_max'] for m in metrics),
                    'kl_mean':sum(m['kl_sum'] for m in metrics)/len(steps),
                    'selected_logp_max':max(m['selected_logp_max'] for m in metrics)}
            print(json.dumps(result),flush=True);results.append(result)
    args.output.write_text(json.dumps({'optimizer_steps':0,'results':results},indent=2)+'\n')
    if results[0]['probability_max']>.005 or results[0]['kl_max']>1e-4 or results[0]['kl_mean']>1e-5 or results[0]['selected_logp_max']>.05:
        raise RuntimeError('collected-layout replay failed existing tolerances')


if __name__=='__main__':main()
