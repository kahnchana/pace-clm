#!/usr/bin/env python3
"""Replay initial collected batches on the evaluation node, without updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket

import torch
import torch.nn.functional as F
from benchmark_qwen35_2b_training import batched_logits,chunk_conv1d_batches
from jevany_rl_mixed_workers import prepare_chunks
from jevany_mixed_inputs import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl import parity_metrics
from jevany_rl_reward import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if socket.gethostname()!='ip-10-3-70-177':raise RuntimeError('evaluation node only')
    cfg=json.loads((a.run/'freeze.json').read_text());initial=Path(cfg['checkpoint'])
    torch.set_num_threads(1);runtime,_=load_2b_controller(checkpoint=str(initial/'controller_adapter'))
    install_cached_vision_inputs(runtime);runtime.model.eval();chunk_conv1d_batches(runtime.model,128)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(initial/'pointer_head.pt',map_location='cpu',weights_only=True));head.eval()
    traces=json.loads((a.run/'batch-000000/scored.json').read_text());result=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for rank in range(7):
            selected=[s for t in traces if t['trajectory_id']%7==rank for s in t['steps'] if s['row']['collection_batch']==0]
            selected.sort(key=lambda s:s['row']['collection_position'])
            frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
            frames.physical_batch=24;frames.token_bucket=512;frames.token_budget=262144
            for _,samples,packed in prepare_chunks([s['row'] for s in selected],frames,runtime,24,pool):
                with torch.no_grad():logp=F.log_softmax(batched_logits(runtime,head,samples,packed)[0].float(),-1)
                old=torch.tensor([s['old_logp'] for s in selected],device=runtime.device)
                actions=torch.tensor([s['action'] for s in selected],device=runtime.device)
                metrics=parity_metrics(logp,old,actions);result.append({'rank':rank,**metrics})
                if metrics['probability_max']>.005 or metrics['kl_max']>1e-4 or metrics['kl_sum']/len(selected)>1e-5 or metrics['selected_logp_max']>.05:
                    raise RuntimeError(f'cross-node replay parity failed: {metrics}')
    write(a.run/'cross-node-parity.json',{'verified':True,'optimizer_steps':0,'ranks':result})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
