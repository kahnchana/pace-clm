#!/usr/bin/env python3
"""Capacity-only stress: 32 frames and 31 maximum-length prior answers; no update."""
import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_qwen35_2b_training import batched_logits, chunk_conv1d_batches
from jevany_rl_workers import prepare_chunks
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl import observation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
    runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
    runtime.model.load_adapter(str(checkpoint/'controller_adapter'),adapter_name='reference',is_trainable=False)
    runtime.model.set_adapter('default')
    for name,parameter in runtime.model.named_parameters():
        parameter.requires_grad_('.default.' in name and ('lora_A' in name or 'lora_B' in name))
    for module in runtime.model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    chunk_conv1d_batches(runtime.model,128)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
    parameters=list(head.parameters())+[p for p in runtime.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters)
    for parameter in parameters:
        optimizer.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),
                                    'exp_avg_sq':torch.zeros_like(parameter)}
    specs=[json.loads(line) for line in (root/'outputs/jevany-2b-rl/data/spans.jsonl').read_text().splitlines()]
    spec=next(s for s in specs if len(s['prefix_turns'])==31)
    row=observation(spec,spec['prefix_turns'],spec['rollout_start'])
    tokens=runtime.tokenizer.encode('A person moves an object across the table. '*40,add_special_tokens=False)[:96]
    answer=runtime.tokenizer.decode(tokens)
    for turn in row['turns'][:-1]:
        turn['emit']='</Response> '+answer
        turn['state']='Response'
    rows=[copy.deepcopy(row) for _ in range(48)]
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=512,defer_vision=True)
    frames.token_budget=262144
    sample=frames.prepare(rows[0],cpu_only=True)
    padded_length=(sample[0]['input_ids'].shape[1]//512+1)*512
    for row in rows:
        row['padded_length']=padded_length
    results=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for size in (16,32,48):
            torch.cuda.reset_peak_memory_stats()
            runtime.model.train()
            runtime._conditional_model().model.visual.eval()
            frames.rl_stats={'cpu_prepare_seconds':0.,'prepare_wait_seconds':0.,'vision_seconds':0.}
            for _,samples,packed in prepare_chunks(rows,frames,runtime,size,pool):
                logits,_=batched_logits(runtime,head,samples,packed)
                loss=-F.log_softmax(logits.float(),-1)[:,2].sum()/len(rows)
                with runtime._cached_image_features(packed[1]): loss.backward()
            torch.cuda.synchronize()
            result={'microbatch_cap':size,'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                    'reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                    'shapes':frames.rl_stats['microbatch_shapes'],
                    'head_grad_norm':float(head.q.weight.grad.norm())}
            print(json.dumps(result),flush=True)
            results.append(result)
            optimizer.zero_grad(set_to_none=True)
    a.output.write_text(json.dumps({'optimizer_steps':0,'prior_answer_tokens':len(tokens),
                                   'context_frames':32,'optimizer_state_allocated':True,
                                   'frozen_reference_adapter_resident':True,'results':results},indent=2)+'\n')


if __name__=='__main__': main()
