#!/usr/bin/env python3
"""Real-model replay parity/gradient capacity probe; takes no optimizer step."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from jevany_rl_workers import prepare_chunks, fp32_depthwise_convolutions
from benchmark_qwen35_2b_training import batched_logits, chunk_conv1d_batches
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl import observation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--conv-fp32',action='store_true')
    p.add_argument('--precision-probe',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
    specs=[json.loads(line) for line in (root/'outputs/jevany-2b-rl/data/spans.jsonl').read_text().splitlines()][:32]
    rows=[observation(s,s['prefix_turns'],s['rollout_start']) for s in specs]
    if a.diagnostic:
        rows = rows[:8]
    runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
    if a.conv_fp32:
        fp32_depthwise_convolutions(runtime.model)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
    for name,parameter in runtime.model.named_parameters():
        parameter.requires_grad_('lora_A' in name or 'lora_B' in name)
    for module in runtime.model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    chunk_conv1d_batches(runtime.model,128)
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
    results=[]
    baseline=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        runtime.model.eval()
        for start,samples,packed in prepare_chunks(rows,frames,runtime,1,pool):
            with torch.no_grad():
                logits,_=batched_logits(runtime,head,samples,packed)
                baseline.append(F.log_softmax(logits.float(),-1).cpu())
            for row in rows[start:start+len(samples)]:
                row['padded_length']=packed[0]['input_ids'].shape[1]
        baseline=torch.cat(baseline)
        order=sorted(range(len(rows)),key=lambda i:rows[i]['padded_length'])
        rows=[rows[i] for i in order]
        baseline=baseline[order]
        if a.precision_probe:
            for variant in ('full_precision_reduction','fp32_linear'):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
                if variant == 'fp32_linear':
                    for module in runtime.model.modules():
                        if isinstance(module,torch.nn.Linear) and module.weight.dtype == torch.bfloat16:
                            def forward(inputs,module=module):
                                bias=module.bias.float() if module.bias is not None else None
                                return F.linear(inputs.float(),module.weight.float(),bias).to(inputs.dtype)
                            module.forward=forward
                values_by_size={}
                for size in (1,4):
                    values=[]
                    for _,samples,packed in prepare_chunks(rows,frames,runtime,size,pool):
                        with torch.no_grad():
                            logits,_=batched_logits(runtime,head,samples,packed)
                            values.append(F.log_softmax(logits.float(),-1).cpu())
                    values_by_size[size]=torch.cat(values)
                reference,replay=values_by_size[1],values_by_size[4]
                row={'variant':variant,'max_logp_error':float((reference-replay).abs().max()),
                     'max_probability_error':float((reference.exp()-replay.exp()).abs().max()),
                     'max_kl':float((reference.exp()*(reference-replay)).sum(-1).max())}
                print(json.dumps(row),flush=True)
                results.append(row)
            a.output.write_text(json.dumps(results,indent=2)+'\n')
            return
        if a.diagnostic:
            for mode, grad, size in (('eval',False,4),('train',False,1),('train',True,1),('train',True,4)):
                runtime.model.train(mode == 'train')
                runtime._conditional_model().model.visual.eval()
                values = []
                for _,samples,packed in prepare_chunks(rows,frames,runtime,size,pool):
                    with torch.set_grad_enabled(grad):
                        logits,_=batched_logits(runtime,head,samples,packed)
                        values.append(F.log_softmax(logits.float(),-1).detach().cpu())
                logp = torch.cat(values)
                row = {'mode':mode,'grad_enabled':grad,'microbatch':size,
                       'max_logp_error':float((logp-baseline).abs().max()),
                       'max_probability_error':float((logp.exp()-baseline.exp()).abs().max()),
                       'max_kl':float((baseline.exp()*(baseline-logp)).sum(-1).max()),
                       'baseline':baseline.tolist(),'replay':logp.tolist()}
                print(json.dumps(row),flush=True)
                results.append(row)
            a.output.write_text(json.dumps(results,indent=2)+'\n')
            return
        for size in (4,8,16,32):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started=time.monotonic()
            maximum=0.
            runtime.model.train()
            runtime._conditional_model().model.visual.eval()
            for start,samples,packed in prepare_chunks(rows,frames,runtime,size,pool):
                logits,_=batched_logits(runtime,head,samples,packed)
                logp=F.log_softmax(logits.float(),-1)
                maximum=max(maximum,float((logp.detach().cpu()-baseline[start:start+len(samples)]).abs().max()))
                with runtime._cached_image_features(packed[1]):
                    (-logp[:,2].sum()/len(rows)).backward()
            torch.cuda.synchronize()
            lora_norm=sum(float(p.grad.float().square().sum()) for p in runtime.model.parameters() if p.grad is not None)**.5
            head_norm=sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
            row={'microbatch':size,'seconds':time.monotonic()-started,'max_logp_error':maximum,
                 'lora_grad_norm':lora_norm,'head_grad_norm':head_norm,
                 'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                 'reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            print(json.dumps(row),flush=True)
            results.append(row)
            runtime.model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
    a.output.write_text(json.dumps({'optimizer_steps':0,'results':results},indent=2)+'\n')


if __name__=='__main__': main()
