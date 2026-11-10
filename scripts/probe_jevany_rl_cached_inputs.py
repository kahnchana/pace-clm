#!/usr/bin/env python3
"""Verify cached-vision input optimization preserves real decoder gradients."""
import argparse,json,math
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from benchmark_qwen35_2b_training import batched_logits
from jevany_rl_workers import prepare_chunks
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl_inputs import install_cached_vision_inputs


def main():
 p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1)
 checkpoint=Path('outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243')
 traces=json.loads(a.traces.read_text());steps=[s for t in traces for s in t['steps']]
 steps.sort(key=lambda s:-sum(-math.exp(v)*v for v in s['old_logp']))
 rows=[s['row'] for s in steps[:8]];length=max(r['padded_length'] for r in rows)
 for r in rows:r['padded_length']=length;r.pop('physical_batch',None)
 runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
 head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
 head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
 for name,param in runtime.model.named_parameters():param.requires_grad_('lora_' in name)
 for module in runtime.model.modules():
  if isinstance(module,torch.nn.Dropout):module.p=0.
 runtime.model.train();runtime._conditional_model().model.visual.eval()
 runtime.model.gradient_checkpointing_enable();runtime.model.enable_input_require_grads()
 frames=EpisodeFrames(runtime,response_blind=False,cache_frames=1024,defer_vision=True);frames.physical_batch=24
 parameters={**dict(runtime.model.named_parameters()),**{'head.'+n:p for n,p in head.named_parameters()}}
 references={}
 with ThreadPoolExecutor(max_workers=1) as pool:
  for mode in ('original','cached-vision'):
   if mode=='cached-vision':install_cached_vision_inputs(runtime)
   for param in parameters.values():param.grad=None
   for _,samples,packed in prepare_chunks(rows,frames,runtime,8,pool):
    tokens={k:v.cpu() for k,v in packed[0].items() if k!='pixel_values'}
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if mode=='cached-vision' else None
    with profiler if profiler is not None else nullcontext():
     logits,_=batched_logits(runtime,head,samples,packed)
     with runtime._cached_image_features(packed[1]):logits.float().square().mean().backward()
    if profiler is not None:
     a.output.with_suffix('.profile.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))
    gradients={n:p.grad.cpu().clone() for n,p in parameters.items() if p.grad is not None}
    frozen=[n for n,p in parameters.items() if not p.requires_grad and p.grad is not None]
    data={'tokens':tokens,'logits':logits.detach().cpu(),'gradients':gradients,'frozen':frozen}
    references[mode]=data
 first,second=references.values()
 result={'input_exact':all(torch.equal(v,second['tokens'][k]) for k,v in first['tokens'].items()),
  'logits_exact':torch.equal(first['logits'],second['logits']),
  'gradients_exact':set(first['gradients'])==set(second['gradients']) and all(torch.equal(v,second['gradients'][k]) for k,v in first['gradients'].items()),
  'max_gradient_error':max(float((v-second['gradients'][k]).abs().max()) for k,v in first['gradients'].items()),
  'gradient_tensors':len(first['gradients']),'frozen_gradient_tensors':len(first['frozen'])+len(second['frozen']), 'optimizer_steps':0}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 assert result['input_exact'] and result['logits_exact'] and result['gradients_exact'] and not result['frozen_gradient_tensors']


if __name__=='__main__':main()
