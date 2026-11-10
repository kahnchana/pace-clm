#!/usr/bin/env python3
"""Test BF16 LoRA compute against the strong SFT action distribution; no updates."""
import json,math,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import torch
from peft.helpers import disable_input_dtype_casting as disable_lora_input_dtype_casting
from jevany_rl_workers import prepare_chunks
from train_jevany_streaming import EpisodeFrames
from pace_clm.backbone import language_backbone
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl import parity_metrics


def main():
 torch.set_num_threads(1)
 output=Path('outputs/jevany-2b-rl/lora-precision-proof.json')
 checkpoint=Path('outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243')
 traces=json.loads(Path('outputs/jevany-2b-rl/profile-7p1-v8/batch-000000/scored.json').read_text())
 steps=[s for t in traces for s in t['steps']];steps.sort(key=lambda s:-sum(-math.exp(v)*v for v in s['old_logp']))
 steps=steps[:8];rows=[s['row'] for s in steps];length=max(r['padded_length'] for r in rows)
 for row in rows:row['padded_length']=length;row['physical_batch']=20
 runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'));install_cached_vision_inputs(runtime)
 head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
 head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
 frames=EpisodeFrames(runtime,response_blind=False,cache_frames=1024,defer_vision=True);frames.physical_batch=20
 values={};timings={}
 with ThreadPoolExecutor(max_workers=1) as pool,torch.no_grad():
  for _,samples,packed in prepare_chunks(rows,frames,runtime,20,pool):
   for mode in ('sft-fp32-lora','bf16-lora'):
    torch.cuda.synchronize();started=time.monotonic()
    context=disable_lora_input_dtype_casting(runtime.model) if mode=='bf16-lora' else nullcontext()
    with runtime._cached_image_features(packed[1]),context,torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16-lora'):
     hidden=language_backbone(runtime.model)(**packed[0],use_cache=False,return_dict=True).last_hidden_state
    # Keep the trained pointer head in FP32 in both cases.
    logits=torch.stack([head(hidden[i,s[3]].float(),hidden[i,s[2]].float()) for i,s in enumerate(samples)])
    values[mode]=logits.float().log_softmax(-1).detach()
    torch.cuda.synchronize();timings[mode]=time.monotonic()-started
 old,new=values.values();actions=torch.tensor([s['action'] for s in steps],device=runtime.device)
 metrics=parity_metrics(new,old,actions);metrics['mean_kl']=metrics['kl_sum']/len(rows)
 accepted=metrics['probability_max']<=.005 and metrics['kl_max']<=1e-4 and metrics['mean_kl']<=1e-5 and metrics['selected_logp_max']<=.05
 result={'accepted':accepted,'metrics':metrics,'seconds':timings,'optimizer_steps':0,
         'lora_master_dtypes':sorted({str(p.dtype) for n,p in runtime.model.named_parameters() if 'lora_' in n})}
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
