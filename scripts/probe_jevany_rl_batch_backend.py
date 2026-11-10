#!/usr/bin/env python3
"""Compare batch-invariant runtime candidates on uncertain real rollout states."""
import argparse
import contextlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from benchmark_qwen35_2b_training import batched_logits
from jevany_rl_workers import prepare_chunks, fp32_depthwise_convolutions
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--singleton-mask',action='store_true')
    p.add_argument('--trace-layers',action='store_true')
    p.add_argument('--fixed-batch',type=int,default=0)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
    traces=json.loads(a.traces.read_text())
    steps=[s for t in traces for s in t['steps']]
    steps.sort(key=lambda s:-sum(-math.exp(v)*v for v in s['old_logp']))
    rows=[s['row'] for s in steps[:8]]
    if a.singleton_mask or a.trace_layers:
        import copy
        rows=[copy.deepcopy(rows[0]) for _ in range(8)]
    length=max(r['padded_length'] for r in rows)
    for row in rows:row['padded_length']=length
    runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
    runtime.model.eval()
    for module in runtime.model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=1024,defer_vision=True)
    frames.physical_batch=a.fixed_batch
    linears=[(m,m.forward) for m in runtime._conditional_model().model.language_model.modules()
             if isinstance(m,torch.nn.Linear)]
    results=[]
    captured={}
    active_length=[0]
    hooks=[]
    if a.trace_layers:
        for name,module in runtime._conditional_model().model.language_model.named_modules():
            if not name or (list(module.children()) and not name.endswith(('linear_attn','self_attn'))):continue
            def hook(module,args,output,name=name):
                value=output[0] if isinstance(output,tuple) else output
                if not isinstance(value,torch.Tensor) or value.ndim<3:return
                value=value[0].detach()
                if isinstance(module,torch.nn.Conv1d):value=value.transpose(0,1)
                value=value[:active_length[0]]
                captured[name]=torch.cat([value[:4],value[-128:]]).float().cpu()
            hooks.append(module.register_forward_hook(hook))
    with ThreadPoolExecutor(max_workers=1) as pool:
        for variant in (('fixed-shape',) if a.fixed_batch else ('standard','fp32-conv') if a.trace_layers else ('standard','singleton-mask') if a.singleton_mask else ('standard','efficient','bmm','bmm-efficient','bmm-fp32-efficient')):
            if variant=='fp32-conv':fp32_depthwise_convolutions(runtime.model)
            if variant=='singleton-mask':
                from transformers.models.qwen3_5 import modeling_qwen3_5
                def mask(hidden_states,attention_mask):
                    if attention_mask is not None and attention_mask.shape[1]>1:
                        return (hidden_states*attention_mask[:,:,None]).to(hidden_states.dtype)
                    return hidden_states
                modeling_qwen3_5.apply_mask_to_padding_states=mask
            for module,original in linears:module.forward=original
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            if variant.startswith('bmm'):
                for module,original in linears:
                    def forward(x,module=module,original=original):
                        if x.ndim!=3 or module.bias is not None:return original(x)
                        weight=module.weight
                        if 'fp32' in variant:
                            result=torch.bmm(x.float(),weight.float().t().unsqueeze(0).expand(x.shape[0],-1,-1))
                        else:
                            result=torch.bmm(x,weight.t().unsqueeze(0).expand(x.shape[0],-1,-1))
                        return result.to(x.dtype)
                    module.forward=forward
            try:
                values={}
                layer_values={}
                context=(lambda:sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)) if 'efficient' in variant else contextlib.nullcontext
                for size in (1,8):
                    outputs=[]
                    with context(),torch.no_grad():
                        for _,samples,packed in prepare_chunks(rows[:1] if (a.singleton_mask or a.trace_layers) and size==1 else rows,frames,runtime,size,pool):
                            active_length[0]=packed[2][0]
                            logits,_=batched_logits(runtime,head,samples,packed)
                            outputs.append(F.log_softmax(logits.float(),-1).cpu())
                    values[size]=torch.cat(outputs)
                    layer_values[size]=dict(captured)
                    if (a.singleton_mask or a.trace_layers) and size==1:values[size]=values[size].expand(8,-1)
                ref,replay=values[1],values[8]
                result={'variant':variant,'max_logp_error':float((ref-replay).abs().max()),
                        'max_probability_error':float((ref.exp()-replay.exp()).abs().max()),
                        'max_kl':float((ref.exp()*(ref-replay)).sum(-1).max()),
                        'reference':ref.tolist(),'replay':replay.tolist()}
                if a.trace_layers:
                    result['layers']=[{'name':n,'max_abs':float((v-layer_values[8][n]).abs().max()),'mean_abs':float((v-layer_values[8][n]).abs().mean())} for n,v in layer_values[1].items()]

            except Exception as exc:
                result={'variant':variant,'error':repr(exc)}
            results.append(result)
            a.output.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
