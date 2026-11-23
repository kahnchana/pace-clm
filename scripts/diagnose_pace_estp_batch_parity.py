#!/usr/bin/env python3
"""No-update diagnosis of variable-length batch numerical differences."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark_qwen35_2b_training import batched_logits,collate,encode_batch_vision
from train_pace_estp_sft_batched import MixedRows
from train_jevany_streaming import EpisodeFrames
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.jevany_streaming import PointerHead,controller_logits
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl import mask_padding_states
from prepare_pace_estp_sft import read,write


def main(run):
    torch.cuda.set_device(0);torch.set_num_threads(1)
    cp=run/'profiles/batch-48/train/step-000000'
    runtime,_=load_2b_controller(checkpoint=str(cp/'controller_adapter'));install_cached_vision_inputs(runtime)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(cp/'pointer_head.pt',map_location='cpu',weights_only=True));head.eval()
    dataset=MixedRows(run/'data');rows=dataset.batch(0,0)
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=128,defer_vision=True)
    samples=[frames.prepare(r,cpu_only=True) for r in rows]
    ordered=sorted(zip(rows,samples),key=lambda t:t[1][0]['input_ids'].shape[1]);chosen=[ordered[0],ordered[-1]]
    samples=[s for _,s in chosen];encode_batch_vision(runtime,samples,8)
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    result={}
    def check(name):
        packed,visions,lengths=collate(samples,runtime);packed={k:v.to(runtime.device) for k,v in packed.items()}
        with torch.no_grad():batch,_=batched_logits(runtime,head,samples,(packed,visions,lengths))
        singles=[];padded=[]
        for inputs,vision,options,decision in samples:
            inputs={k:v.to(runtime.device) for k,v in inputs.items()}
            with torch.no_grad():single=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False)
            singles.append(single)
            padded_inputs=dict(inputs)
            for key,fill in [('input_ids',runtime.tokenizer.pad_token_id),('attention_mask',0),('mm_token_type_ids',0)]:
                padded_inputs[key]=F.pad(inputs[key],(0,max(lengths)-inputs[key].shape[1]),value=fill)
            with torch.no_grad():value=controller_logits(runtime,head,padded_inputs,vision,options,decision,train_decoder=False)
            padded.append(value)
        singles=torch.stack(singles);padded=torch.stack(padded)
        info=dict(lengths=lengths,batch=batch.float().tolist(),single=singles.float().tolist(),padded=padded.float().tolist(),
            batch_vs_single_probability=float((batch.float().softmax(-1)-singles.float().softmax(-1)).abs().max()),
            batch_vs_padded_probability=float((batch.float().softmax(-1)-padded.float().softmax(-1)).abs().max()))
        result[name]=info;write(run/'parity-diagnosis.json',result);print(name,json.dumps(info),flush=True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_math_sdp(False)
    check('efficient_sdpa')
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(True)
    original=F.scaled_dot_product_attention
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    check('native_full_accumulation')
    def sdpa_fp32(q,k,v,*args,**kwargs):
        if kwargs.get('attn_mask') is not None and kwargs['attn_mask'].dtype!=torch.bool:kwargs['attn_mask']=kwargs['attn_mask'].float()
        return original(q.float(),k.float(),v.float(),*args,**kwargs).to(q.dtype)
    F.scaled_dot_product_attention=sdpa_fp32
    check('fp32_sdpa')
    F.scaled_dot_product_attention=original
    for module in runtime.model.modules():
        if hasattr(module,'chunk_gated_delta_rule'):module.chunk_gated_delta_rule=qwen.torch_chunk_gated_delta_rule
    runtime.model.float()
    check('fp32_decoder')



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
