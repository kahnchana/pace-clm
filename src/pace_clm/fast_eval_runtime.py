"""Pinned MoE runtime with direct GPU placement; no inference-policy changes."""
import json
import os
import time
import torch
from stream_gemma.config import QWEN_MODEL_NAME,QWEN_MODEL_REVISION
from stream_gemma.inference.qwen_stream import QwenRuntime
from stream_gemma.inference.qwen_adapters import apply_adapter_scale


class DirectGPUQwenRuntime(QwenRuntime):
    @classmethod
    def load(cls,*,checkpoint=None,model_name=QWEN_MODEL_NAME,model_revision=QWEN_MODEL_REVISION,
             frame_fps=1,frame_num_tokens=144,adapter_scale=1.,generation_prefix_mode='direct'):
        from peft import PeftModel
        from transformers import AutoProcessor,Qwen3_5MoeForConditionalGeneration
        if int(frame_fps)!=1 or generation_prefix_mode not in {'native','direct'}:raise ValueError('native runtime input contract changed')
        if not torch.cuda.is_available():raise RuntimeError('direct GPU runtime requires the assigned GPU')
        local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device(f'cuda:{local}')
        started=time.monotonic();offline=os.environ.get('HF_HUB_OFFLINE')=='1'
        model=Qwen3_5MoeForConditionalGeneration.from_pretrained(model_name,revision=model_revision,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm',
            device_map={'':str(device)},local_files_only=offline)
        text_cfg=getattr(model.config,'text_config',None)
        if text_cfg is not None:text_cfg._experts_implementation='grouped_mm'
        model.config._experts_implementation='grouped_mm'
        if checkpoint:
            model=PeftModel.from_pretrained(model,checkpoint,is_trainable=False)
            apply_adapter_scale(model,adapter_scale)
        elif adapter_scale!=1.:raise ValueError('adapter scale requires a checkpoint')
        model.requires_grad_(False)
        if any(p.device!=device for p in model.parameters()):raise RuntimeError('unexpected CPU/offloaded model parameter')
        model.eval()
        processor=AutoProcessor.from_pretrained(model_name,revision=model_revision,local_files_only=offline)
        runtime=cls(model,processor.tokenizer,processor,device,int(frame_fps),frame_num_tokens=frame_num_tokens,
            generation_prefix_mode=generation_prefix_mode,adapter_scale=float(adapter_scale))
        runtime.direct_load_seconds=time.monotonic()-started
        print(json.dumps({'direct_gpu_moe_load_seconds':runtime.direct_load_seconds,'local_rank':local,
            'model_name':model_name,'revision':model_revision}),flush=True)
        return runtime
