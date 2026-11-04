"""Load pinned official dense Qwen3.5 controllers with model-specific LoRA."""

from __future__ import annotations

import os

import torch


MODEL_NAME = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    MODEL_NAME: MODEL_REVISION,
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
    "down_proj", "in_proj_qkv", "in_proj_z", "out_proj",
)


def load_dense_controller(*, model_name: str, model_revision: str,
                          checkpoint: str | None = None, fresh_lora: bool = False):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from stream_gemma.inference.qwen_stream import QwenRuntime

    if bool(checkpoint) == bool(fresh_lora):
        raise ValueError("specify either a saved controller adapter or fresh LoRA")
    if MODEL_REVISIONS.get(model_name) != model_revision:
        raise ValueError("controller model or pinned revision mismatch")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 controller experiment requires an H200 GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, revision=model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=offline,
    )
    if fresh_lora:
        matched = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and (name.startswith("model.language_model.") or
                        name.startswith("language_model."))
                   and name.rsplit(".", 1)[-1] in LORA_TARGETS]
        if not matched:
            raise RuntimeError("no language-model LoRA projection modules matched")
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
            target_modules=matched, task_type="CAUSAL_LM",
        ))
    else:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        matched = []
    model.requires_grad_(False)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_name, revision=model_revision, local_files_only=offline,
    )
    runtime = QwenRuntime(model, processor.tokenizer, processor, device,
                          frame_num_tokens=144, generation_prefix_mode="direct")
    return runtime, matched


def load_2b_controller(*, checkpoint: str | None = None, fresh_lora: bool = False):
    """Compatibility entrypoint for the completed 2B experiment."""
    return load_dense_controller(model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                                 checkpoint=checkpoint, fresh_lora=fresh_lora)
