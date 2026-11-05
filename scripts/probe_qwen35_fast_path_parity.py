#!/usr/bin/env python3
"""Compare a short Qwen3.5 training step across optional kernel environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_qwen35_2b_training import batched_logits
from train_jevany_streaming import EpisodeFrames, row_stream
from pace_clm.jevany_streaming import ACTIONS, PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    runtime, _ = load_2b_controller(
        checkpoint=str(args.checkpoint / "controller_adapter") if args.checkpoint else None,
        fresh_lora=args.checkpoint is None)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    if args.checkpoint:
        head.load_state_dict(torch.load(args.checkpoint / "pointer_head.pt",
                                        map_location="cpu", weights_only=True))
    for name, parameter in runtime.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(True)
    runtime.model.train()
    for module in runtime.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    runtime._conditional_model().model.visual.eval()
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    rows = []
    source = row_stream(args.manifest, 0, 8)
    for _ in range(4):
        rows.append(next(source))
    frames = EpisodeFrames(runtime, response_blind=False, cache_frames=1024)
    samples = [frames.prepare(row) for row in rows]
    logits, lengths = batched_logits(runtime, head, samples)
    targets = torch.tensor([ACTIONS.index(row["label"]) for row in rows], device=runtime.device)
    loss = F.cross_entropy(logits, targets)
    with runtime._cached_image_features([frame for item in samples for frame in item[1]]):
        loss.backward()
    torch.cuda.synchronize()
    lora = next(parameter for name, parameter in runtime.model.named_parameters()
                if "lora_B" in name and parameter.grad is not None)
    result = {
        "loss": float(loss.detach()),
        "logits": logits.detach().float().cpu().tolist(),
        "lengths": lengths,
        "head_gradient_norm": float(head.q.weight.grad.float().norm()),
        "head_gradient_slice": head.q.weight.grad.flatten()[:32].float().cpu().tolist(),
        "lora_gradient_norm": float(lora.grad.float().norm()),
        "lora_gradient_slice": lora.grad.flatten()[:32].float().cpu().tolist(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: result[key] for key in ("loss", "lengths", "peak_allocated_gib")}))


if __name__ == "__main__":
    main()
