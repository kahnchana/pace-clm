#!/usr/bin/env python3
"""Real-model PACE/JevAny bridge smoke with one small gradient step."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STREAM = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM / "src")]

from stream_gemma.inference.qwen_stream import QwenRuntime
from pace_clm.experiment import decode_indices
from pace_clm.jevany_streaming import (
    ACTIONS, PointerHead, controller_logits, decision_inputs, decision_messages,
)


def choose_rows(path: Path) -> list[dict]:
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["label"] not in selected and len(row["turns"]) <= 2:
                selected[row["label"]] = row
            if len(selected) == len(ACTIONS):
                break
    if set(selected) != set(ACTIONS):
        raise ValueError(f"missing smoke classes: {set(ACTIONS) - set(selected)}")
    return [selected[action] for action in ACTIONS]


def choose_long_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if len(row["turns"]) == 32 and row.get("memory"):
                return row
    raise ValueError("no full-window row with memory in manifest")


def prepare(runtime, row: dict):
    frame_ids = [int(turn["frame_index"]) for turn in row["turns"]]
    if frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1)):
        raise ValueError("episode frame indices are not contiguous")
    images = runtime._resize(decode_indices(row["video_path"], frame_ids))
    vision = [runtime.encode_frame(image) for image in images]
    runtime.encode_frames_features(vision)
    messages = decision_messages(row, images)
    inputs, options, decision = decision_inputs(runtime, messages, vision)
    expected_visual = sum(frame.num_tokens for frame in vision)
    actual_visual = int((inputs["input_ids"] == runtime.processor.image_token_id).sum())
    if actual_visual != expected_visual:
        raise ValueError(f"visual slots {actual_visual} != {expected_visual}")
    return inputs, vision, options, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/jevany/manifests/train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/jevany/smoke/report.json")
    parser.add_argument("--long-window", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.monotonic()
    rows = [choose_long_row(args.manifest)] if args.long_window else choose_rows(args.manifest)
    runtime = QwenRuntime.load(
        checkpoint=str((STREAM / "outputs/lora-qwen-w1-long-neg").resolve()),
        model_name="Qwen/Qwen3.5-35B-A3B",
        model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        frame_num_tokens=144, generation_prefix_mode="direct",
    )
    hidden = int(runtime.model.config.text_config.hidden_size)
    head = PointerHead(hidden).to(runtime.device)
    data = []
    for row in rows:
        inputs, vision, options, decision = prepare(runtime, row)
        data.append((row, inputs, vision, options, decision))
    with torch.no_grad():
        initial = [controller_logits(runtime, head, inputs, vision, options, decision,
                                     train_decoder=False).detach().cpu().tolist()
                   for _, inputs, vision, options, decision in data]
    trainable = []
    for name, parameter in runtime.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(True)
            trainable.append(name)
    if not trainable or any("visual" in name for name in trainable):
        raise ValueError(f"LoRA parameter audit failed: {trainable[:5]}")
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([
        {"params": list(head.parameters()), "lr": 1e-4},
        {"params": [p for p in runtime.model.parameters() if p.requires_grad], "lr": 2e-6},
    ])
    row, inputs, vision, options, decision = data[0]
    head.train()
    optimizer.zero_grad(set_to_none=True)
    logits = controller_logits(runtime, head, inputs, vision, options, decision,
                               train_decoder=True)
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor(
        [ACTIONS.index(row["label"])], device=runtime.device))
    loss.backward()
    lora_grads = [float(p.grad.float().abs().sum().item()) for p in runtime.model.parameters()
                  if p.requires_grad and p.grad is not None]
    if not torch.isfinite(loss) or not any(value > 0 for value in lora_grads):
        raise ValueError("loss or LoRA gradients invalid")
    optimizer.step()
    result = {
        "labels": [row["label"] for row in rows],
        "frame_counts": [len(row["turns"]) for row in rows],
        "input_tokens": [int(inputs["input_ids"].shape[1]) for _, inputs, *_ in data],
        "initial_logits": initial,
        "loss": float(loss.detach().cpu()),
        "trainable_lora_tensors": len(trainable),
        "nonzero_lora_gradient_tensors": sum(value > 0 for value in lora_grads),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
