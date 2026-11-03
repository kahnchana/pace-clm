#!/usr/bin/env python3
"""Single-seed PACE/JevAny pointer pilot or full-exposure training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STREAM = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM / "src")]

from stream_gemma.inference.qwen_stream import QwenRuntime
from pace_clm.experiment import decode_indices
from pace_clm.jevany_streaming import (
    ACTIONS, PointerHead, controller_logits, decision_inputs, decision_messages,
)


def pilot_rows(path: Path) -> list[dict]:
    choices: dict[str, list[dict]] = {action: [] for action in ACTIONS}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("memory") and len(row["turns"]) <= 3 and len(choices[row["label"]]) < 3:
                choices[row["label"]].append(row)
            if all(len(group) == 3 for group in choices.values()):
                break
    if not all(len(group) == 3 for group in choices.values()):
        raise ValueError("pilot requires three short memory examples per state")
    return [row for action in ACTIONS for row in choices[action]]


def row_stream(path: Path, rank: int, world: int):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % world == rank:
                yield json.loads(line)


class EpisodeFrames:
    def __init__(self, runtime, *, response_blind: bool):
        self.runtime = runtime
        self.response_blind = response_blind
        self.episode_id = None
        self.frames = {}

    def prepare(self, row: dict):
        if row["episode_id"] != self.episode_id:
            self.episode_id = row["episode_id"]
            self.frames = {}
        indices = [int(turn["frame_index"]) for turn in row["turns"]]
        missing = [index for index in indices if index not in self.frames]
        if missing:
            images = self.runtime._resize(decode_indices(row["video_path"], missing))
            frames = [self.runtime.encode_frame(image) for image in images]
            self.runtime.encode_frames_features(frames)
            self.frames.update(zip(missing, zip(images, frames, strict=True), strict=True))
        images = [self.frames[index][0] for index in indices]
        vision = [self.frames[index][1] for index in indices]
        messages = decision_messages(row, images, response_blind=self.response_blind)
        inputs, options, decision = decision_inputs(self.runtime, messages, vision)
        if int((inputs["input_ids"] == self.runtime.processor.image_token_id).sum()) != 144 * len(indices):
            raise ValueError("visual slot admission failure")
        return inputs, vision, options, decision


def sync_gradients(parameters: list[torch.nn.Parameter], world: int):
    if world == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def save_checkpoint(out: Path, runtime, head, optimizer, *, step: int, seen: int,
                    rank: int, world: int, mode: str, seed: int):
    target = out / f"step-{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
               target / f"rng.rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        if mode in ("J1", "J2"):
            runtime.model.save_pretrained(target / "controller_adapter")
        torch.save(head.state_dict(), target / "pointer_head.pt")
        torch.save({"optimizer": optimizer.state_dict(), "step": step,
                    "seen_per_rank": seen, "world": world, "mode": mode,
                    "seed": seed}, target / "training_state.pt")
        (out / "latest.json").write_text(json.dumps({"step": step, "checkpoint": str(target)}) + "\n")
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("J0", "J1", "J2"), required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/jevany/manifests/train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-6)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--stop-after-steps", type=int, default=0,
                        help="Stop after a saved full-run checkpoint for a closed-loop gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.output / "latest.json").read_text()) if args.resume else None
    previous_dir = Path(previous["checkpoint"]) if previous else None
    pilot = pilot_rows(args.manifest) if args.stage == "pilot" else None
    if pilot is not None and rank == 0:
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in pilot)
        (args.output / "pilot_manifest.jsonl").write_text(body)
        (args.output / "pilot_manifest.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    runtime = QwenRuntime.load(
        checkpoint=str(previous_dir / "controller_adapter") if previous_dir and args.mode in ("J1", "J2")
                   else str((STREAM / "outputs/lora-qwen-w1-long-neg").resolve()),
        model_name="Qwen/Qwen3.5-35B-A3B",
        model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        frame_num_tokens=144, generation_prefix_mode="direct",
    )
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    if args.mode in ("J1", "J2"):
        for name, parameter in runtime.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.requires_grad_(True)
        runtime.model.train()
        runtime._conditional_model().model.visual.eval()
        runtime.model.gradient_checkpointing_enable()
        runtime.model.enable_input_require_grads()
    backbone_parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    if args.mode in ("J1", "J2") and not backbone_parameters:
        raise ValueError("J1 has no trainable LoRA parameters")
    if args.mode == "J0" and backbone_parameters:
        raise ValueError("J0 unexpectedly trains decoder parameters")
    params = list(head.parameters()) + backbone_parameters
    groups = [{"params": list(head.parameters()), "lr": args.head_lr}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(groups)
    resumed_presentations = 0
    resumed_steps = 0
    if previous_dir:
        state = torch.load(previous_dir / "training_state.pt", map_location="cpu", weights_only=False)
        if (state["world"], state["mode"], state["seed"]) != (world, args.mode, args.seed):
            raise ValueError("resume world/mode/seed mismatch")
        head.load_state_dict(torch.load(previous_dir / "pointer_head.pt", map_location="cpu",
                                        weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        rng = torch.load(previous_dir / f"rng.rank{rank}.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        resumed_presentations = int(state["seen_per_rank"])
        resumed_steps = int(state["step"])
    frames = EpisodeFrames(runtime, response_blind=args.mode == "J2")
    presentations = resumed_presentations
    steps = resumed_steps
    losses = []
    labels = Counter()
    started = time.monotonic()
    epochs = 3 if pilot is not None else 1
    for epoch in range(epochs):
        records = pilot[rank::world] if pilot is not None else row_stream(args.manifest, rank, world)
        for local_index, row in enumerate(records):
            if args.resume and epoch == 0 and local_index < resumed_presentations:
                continue
            inputs, vision, options, decision = frames.prepare(row)
            # Gradient checkpointing replays the forward during backward. Keep
            # the frozen vision cache installed for both passes, or that replay
            # silently re-encodes every frame in the context window.
            with runtime._cached_image_features(vision):
                logits = controller_logits(runtime, head, inputs, vision, options, decision,
                                           train_decoder=args.mode in ("J1", "J2"))
                target = torch.tensor([ACTIONS.index(row["label"])], device=runtime.device)
                loss = F.cross_entropy(logits.unsqueeze(0), target)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at presentation {presentations}")
                (loss / args.accumulation).backward()
            presentations += 1
            labels[row["label"]] += 1
            losses.append(float(loss.detach().cpu()))
            if presentations % args.accumulation == 0:
                sync_gradients(params, world)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if rank == 0 and steps % 25 == 0:
                    print(json.dumps({"step": steps, "seen_per_rank": presentations,
                                      "recent_loss": sum(losses[-25:]) / min(25, len(losses)),
                                      "seconds": time.monotonic() - started}), flush=True)
                if args.stage == "full" and (steps == 25 or steps % args.save_every == 0):
                    save_checkpoint(args.output, runtime, head, optimizer, step=steps,
                                    seen=presentations, rank=rank, world=world,
                                    mode=args.mode, seed=args.seed)
                if args.stop_after_steps and steps >= args.stop_after_steps:
                    if args.stage != "full":
                        raise ValueError("stop-after-steps requires a full run")
                    if steps != 25 and steps % args.save_every != 0:
                        save_checkpoint(args.output, runtime, head, optimizer, step=steps,
                                        seen=presentations, rank=rank, world=world,
                                        mode=args.mode, seed=args.seed)
                    if rank == 0:
                        print(json.dumps({"gate_checkpoint_step": steps,
                                          "seen_per_rank": presentations}), flush=True)
                    return
    if presentations % args.accumulation:
        sync_gradients(params, world)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    save_checkpoint(args.output, runtime, head, optimizer, step=steps,
                    seen=presentations, rank=rank, world=world, mode=args.mode, seed=args.seed)
    summary = {"mode": args.mode, "stage": args.stage, "seed": args.seed,
               "world": world, "presentations_per_rank": presentations,
               "resumed_presentations_per_rank": resumed_presentations,
               "global_presentations": presentations * world,
               "optimizer_steps": steps, "labels_per_rank": dict(labels),
               "first_loss": losses[0], "last_loss": losses[-1],
               "mean_loss_first_nine": sum(losses[:9]) / min(9, len(losses)),
               "mean_loss_last_nine": sum(losses[-9:]) / min(9, len(losses)),
               "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "seconds": time.monotonic() - started}
    (args.output / f"summary.rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
