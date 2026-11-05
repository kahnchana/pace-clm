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
from collections import Counter, OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STREAM = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM / "src")]

from stream_gemma.inference.qwen_stream import QwenRuntime
from pace_clm.experiment import decode_indices
from pace_clm.qwen_controller_runtime import MODEL_REVISIONS, load_dense_controller
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
    def __init__(self, runtime, *, response_blind: bool, cache_frames: int = 0,
                 defer_vision: bool = False):
        self.runtime = runtime
        self.response_blind = response_blind
        if cache_frames < 0:
            raise ValueError("cache_frames must be nonnegative")
        self.cache_frames = cache_frames
        self.defer_vision = defer_vision
        self.episode_id = None
        self.frames = OrderedDict()

    def prepare(self, row: dict, timings: dict[str, float] | None = None,
                prefetched: dict | None = None, *, cpu_only: bool = False):
        def mark(name: str, start: float):
            if timings is not None:
                torch.cuda.synchronize(self.runtime.device)
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        if not self.cache_frames and row["episode_id"] != self.episode_id:
            self.episode_id = row["episode_id"]
            self.frames = OrderedDict()
        indices = [int(turn["frame_index"]) for turn in row["turns"]]
        keys = [(row["video_path"], index) if self.cache_frames else index for index in indices]
        missing = [index for index, key in zip(indices, keys) if key not in self.frames]
        if missing:
            start = time.perf_counter()
            path = row["video_path"]
            if prefetched is not None and all((path, index) in prefetched for index in missing):
                images = [prefetched[path, index] for index in missing]
            else:
                images = self.runtime._resize(decode_indices(path, missing))
            mark("decode_resize", start)
            start = time.perf_counter()
            frames = [self.runtime.encode_frame(image) for image in images]
            mark("image_prepare", start)
            start = time.perf_counter()
            if not self.defer_vision and not cpu_only:
                self.runtime.encode_frames_features(frames)
            mark("vision_encode", start)
            for index, pair in zip(missing, zip(images, frames, strict=True), strict=True):
                key = (row["video_path"], index) if self.cache_frames else index
                self.frames[key] = pair
        start = time.perf_counter()
        images = [self.frames[key][0] for key in keys]
        vision = [self.frames[key][1] for key in keys]
        if self.cache_frames:
            for key in keys:
                self.frames.move_to_end(key)
            while len(self.frames) > self.cache_frames:
                self.frames.popitem(last=False)
        messages = decision_messages(row, images, response_blind=self.response_blind)
        inputs, options, decision = decision_inputs(
            self.runtime, messages, vision, device="cpu" if cpu_only else None)
        if int((inputs["input_ids"] == self.runtime.processor.image_token_id).sum()) != 144 * len(indices):
            raise ValueError("visual slot admission failure")
        mark("input_assembly", start)
        return inputs, vision, options, decision


def sync_gradients(parameters: list[torch.nn.Parameter], world: int):
    if world == 1:
        return
    if dist.get_backend() == "gloo":
        # This node's CUDA/NCCL reduction returns error 802. Reduce one packed
        # CPU buffer to keep the eight-GPU gradient average identical.
        sizes = [parameter.numel() for parameter in parameters]
        packed = torch.empty(sum(sizes), dtype=torch.float32)
        offset = 0
        for parameter, size in zip(parameters, sizes, strict=True):
            if parameter.grad is None:
                packed[offset:offset + size].zero_()
            else:
                packed[offset:offset + size].copy_(parameter.grad.detach().reshape(-1).float().cpu())
            offset += size
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(world)
        offset = 0
        for parameter, size in zip(parameters, sizes, strict=True):
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            parameter.grad.copy_(packed[offset:offset + size].view_as(parameter).to(parameter.device))
            offset += size
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def save_checkpoint(out: Path, runtime, head, optimizer, *, step: int, seen: int,
                    rank: int, world: int, mode: str, seed: int, model_name: str,
                    model_revision: str):
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
                    "seed": seed, "model_name": model_name,
                    "model_revision": model_revision}, target / "training_state.pt")
        (target / "model_identity.json").write_text(json.dumps({
            "model_name": model_name, "model_revision": model_revision,
            "controller_adapter": "controller_adapter", "pointer_head": "pointer_head.pt",
        }, indent=2) + "\n")
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
    parser.add_argument("--controller-model", choices=("promoted-35b", "official-0p8b", "official-2b", "official-4b"),
                        default="promoted-35b")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(os.environ.get("PACE_DISTRIBUTED_BACKEND", "nccl"))
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
    if args.controller_model != "promoted-35b":
        if args.mode != "J1":
            raise ValueError("official dense Qwen experiment uses J1 history")
        model_name = {
            "official-0p8b": "Qwen/Qwen3.5-0.8B",
            "official-2b": "Qwen/Qwen3.5-2B",
            "official-4b": "Qwen/Qwen3.5-4B",
        }[args.controller_model]
        model_revision = MODEL_REVISIONS[model_name]
        runtime, matched_modules = load_dense_controller(
            model_name=model_name, model_revision=model_revision,
            checkpoint=str(previous_dir / "controller_adapter") if previous_dir else None,
            fresh_lora=previous_dir is None,
        )
        if rank == 0 and matched_modules:
            (args.output / "lora_modules.json").write_text(json.dumps(matched_modules, indent=2) + "\n")
    else:
        runtime = QwenRuntime.load(
            checkpoint=str(previous_dir / "controller_adapter") if previous_dir and args.mode in ("J1", "J2")
                       else str((STREAM / "outputs/lora-qwen-w1-long-neg").resolve()),
            model_name="Qwen/Qwen3.5-35B-A3B",
            model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
            frame_num_tokens=144, generation_prefix_mode="direct",
        )
        model_name, model_revision = "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"
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
    if rank == 0:
        (args.output / "trainable_counts.json").write_text(json.dumps({
            "model_name": model_name, "model_revision": model_revision,
            "lora_parameters": sum(p.numel() for p in backbone_parameters),
            "pointer_parameters": sum(p.numel() for p in head.parameters()),
        }, indent=2) + "\n")
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
        if state.get("model_name", model_name) != model_name or state.get("model_revision", model_revision) != model_revision:
            raise ValueError("resume model identity mismatch")
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
                                    mode=args.mode, seed=args.seed,
                                    model_name=model_name, model_revision=model_revision)
                if args.stop_after_steps and steps >= args.stop_after_steps:
                    if args.stage != "full":
                        raise ValueError("stop-after-steps requires a full run")
                    if steps != 25 and steps % args.save_every != 0:
                        save_checkpoint(args.output, runtime, head, optimizer, step=steps,
                                        seen=presentations, rank=rank, world=world,
                                        mode=args.mode, seed=args.seed,
                                        model_name=model_name, model_revision=model_revision)
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
                    seen=presentations, rank=rank, world=world, mode=args.mode, seed=args.seed,
                    model_name=model_name, model_revision=model_revision)
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
