#!/usr/bin/env python3
"""Short, sequential eight-rank throughput benchmark for the 2B controller."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from train_jevany_streaming import EpisodeFrames, row_stream, sync_gradients
from pace_clm.backbone import language_backbone
from pace_clm.jevany_streaming import ACTIONS, PointerHead, controller_logits
from pace_clm.qwen_controller_runtime import load_2b_controller


def collate(samples: list[tuple], runtime):
    lengths = [item[0]["input_ids"].shape[1] for item in samples]
    longest = max(lengths)
    pad_id = runtime.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad token")
    result = {}
    for key, fill in (("input_ids", pad_id), ("attention_mask", 0),
                      ("mm_token_type_ids", 0)):
        tensors = [item[0][key][0] for item in samples]
        result[key] = torch.stack([F.pad(t, (0, longest - len(t)), value=fill)
                                   for t in tensors])
    for key in ("pixel_values", "image_grid_thw"):
        result[key] = torch.cat([item[0][key] for item in samples], dim=0)
    visions = [frame for _, vision, _, _ in samples for frame in vision]
    return result, visions, lengths


def batched_logits(runtime, head, samples: list[tuple]):
    inputs, visions, lengths = collate(samples, runtime)
    with runtime._cached_image_features(visions):
        hidden = language_backbone(runtime.model)(
            **inputs, use_cache=False, return_dict=True).last_hidden_state
        logits = torch.stack([
            head(hidden[i, item[3]].float(), hidden[i, item[2]].float())
            for i, item in enumerate(samples)
        ])
        return logits, lengths


def packed_nccl_sync(parameters: list[torch.nn.Parameter], world: int):
    if world == 1:
        return
    sizes = [parameter.numel() for parameter in parameters]
    packed = torch.empty(sum(sizes), dtype=torch.float32, device=parameters[0].device)
    offset = 0
    for parameter, size in zip(parameters, sizes, strict=True):
        if parameter.grad is None:
            packed[offset:offset + size].zero_()
        else:
            packed[offset:offset + size].copy_(parameter.grad.detach().reshape(-1))
        offset += size
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed.div_(world)
    offset = 0
    for parameter, size in zip(parameters, sizes, strict=True):
        if parameter.grad is None:
            parameter.grad = torch.empty_like(parameter)
        parameter.grad.copy_(packed[offset:offset + size].view_as(parameter))
        offset += size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--throughput-only", action="store_true",
                        help="Avoid per-phase CUDA synchronizations for a less intrusive throughput run")
    args = parser.parse_args()
    if args.warmup < 0 or args.updates < 1:
        raise ValueError("warmup must be nonnegative and updates positive")
    accumulation = max(1, 4 // args.microbatch)
    per_rank_batch = args.microbatch * accumulation
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(42)
    dist.init_process_group(args.backend) if world > 1 else None
    runtime, _ = load_2b_controller(fresh_lora=True)
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    for name, parameter in runtime.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(True)
    runtime.model.train()
    runtime._conditional_model().model.visual.eval()
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    parameters = list(head.parameters()) + [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": list(head.parameters()), "lr": 1e-4},
        {"params": [p for p in runtime.model.parameters() if p.requires_grad], "lr": 2e-6},
    ])
    frames = EpisodeFrames(runtime, response_blind=False)
    records = iter(row_stream(args.manifest, rank, world))
    totals = {}
    lengths_total = 0
    padded_total = 0
    step_times = []
    measured_presentations = 0
    measured_started = None
    measured_start_utc = None

    def mark(name, started, enabled):
        if args.throughput_only:
            return
        torch.cuda.synchronize(runtime.device)
        if enabled:
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - started

    for step in range(args.warmup + args.updates):
        enabled = step >= args.warmup
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats(runtime.device)
            if world > 1:
                dist.barrier()
            measured_started = time.perf_counter()
            measured_start_utc = datetime.now(timezone.utc).isoformat()
        started_step = time.perf_counter()
        for _ in range(accumulation):
            samples = []
            labels = []
            for _ in range(args.microbatch):
                row = next(records)
                labels.append(ACTIONS.index(row["label"]))
                samples.append(frames.prepare(row, totals if enabled and not args.throughput_only else None))
            started = time.perf_counter()
            if args.microbatch == 1:
                item = samples[0]
                with runtime._cached_image_features(item[1]):
                    logits = controller_logits(runtime, head, *item, train_decoder=True).unsqueeze(0)
                lengths = [item[0]["input_ids"].shape[1]]
            else:
                logits, lengths = batched_logits(runtime, head, samples)
            mark("forward", started, enabled)
            if enabled:
                lengths_total += sum(lengths)
                padded_total += max(lengths) * len(lengths) - sum(lengths)
            target = torch.tensor(labels, device=runtime.device)
            loss = F.cross_entropy(logits, target) * (args.microbatch / per_rank_batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            started = time.perf_counter()
            # Cached vision features must remain installed for checkpoint replay.
            with runtime._cached_image_features([frame for item in samples for frame in item[1]]):
                loss.backward()
            mark("backward", started, enabled)
        started = time.perf_counter()
        if args.backend == "nccl":
            packed_nccl_sync(parameters, world)
        else:
            sync_gradients(parameters, world)
        mark("gradient_sync", started, enabled)
        started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mark("optimizer", started, enabled)
        if enabled:
            step_times.append(time.perf_counter() - started_step)
            measured_presentations += per_rank_batch
        if rank == 0:
            print(json.dumps({"step": step + 1, "warmup": not enabled,
                              "microbatch": args.microbatch,
                              "backend": args.backend,
                              "elapsed_seconds": time.perf_counter() - started_step}),
                  flush=True)
    torch.cuda.synchronize(runtime.device)
    if world > 1:
        dist.barrier()
    measured_seconds = time.perf_counter() - measured_started
    measured_end_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "rank": rank, "backend": args.backend, "microbatch": args.microbatch,
        "global_batch": world * per_rank_batch,
        "throughput_only": args.throughput_only,
        "accumulation": accumulation, "warmup_updates": args.warmup,
        "measured_updates": args.updates, "measured_presentations": measured_presentations,
        "phase_seconds": totals, "step_seconds": step_times,
        "measured_seconds": measured_seconds,
        "measured_start_utc": measured_start_utc,
        "measured_end_utc": measured_end_utc,
        "token_count": lengths_total, "padding_tokens": padded_total,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 1024**3,
    }
    results = [None] * world
    if world > 1:
        dist.all_gather_object(results, result)
    else:
        results = [result]
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        wall_seconds = max(item["measured_seconds"] for item in results)
        print(json.dumps({"output": str(args.output), "updates_per_second": args.updates / wall_seconds,
                          "presentations_per_second": args.updates * world * per_rank_batch / wall_seconds,
                          "max_peak_allocated_gib": max(item["peak_allocated_gib"] for item in results),
                          "max_peak_reserved_gib": max(item["peak_reserved_gib"] for item in results)}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
