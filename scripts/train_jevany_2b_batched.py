#!/usr/bin/env python3
"""Checkpointed eight-rank Qwen3.5-2B J1 training with physical batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmark_qwen35_2b_training import (
    batched_logits, chunk_conv1d_batches, collate, encode_batch_vision, packed_nccl_sync,
    predecode_batch,
)
from train_jevany_streaming import EpisodeFrames, row_stream, save_checkpoint, sync_gradients
from pace_clm.jevany_streaming import ACTIONS, PointerHead
from pace_clm.qwen_controller_runtime import MODEL_NAME, MODEL_REVISION, load_2b_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path,
                        help="JSONL plan with synchronized global_batch for each update")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=48)
    parser.add_argument("--microbatch-ratio", type=float, default=1.0,
                        help="maximum fraction of a planned per-rank batch in one forward/backward")
    parser.add_argument("--frame-cache", type=int, default=0)
    parser.add_argument("--vision-chunk", type=int, default=0)
    parser.add_argument("--predecode-workers", type=int, default=4)
    parser.add_argument("--prefetch-mode", choices=("prepare", "decode", "none"), default="prepare",
                        help="Prepare and collate the next batch on CPU by default")
    parser.add_argument("--prefetch-decode", action="store_true")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--conv-batch-chunk", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.microbatch < 1 or not 0 < args.microbatch_ratio <= 1 or
        args.save_every < 1 or args.log_every < 1 or args.predecode_workers < 0 or
        args.vision_chunk < 0 or args.conv_batch_chunk < 0):
        parser.error("microbatch and save-every must be positive; worker and chunk sizes nonnegative")
    prefetch_mode = "decode" if args.prefetch_decode else args.prefetch_mode
    if prefetch_mode != "none" and not args.predecode_workers:
        parser.error("prefetch requires predecode-workers")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 8:
        raise ValueError("this full-data recipe requires exactly eight local ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.output / "latest.json").read_text()) if args.resume else None
    previous_dir = Path(previous["checkpoint"]) if previous else None
    runtime, matched = load_2b_controller(
        checkpoint=str(previous_dir / "controller_adapter") if previous_dir else None,
        fresh_lora=previous_dir is None,
    )
    patched_convs = chunk_conv1d_batches(runtime.model, args.conv_batch_chunk) if args.conv_batch_chunk else 0
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    for name, parameter in runtime.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(True)
    runtime.model.train()
    runtime._conditional_model().model.visual.eval()
    runtime.model.gradient_checkpointing_enable()
    runtime.model.enable_input_require_grads()
    backbone = [p for p in runtime.model.parameters() if p.requires_grad]
    parameters = list(head.parameters()) + backbone
    optimizer = torch.optim.AdamW([
        {"params": list(head.parameters()), "lr": 1e-4},
        {"params": backbone, "lr": 2e-6},
    ])
    seen = 0
    step = 0
    if previous_dir:
        state = torch.load(previous_dir / "training_state.pt", map_location="cpu", weights_only=False)
        if (state["world"], state["mode"], state["seed"], state["model_revision"]) != (
            world, "J1", args.seed, MODEL_REVISION
        ):
            raise ValueError("checkpoint identity mismatch")
        head.load_state_dict(torch.load(previous_dir / "pointer_head.pt", map_location="cpu", weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        rng = torch.load(previous_dir / f"rng.rank{rank}.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        seen = int(state["seen_per_rank"])
        step = int(state["step"])
    if rank == 0:
        manifest_hash = hashlib.sha256()
        with args.manifest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                manifest_hash.update(chunk)
        recipe_path = args.output / "recipe.json"
        plan_hash = hashlib.sha256(args.batch_plan.read_bytes()).hexdigest() if args.batch_plan else None
        if args.resume:
            old_recipe = json.loads(recipe_path.read_text())
            if (old_recipe["manifest_sha256"], old_recipe.get("batch_plan_sha256")) != (
                manifest_hash.hexdigest(), plan_hash
            ):
                raise ValueError("resume manifest or batch-plan checksum mismatch")
        recipe_path.write_text(json.dumps({
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash.hexdigest(),
            "batch_plan": str(args.batch_plan) if args.batch_plan else None,
            "batch_plan_sha256": plan_hash,
            "model": MODEL_NAME, "revision": MODEL_REVISION,
            "physical_batch_per_gpu": "planned" if args.batch_plan else args.microbatch,
            "microbatch_ratio": args.microbatch_ratio,
            "global_batch": "planned" if args.batch_plan else args.microbatch * world,
            "frame_cache_per_gpu": args.frame_cache,
            "vision_chunk": args.vision_chunk,
            "predecode_workers": args.predecode_workers,
            "prefetch_decode": args.prefetch_decode,
            "prefetch_mode": prefetch_mode,
            "backend": args.backend,
            "conv_batch_chunk": args.conv_batch_chunk,
            "patched_convs": patched_convs,
            "seed": args.seed, "lora_modules": matched,
        }, indent=2) + "\n")
    frames = EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache,
                           defer_vision=args.vision_chunk > 0)
    cpu_frames = (EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache)
                  if prefetch_mode == "prepare" else None)
    decode_pool = ThreadPoolExecutor(max_workers=args.predecode_workers) if args.predecode_workers else None
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if prefetch_mode != "none" else None
    rows = islice(row_stream(args.manifest, rank, world), seen, None)
    if args.batch_plan:
        resumed_step = step
        resumed_seen = seen
        def batch_sizes():
            with args.batch_plan.open() as handle:
                for index, line in enumerate(handle):
                    if index < resumed_step:
                        continue
                    planned = json.loads(line)
                    size = int(planned["global_batch"])
                    if size < 1 or size % world:
                        raise ValueError(f"invalid planned global batch at update {index + 1}")
                    if index == resumed_step and int(planned["start_row"]) != resumed_seen * world:
                        raise ValueError("checkpoint progress does not match batch plan")
                    yield size // world
    else:
        def batch_sizes():
            while True:
                yield args.microbatch
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    if rank == 0:
        print(json.dumps({"training_start_utc": started_utc, "resumed_step": step,
                          "seen_per_rank": seen}), flush=True)
    labels = Counter()
    recent_losses = []
    planned_sizes = iter(batch_sizes())

    def next_batch():
        planned_per_rank = next(planned_sizes, None)
        if planned_per_rank is None:
            return None
        return planned_per_rank, list(islice(rows, planned_per_rank))

    def cache_images():
        return ({key: image for key, (image, _) in frames.frames.items()}
                if args.frame_cache else None)

    def prepare_on_cpu(batch_rows):
        cached_images = ({key: image for key, (image, _) in cpu_frames.frames.items()}
                         if args.frame_cache else None)
        decoded = predecode_batch(runtime, batch_rows, decode_pool, cached_images)
        samples = [cpu_frames.prepare(row, prefetched=decoded, cpu_only=True)
                   for row in batch_rows]
        chunk_size = max(1, int(len(batch_rows) * args.microbatch_ratio))
        return [(samples[start:start + chunk_size],
                 collate(samples[start:start + chunk_size], runtime))
                for start in range(0, len(samples), chunk_size)]

    def schedule(batch):
        if batch is None:
            return None
        planned_per_rank, batch_rows = batch
        if prefetch_mode == "prepare":
            future = prefetch_pool.submit(prepare_on_cpu, batch_rows)
        else:
            future = prefetch_pool.submit(predecode_batch, runtime, batch_rows,
                                          decode_pool, cache_images())
        return planned_per_rank, batch_rows, future

    pending = schedule(next_batch()) if prefetch_pool else None
    while True:
        step_started = time.monotonic()
        predecode_wait_seconds = 0.0
        prepare_wait_seconds = 0.0
        prepared_chunks = None
        if prefetch_pool:
            if pending is None:
                break
            planned_per_rank, batch_rows, future = pending
            waiting_started = time.monotonic()
            prepared = future.result()
            wait_seconds = time.monotonic() - waiting_started
            if prefetch_mode == "prepare":
                prepared_chunks = prepared
                prefetched = None
                prepare_wait_seconds = wait_seconds
            else:
                prefetched = prepared
                predecode_wait_seconds = wait_seconds
            next_item = (next_batch() if not args.stop_after_steps or
                         step + 1 < args.stop_after_steps else None)
            pending = schedule(next_item)
        else:
            batch = next_batch()
            if batch is None:
                break
            planned_per_rank, batch_rows = batch
            prefetched = (predecode_batch(runtime, batch_rows, decode_pool, cache_images())
                          if decode_pool else None)
        count = torch.tensor(len(batch_rows),
                             device=runtime.device if args.backend == "nccl" else "cpu",
                             dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = int(count.item())
        if total == 0:
            break
        if args.batch_plan and total != planned_per_rank * world:
            raise ValueError(f"batch plan expected {planned_per_rank * world} rows, got {total}")
        prepare_seconds = 0.0
        model_seconds = 0.0
        if batch_rows:
            chunk_size = max(1, int(len(batch_rows) * args.microbatch_ratio))
            loss_sum = 0.0
            for chunk_index, chunk_start in enumerate(range(0, len(batch_rows), chunk_size)):
                chunk_rows = batch_rows[chunk_start:chunk_start + chunk_size]
                preparing_started = time.monotonic()
                ready_collated = None
                if prepared_chunks is not None:
                    samples, cpu_collated = prepared_chunks[chunk_index]
                    encode_batch_vision(runtime, samples, args.vision_chunk or 8)
                    inputs, visions, lengths = cpu_collated
                    ready_collated = ({key: value.to(runtime.device) for key, value in inputs.items()},
                                      visions, lengths)
                else:
                    samples = [frames.prepare(row, prefetched=prefetched) for row in chunk_rows]
                    if args.vision_chunk:
                        encode_batch_vision(runtime, samples, args.vision_chunk)
                prepare_seconds += time.monotonic() - preparing_started
                modeling_started = time.monotonic()
                logits, _ = batched_logits(runtime, head, samples, ready_collated)
                targets = torch.tensor([ACTIONS.index(row["label"]) for row in chunk_rows],
                                       device=runtime.device)
                loss = F.cross_entropy(logits, targets, reduction="sum") * (world / total)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at update {step + 1}")
                with runtime._cached_image_features([frame for item in samples for frame in item[1]]):
                    loss.backward()
                loss_sum += float(loss.detach())
                model_seconds += time.monotonic() - modeling_started
                del samples, logits, targets, loss
                if prepared_chunks is not None:
                    prepared_chunks[chunk_index] = None
                    del cpu_collated, ready_collated, inputs, visions, lengths
            recent_losses.append(loss_sum * total / (world * len(batch_rows)))
            labels.update(row["label"] for row in batch_rows)
        syncing_started = time.monotonic()
        if args.backend == "nccl":
            packed_nccl_sync(parameters, world)
        else:
            sync_gradients(parameters, world)
        sync_seconds = time.monotonic() - syncing_started
        optimizing_started = time.monotonic()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_seconds = time.monotonic() - optimizing_started
        seen += len(batch_rows)
        step += 1
        if rank == 0 and (step <= 10 or step % args.log_every == 0 or
                          (args.stop_after_steps and step == args.stop_after_steps)):
            print(json.dumps({"step": step, "seen_per_rank": seen,
                              "global_batch_this_step": total,
                              "recent_loss": sum(recent_losses[-25:]) / min(25, len(recent_losses)),
                              "step_seconds": time.monotonic() - step_started,
                              "predecode_wait_seconds": predecode_wait_seconds,
                              "prepare_wait_seconds": prepare_wait_seconds,
                              "prepare_seconds": prepare_seconds,
                              "model_seconds": model_seconds,
                              "gradient_sync_seconds": sync_seconds,
                              "optimizer_seconds": optimizer_seconds,
                              "seconds": time.monotonic() - started,
                              "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        if step == 25 or step % args.save_every == 0 or (args.stop_after_steps and step >= args.stop_after_steps):
            save_checkpoint(args.output, runtime, head, optimizer, step=step, seen=seen,
                            rank=rank, world=world, mode="J1", seed=args.seed,
                            model_name=MODEL_NAME, model_revision=MODEL_REVISION)
        if args.stop_after_steps and step >= args.stop_after_steps:
            break
    if not (args.stop_after_steps and step >= args.stop_after_steps):
        save_checkpoint(args.output, runtime, head, optimizer, step=step, seen=seen,
                        rank=rank, world=world, mode="J1", seed=args.seed,
                        model_name=MODEL_NAME, model_revision=MODEL_REVISION)
    summary = {"step": step, "seen_per_rank": seen, "labels": dict(labels),
               "training_start_utc": started_utc,
               "seconds": time.monotonic() - started,
               "peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 1024**3}
    (args.output / f"summary.rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if prefetch_pool is not None:
        prefetch_pool.shutdown()
    if decode_pool is not None:
        decode_pool.shutdown()


if __name__ == "__main__":
    main()
