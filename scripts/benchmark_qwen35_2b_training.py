#!/usr/bin/env python3
"""Short, sequential eight-rank throughput benchmark for the 2B controller."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from train_jevany_streaming import EpisodeFrames, row_stream, sync_gradients
from pace_clm.experiment import decode_indices
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


def batched_logits(runtime, head, samples: list[tuple], collated=None):
    inputs, visions, lengths = collated if collated is not None else collate(samples, runtime)
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


def encode_batch_vision(runtime, samples: list[tuple], chunk: int):
    seen = set()
    missing = []
    for _, vision, _, _ in samples:
        for frame in vision:
            if id(frame) not in seen and frame.image_embeds is None:
                seen.add(id(frame))
                missing.append(frame)
    runtime.encode_frames_features(missing, chunk=chunk)


def predecode_batch(runtime, rows: list[dict], pool: ThreadPoolExecutor,
                    cached_images: dict | None = None):
    by_video = defaultdict(set)
    prefetched = dict(cached_images or {})
    for row in rows:
        path = row["video_path"]
        by_video[path].update(int(turn["frame_index"]) for turn in row["turns"]
                              if (path, int(turn["frame_index"])) not in prefetched)

    def decode_video(item):
        path, indices = item
        ordered = sorted(indices)
        images = runtime._resize(decode_indices(path, ordered))
        return path, ordered, images

    for path, indices, images in pool.map(decode_video,
                                          (item for item in by_video.items() if item[1])):
        prefetched.update({(path, index): image for index, image in zip(indices, images, strict=True)})
    return prefetched


def chunk_conv1d_batches(model: torch.nn.Module, max_batch: int) -> int:
    """Avoid cuDNN's 32-bit indexing limit for large physical batches."""
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    patched = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Conv1d):
            continue
        original = module.forward

        def forward(inputs, original=original):
            if inputs.shape[0] <= max_batch:
                return original(inputs)
            return torch.cat([original(part) for part in inputs.split(max_batch, dim=0)], dim=0)

        module.forward = forward
        patched += 1
    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int,
                        choices=(1, 2, 4, 8, 16, 32, 48, 52, 56, 64, 96, 128, 160, 192, 272, 288, 320), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--throughput-only", action="store_true",
                        help="Avoid per-phase CUDA synchronizations for a less intrusive throughput run")
    parser.add_argument("--frame-cache", type=int, default=0,
                        help="Maximum decoded and encoded frames cached across episodes per rank")
    parser.add_argument("--vision-chunk", type=int, default=0,
                        help="Defer vision encoding and batch unique frames in chunks of this size")
    parser.add_argument("--predecode-workers", type=int, default=0,
                        help="Decode each batch's unique video frames with this many CPU threads")
    parser.add_argument("--prefetch-decode", action="store_true",
                        help="Decode the next batch concurrently with current GPU work")
    parser.add_argument("--prefetch-prepare", action="store_true",
                        help="Decode and prepare the next batch on CPU during current GPU work")
    parser.add_argument("--conv-batch-chunk", type=int, default=0,
                        help="Split Conv1d over the batch axis to avoid 32-bit indexing limits")
    args = parser.parse_args()
    if args.warmup < 0 or args.updates < 1:
        raise ValueError("warmup must be nonnegative and updates positive")
    if args.predecode_workers < 0 or args.vision_chunk < 0 or args.conv_batch_chunk < 0:
        raise ValueError("predecode-workers, vision-chunk and conv-batch-chunk must be nonnegative")
    if (args.prefetch_decode or args.prefetch_prepare) and not args.predecode_workers:
        raise ValueError("prefetch requires predecode-workers")
    if args.prefetch_decode and args.prefetch_prepare:
        raise ValueError("choose one prefetch mode")
    accumulation = max(1, 4 // args.microbatch)
    per_rank_batch = args.microbatch * accumulation
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(42)
    dist.init_process_group(args.backend) if world > 1 else None
    runtime, _ = load_2b_controller(fresh_lora=True)
    patched_convs = chunk_conv1d_batches(runtime.model, args.conv_batch_chunk) if args.conv_batch_chunk else 0
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
    frames = EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache,
                           defer_vision=args.vision_chunk > 0)
    decode_pool = ThreadPoolExecutor(max_workers=args.predecode_workers) if args.predecode_workers else None
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if (args.prefetch_decode or args.prefetch_prepare) else None
    cpu_frames = (EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache)
                  if args.prefetch_prepare else None)
    records = iter(row_stream(args.manifest, rank, world))
    total_batches = (args.warmup + args.updates) * accumulation
    batches_loaded = 0
    pending = None

    def prepare_on_cpu(batch_rows):
        cached_images = ({key: image for key, (image, _) in cpu_frames.frames.items()}
                         if args.frame_cache else None)
        decoded = predecode_batch(runtime, batch_rows, decode_pool, cached_images)
        samples = [cpu_frames.prepare(row, prefetched=decoded, cpu_only=True)
                   for row in batch_rows]
        return samples, collate(samples, runtime)

    def load_pending():
        nonlocal batches_loaded
        if batches_loaded >= total_batches:
            return None
        batch_rows = [next(records) for _ in range(args.microbatch)]
        batches_loaded += 1
        if args.prefetch_prepare:
            return batch_rows, prefetch_pool.submit(prepare_on_cpu, batch_rows)
        cache_snapshot = {(path, index): image for (path, index), (image, _) in frames.frames.items()} \
                         if args.frame_cache else None
        return batch_rows, prefetch_pool.submit(predecode_batch, runtime, batch_rows,
                                                decode_pool, cache_snapshot)

    if prefetch_pool is not None:
        pending = load_pending()
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
            ready_collated = None
            if pending is not None:
                batch_rows, prefetched_future = pending
                started = time.perf_counter()
                prepared = prefetched_future.result()
                mark("prepare_wait" if args.prefetch_prepare else "predecode_wait", started, enabled)
                pending = load_pending()
                prefetched = None if args.prefetch_prepare else prepared
            else:
                batch_rows = [next(records) for _ in range(args.microbatch)]
                prefetched = None
            if decode_pool is not None and prefetch_pool is None:
                started = time.perf_counter()
                cache_snapshot = {(path, index): image for (path, index), (image, _) in frames.frames.items()} \
                                 if args.frame_cache else None
                prefetched = predecode_batch(runtime, batch_rows, decode_pool, cache_snapshot)
                mark("predecode", started, enabled)
            if args.prefetch_prepare:
                started = time.perf_counter()
                samples, cpu_collated = prepared
                encode_batch_vision(runtime, samples, 8)
                samples = [({key: value.to(runtime.device) if torch.is_tensor(value) else value
                             for key, value in inputs.items()}, vision, options, decision)
                           for inputs, vision, options, decision in samples]
                inputs, visions, lengths = cpu_collated
                ready_collated = ({key: value.to(runtime.device) for key, value in inputs.items()},
                                  visions, lengths)
                mark("vision_and_transfer", started, enabled)
                labels = [ACTIONS.index(row["label"]) for row in batch_rows]
            else:
                for row in batch_rows:
                    labels.append(ACTIONS.index(row["label"]))
                    samples.append(frames.prepare(row, totals if enabled and not args.throughput_only else None,
                                                  prefetched=prefetched))
            if args.vision_chunk:
                started = time.perf_counter()
                encode_batch_vision(runtime, samples, args.vision_chunk)
                mark("vision_encode_batched", started, enabled)
            started = time.perf_counter()
            if args.microbatch == 1:
                item = samples[0]
                with runtime._cached_image_features(item[1]):
                    logits = controller_logits(runtime, head, *item, train_decoder=True).unsqueeze(0)
                lengths = [item[0]["input_ids"].shape[1]]
            else:
                logits, lengths = batched_logits(runtime, head, samples, ready_collated)
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
        "frame_cache": args.frame_cache,
        "vision_chunk": args.vision_chunk,
        "predecode_workers": args.predecode_workers,
        "prefetch_decode": args.prefetch_decode,
        "prefetch_prepare": args.prefetch_prepare,
        "conv_batch_chunk": args.conv_batch_chunk,
        "patched_convs": patched_convs,
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
    if prefetch_pool is not None:
        prefetch_pool.shutdown()
    if decode_pool is not None:
        decode_pool.shutdown()


if __name__ == "__main__":
    main()
