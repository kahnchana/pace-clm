#!/usr/bin/env python3
"""Isolated SFT campaign trainer; explicit new-data continuations retain optimizer/RNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

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
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--row-order", type=Path,
                        help="Immutable N x 2 numpy byte-offset/length plan into manifest")
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
    parser.add_argument("--initialize-from", type=Path, help="Shared fresh step-zero checkpoint")
    parser.add_argument("--fork-from", type=Path, help="Full-state branch with verified unchanged consumed data prefix")
    parser.add_argument("--fork-parent-recipe", type=Path)
    parser.add_argument("--continuation-seed", type=int,
                        help="Explicitly reseed a fresh full-state fork; resumes restore saved RNG without reseeding")
    parser.add_argument("--fixed-l2norm-launch", action="store_true",
                        help="Use validated fixed launch geometry for the existing FLA normalization kernels")
    parser.add_argument("--initialize-only", action="store_true", help="Save fresh state without consuming rows")
    parser.add_argument("--require-global-batch", type=int, default=0)
    parser.add_argument("--verify-restore-only", action="store_true",
                        help="Verify exact checkpoint restoration without taking an update")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-6)
    parser.add_argument("--continue-from", type=Path,
                        help="Retain full learned/optimizer/RNG state, reset only new-data counters")
    parser.add_argument("--feature-cache", type=Path)
    args = parser.parse_args()
    if args.head_lr <= 0 or args.lora_lr <= 0:
        parser.error("learning rates must be positive")
    if args.continue_from and (args.resume or args.fork_from or args.initialize_from):
        parser.error("continue-from is an explicit new-data branch, incompatible with other initializers")
    if args.resume and args.initialize_from:
        parser.error("resume and initialize-from are mutually exclusive")
    if args.fork_from and (args.resume or args.initialize_from or not args.fork_parent_recipe or not args.row_order):
        parser.error("fork-from requires parent recipe and row-order, and cannot resume or initialize-from")
    if args.continuation_seed is not None and (args.continuation_seed < 0 or not (args.fork_from or args.resume)):
        parser.error("continuation-seed requires a nonnegative seed and a fork or resume")
    if not args.resume and (args.output / "latest.json").exists():
        raise FileExistsError("output already contains training state; use --resume")
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
    fixed_l2norm = None
    if args.fixed_l2norm_launch:
        from qwen35_diagnostic_kernels import pin_safe_l2norm
        fixed_l2norm = pin_safe_l2norm()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.output / "latest.json").read_text()) if args.resume else None
    previous_dir = Path(previous["checkpoint"]) if previous else (args.continue_from or args.fork_from or args.initialize_from)
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
        {"params": list(head.parameters()), "lr": args.head_lr},
        {"params": backbone, "lr": args.lora_lr},
    ])
    seen = 0
    step = 0
    cumulative_training_seconds = 0.0
    cumulative_labels = Counter()
    if previous_dir:
        state = torch.load(previous_dir / "training_state.pt", map_location="cpu", weights_only=False)
        if (state["world"], state["mode"], state["seed"], state["model_revision"]) != (
            world, "J1", args.seed, MODEL_REVISION
        ):
            raise ValueError("checkpoint identity mismatch")
        head.load_state_dict(torch.load(previous_dir / "pointer_head.pt", map_location="cpu", weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        if args.resume and state.get("continuation_seed") != args.continuation_seed:
            raise ValueError("resume continuation-seed mismatch")
        if args.fork_from and state.get("continuation_seed") is not None and args.continuation_seed is None:
            raise ValueError("fork must retain or explicitly change the parent's continuation seed")
        rng = torch.load(previous_dir / f"rng.rank{rank}.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        if "python" in rng:
            random.setstate(rng["python"])
        seen = int(state["seen_per_rank"])
        step = int(state["step"])
        cumulative_training_seconds = state.get("training_seconds", 0.0)
        cumulative_labels.update(state.get("global_labels", {}))
        if args.initialize_from and (step or seen) and not args.verify_restore_only:
            raise ValueError("initialize-from requires a fresh step-zero state")
    if args.verify_restore_only:
        if previous_dir is None:
            raise ValueError("restore verification requires a checkpoint")
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file

        def exact(left, right):
            if isinstance(left, torch.Tensor):
                assert left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    exact(left[key],right[key])
            elif isinstance(left, (list,tuple)):
                assert len(left) == len(right)
                for lval,rval in zip(left,right):
                    exact(lval,rval)
            else:
                assert left == right

        exact(get_peft_model_state_dict(runtime.model), load_file(previous_dir/'controller_adapter/adapter_model.safetensors'))
        exact(head.state_dict(),torch.load(previous_dir/'pointer_head.pt',map_location='cpu',weights_only=True))
        exact(optimizer.state_dict(),state['optimizer'])
        exact(torch.get_rng_state(),rng['torch'])
        exact(torch.cuda.get_rng_state(),rng['cuda'])
        exact(random.getstate(),rng['python'])
        assert step == state['step'] and seen == state['seen_per_rank']
        assert cumulative_training_seconds == state['training_seconds']
        assert dict(cumulative_labels) == state['global_labels']
        (args.output/f'verified.rank{rank}.json').write_text(json.dumps({'exact_restore':True,'step':step,'seen_per_rank':seen})+'\n')
        dist.barrier()
        dist.destroy_process_group()
        return
    if args.continuation_seed is not None and not args.resume:
        # Keep the parent's model/optimizer identity and consumed prefix intact.
        # This explicit continuation intervention is applied once, never on resume.
        def rng_hashes():
            return {"torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    "cuda": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}
        before = rng_hashes()
        continuation_rank_seed = args.continuation_seed + rank
        reseeded = state.get("continuation_seed") != args.continuation_seed
        if reseeded:
            torch.manual_seed(continuation_rank_seed)
            torch.cuda.manual_seed(continuation_rank_seed)
            random.seed(continuation_rank_seed)
        (args.output / f"continuation_rng.rank{rank}.json").write_text(json.dumps({
            "continuation_seed": args.continuation_seed, "rank_seed": continuation_rank_seed,
            "rank": rank, "parent_step": step, "parent_rng_sha256": before,
            "continuation_rng_sha256": rng_hashes(), "optimizer_restored_before_reseed": True,
            "reseeded": reseeded,
        }, sort_keys=True) + "\n")
    campaign_parent = None
    if args.continue_from:
        campaign_parent = {
            "checkpoint": str(args.continue_from.resolve()),
            "step": step, "global_presentations": seen * world,
            "training_state_sha256": hashlib.sha256((args.continue_from / "training_state.pt").read_bytes()).hexdigest(),
            "optimizer_moments_and_rank_rng_retained": True,
            "new_data_cursor_reset_explicitly": True,
            "parent_learning_rates": [g["lr"] for g in optimizer.param_groups],
        }
        step = seen = 0
        cumulative_training_seconds = 0.0
        cumulative_labels.clear()
        for group, lr in zip(optimizer.param_groups, (args.head_lr, args.lora_lr), strict=True):
            group["lr"] = lr
    if args.resume:
        if [g['lr'] for g in optimizer.param_groups] != [args.head_lr, args.lora_lr]:
            raise ValueError("resume learning rates differ from checkpoint")
    if rank == 0:
        manifest_hash = hashlib.sha256()
        with args.manifest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                manifest_hash.update(chunk)
        if args.expected_manifest_sha256 and manifest_hash.hexdigest() != args.expected_manifest_sha256:
            raise ValueError("manifest differs from the frozen launch plan")
        recipe_path = args.output / "recipe.json"
        plan_hash = hashlib.sha256(args.batch_plan.read_bytes()).hexdigest() if args.batch_plan else None
        order_hash = hashlib.sha256(args.row_order.read_bytes()).hexdigest() if args.row_order else None
        if args.resume:
            old_recipe = json.loads(recipe_path.read_text())
            if (old_recipe["manifest_sha256"], old_recipe.get("batch_plan_sha256")) != (
                manifest_hash.hexdigest(), plan_hash
            ):
                raise ValueError("resume manifest or batch-plan checksum mismatch")
            if old_recipe.get("row_order_sha256") != order_hash:
                raise ValueError("resume row-order checksum mismatch")
        recipe = {
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash.hexdigest(),
            "batch_plan": str(args.batch_plan) if args.batch_plan else None,
            "batch_plan_sha256": plan_hash,
            "row_order": str(args.row_order) if args.row_order else None,
            "row_order_sha256": order_hash,
            "model": MODEL_NAME, "revision": MODEL_REVISION,
            "physical_batch_per_gpu": "planned" if args.batch_plan else args.microbatch,
            "microbatch": args.microbatch,
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
            "continuation_seed": args.continuation_seed,
            "initialize_from": str(args.initialize_from) if args.initialize_from else None,
            "fork_from": str(args.fork_from) if args.fork_from else None,
            "fixed_l2norm_launch": fixed_l2norm,
            "optimizer": [{k:v for k,v in group.items() if k != "params"} for group in optimizer.param_groups],
            "gradient_clip": 1.0, "require_global_batch": args.require_global_batch,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "campaign_parent": campaign_parent,
            "feature_cache": str(args.feature_cache) if args.feature_cache else None,
        }
        if args.fork_from:
            from qwen35_context_order import validate_fork
            recipe["fork_provenance"] = validate_fork(args.fork_parent_recipe,args.row_order,
                manifest_hash.hexdigest(),step,seen,world,recipe)
            recipe["fork_parent_fixed_l2norm_launch"] = json.loads(args.fork_parent_recipe.read_text()).get("fixed_l2norm_launch")
        if args.resume:
            for key in ("seed", "backend", "microbatch_ratio", "microbatch", "require_global_batch", "fixed_l2norm_launch", "continuation_seed", "feature_cache"):
                if old_recipe.get(key) != recipe.get(key):
                    raise ValueError(f"resume recipe mismatch: {key}")
        else:
            recipe["microbatch"] = args.microbatch
            recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    dist.barrier()
    if args.initialize_only:
        save_checkpoint(args.output, runtime, head, optimizer, step=0, seen=0,
                        rank=rank, world=world, mode="J1", seed=args.seed,
                        model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                        extra_state={"training_seconds": 0.0, "global_labels": {}})
        dist.destroy_process_group()
        return
    if args.stop_after_steps and step >= args.stop_after_steps:
        raise ValueError("checkpoint already reached requested stop step")
    frames = EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache,
                           defer_vision=args.vision_chunk > 0)
    cpu_frames = (EpisodeFrames(runtime, response_blind=False, cache_frames=args.frame_cache)
                  if prefetch_mode == "prepare" else None)
    from qwen35_sft_campaign import install_training_contexts, install_feature_cache
    install_training_contexts()
    if args.feature_cache:
        install_feature_cache(runtime, (frames, cpu_frames), args.feature_cache)
    decode_pool = ThreadPoolExecutor(max_workers=args.predecode_workers) if args.predecode_workers else None
    prefetch_pool = ThreadPoolExecutor(max_workers=1) if prefetch_mode != "none" else None
    if args.row_order:
        def indexed_rows():
            offsets = np.load(args.row_order, mmap_mode="r")
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                raise ValueError("invalid byte-offset plan")
            with args.manifest.open("rb") as source:
                for index in range(seen * world + rank, len(offsets), world):
                    offset, size = map(int, offsets[index])
                    source.seek(offset)
                    line = source.read(size)
                    if len(line) != size or not line.endswith(b"\n"):
                        raise ValueError("invalid byte range")
                    yield json.loads(line)
        rows = indexed_rows()
    else:
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

    iteration_started = time.monotonic()
    pending = schedule(next_batch()) if prefetch_pool else None
    saved_step = None
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
        if args.require_global_batch and total != args.require_global_batch:
            raise ValueError(f"required global batch {args.require_global_batch}, got {total}")
        if args.batch_plan and total != planned_per_rank * world:
            raise ValueError(f"batch plan expected {planned_per_rank * world} rows, got {total}")
        prepare_seconds = 0.0
        model_seconds = 0.0
        token_count = 0
        padded_token_count = 0
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
                logits, sequence_lengths = batched_logits(runtime, head, samples, ready_collated)
                token_count += sum(sequence_lengths)
                padded_token_count += max(sequence_lengths) * len(sequence_lengths)
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
        torch.cuda.synchronize(runtime.device)
        optimizer_seconds = time.monotonic() - optimizing_started
        elapsed = torch.tensor(time.monotonic() - iteration_started,
                               device=runtime.device if args.backend == "nccl" else "cpu")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        cumulative_training_seconds += elapsed.item()
        step_counts = torch.tensor([sum(row['label'] == action for row in batch_rows) for action in ACTIONS],
                                   device=runtime.device if args.backend == "nccl" else "cpu")
        dist.all_reduce(step_counts)
        cumulative_labels.update(dict(zip(ACTIONS, step_counts.cpu().tolist())))
        seen += len(batch_rows)
        step += 1
        if rank == 0 and (step <= 10 or step % args.log_every == 0 or
                          (args.stop_after_steps and step == args.stop_after_steps)):
            print(json.dumps({"step": step, "seen_per_rank": seen,
                              "global_batch_this_step": total,
                              "training_seconds": cumulative_training_seconds,
                              "global_labels": dict(cumulative_labels),
                              "rank0_token_padding_fraction": 1 - token_count / max(1, padded_token_count),
                              "rank0_context_frames": [len(row['turns']) for row in batch_rows],
                              "rank0_memory_rows": sum(bool(row.get('memory')) for row in batch_rows),
                              "rank0_frame_cache_hit_fraction": (cpu_frames or frames).frame_cache_hits / max(1, (cpu_frames or frames).frame_requests),
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
                            model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                            extra_state={"training_seconds": cumulative_training_seconds,
                                         "global_labels": dict(cumulative_labels),
                                         "continuation_seed": args.continuation_seed})
            saved_step = step
        if args.stop_after_steps and step >= args.stop_after_steps:
            break
        iteration_started = time.monotonic()
    if saved_step != step:
        save_checkpoint(args.output, runtime, head, optimizer, step=step, seen=seen,
                        rank=rank, world=world, mode="J1", seed=args.seed,
                        model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                        extra_state={"training_seconds": cumulative_training_seconds,
                                     "global_labels": dict(cumulative_labels),
                                         "continuation_seed": args.continuation_seed})
    summary = {"step": step, "seen_per_rank": seen, "labels": dict(labels),
               "training_seconds": cumulative_training_seconds, "global_labels": dict(cumulative_labels),
               "peak_reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 1024**3,
               "device_memory_gib": torch.cuda.get_device_properties(runtime.device).total_memory / 1024**3,
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
