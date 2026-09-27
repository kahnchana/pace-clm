#!/usr/bin/env python3
"""Distributed frozen PACE-Qwen feature extraction for train or PACE-200."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from stream_gemma.inference.qwen_stream import FrameVisionCache, QwenRuntime

from pace_clm.backbone import FrozenFeatureExtractor, assert_frozen
from pace_clm.config import load_mvp_config
from pace_clm.experiment import (
    absolute_timeline,
    annotation_records,
    decode_indices,
    direct_inputs,
    extract_one,
    gold_messages,
    read_jsonl,
    state_timeline,
)
from pace_clm.labels import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "pace200"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos-per-rank", type=int, default=0)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_mvp.yaml")
    return parser.parse_args()


def load_runtime(config: dict) -> tuple[QwenRuntime, FrozenFeatureExtractor]:
    runtime = QwenRuntime.load(
        checkpoint=str((ROOT / config["model"]["adapter"]).resolve()),
        model_name=config["model"]["name"],
        model_revision=config["model"]["revision"],
        frame_num_tokens=int(config["streaming"]["frame_num_tokens"]),
        generation_prefix_mode=config["streaming"]["qwen_prefix_mode"],
    )
    extractor = FrozenFeatureExtractor(
        runtime.model, vision_token_ids=[runtime.processor.image_token_id]
    )
    assert_frozen(runtime.model)
    return runtime, extractor


def extract_actions(runtime: QwenRuntime, extractor: FrozenFeatureExtractor) -> torch.Tensor:
    payload = yaml.safe_load((ROOT / "configs/actions_v1.yaml").read_text())
    features = []
    for action in Action:
        description = payload["actions"][action.name]
        messages = [{"role": "user", "content": [{"type": "text", "text": description}]}]
        features.append(extractor.extract(direct_inputs(runtime, messages)).float().cpu())
    return torch.cat(features, dim=0)


def assigned_video_ids(groups: dict[str, list], rank: int, world: int) -> list[str]:
    weighted = sorted(groups, key=lambda key: (-len(groups[key]), key))
    bins: list[list[str]] = [[] for _ in range(world)]
    loads = [0] * world
    for video_id in weighted:
        target = min(range(world), key=lambda item: (loads[item], item))
        bins[target].append(video_id)
        loads[target] += len(groups[video_id])
    return bins[rank]


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    config = load_mvp_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((STREAM_GEMMA / "datasets/markii/metadata.json").read_text())
    if args.split == "train":
        selected = read_jsonl(ROOT / "manifests/pace_train_balanced_v1.jsonl")
        records = annotation_records(STREAM_GEMMA / "datasets/markii/train.jsonl")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            groups[row["video_id"]].append(row)
        val_rows = None
    else:
        val_path = (
            STREAM_GEMMA
            / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl"
        )
        val_rows = read_jsonl(val_path)
        records = annotation_records(STREAM_GEMMA / "datasets/markii/val.jsonl")
        groups = {
            row["video_id"]: list(range(int(row["inference_frame_count"]))) for row in val_rows
        }

    video_ids = assigned_video_ids(groups, rank, world)
    if args.max_videos_per_rank:
        video_ids = video_ids[: args.max_videos_per_rank]
    val_by_id = {row["video_id"]: row for row in (val_rows or [])}
    runtime, extractor = load_runtime(config)
    action_features = extract_actions(runtime, extractor) if rank == 0 else None
    output_rows = []
    started = time.monotonic()

    for video_number, video_id in enumerate(video_ids, 1):
        if args.split == "train":
            selections = groups[video_id]
            raw_index = int(selections[0]["raw_index"])
            timeline = state_timeline(records[raw_index], metadata)
            frame_count = int(timeline["source_stop"])
            emits, states = absolute_timeline(timeline, frame_count)
            source_indices = list(range(frame_count))
            path = metadata[video_id].get("path") or str(
                STREAM_GEMMA / "datasets/markii/videos" / f"{video_id}.mp4"
            )
            targets = [(int(row["frame_index"]), row) for row in selections]
            question = timeline["question"]
        else:
            source = val_by_id[video_id]
            raw_index = int(source["raw_index"])
            timeline = state_timeline(records[raw_index], metadata)
            frame_count = int(source["inference_frame_count"])
            emits, states = absolute_timeline(timeline, frame_count)
            source_indices = [int(value) for value in source["inference_source_indices"]]
            path = source["resolved_video_path"]
            targets = [(index, None) for index in range(frame_count)]
            question = source["question"]

        images = runtime._resize(decode_indices(path, source_indices))
        cache = FrameVisionCache(runtime, limit=max(32, len(images)))
        cache.prefetch(images)
        for index, selection in targets:
            messages, window_start = gold_messages(
                question=question,
                images=images,
                emits=emits,
                states=states,
                index=index,
            )
            vision = cache.window(images, window_start, index + 1)
            feature = extract_one(runtime, extractor, messages, vision).squeeze(0)
            action = (
                Action[selection["gold_action"]]
                if selection
                else Action["Hold" if states[index] == "Standby" else states[index]]
            )
            output_rows.append(
                {
                    "feature": feature,
                    "gold_action_id": int(action),
                    "gold_action": action.name,
                    "video_id": video_id,
                    "frame_index": index,
                    "timestamp": float(index),
                    "raw_index": raw_index,
                    "window_start": window_start,
                }
            )
        elapsed = time.monotonic() - started
        print(
            f"[rank {rank}] {video_number}/{len(video_ids)} videos "
            f"{len(output_rows)} decisions {elapsed:.1f}s",
            flush=True,
        )

    shard = args.output_dir / f"{args.split}.rank{rank:02d}.pt"
    torch.save(
        {
            "split": args.split,
            "rank": rank,
            "world_size": world,
            "hidden_size": extractor.hidden_size,
            "rows": output_rows,
            "action_features": action_features,
            "seconds": time.monotonic() - started,
            "config": config,
        },
        shard,
    )
    print(f"[rank {rank}] wrote {shard} ({len(output_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
