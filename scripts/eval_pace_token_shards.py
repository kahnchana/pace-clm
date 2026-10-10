#!/usr/bin/env python3
"""Evaluate the existing PACE tag-generation decision on frozen PACE-200."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from stream_gemma.inference.qwen_stream import FrameVisionCache, QwenRuntime
from stream_gemma.inference.v24 import parse_v24_emit

from pace_clm.config import load_mvp_config
from pace_clm.experiment import (
    absolute_timeline,
    annotation_records,
    decode_indices,
    gold_messages,
    read_jsonl,
    state_timeline,
)
from pace_clm.labels import Action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--max-videos-per-rank", type=int, default=0)
    parser.add_argument("--save-every-videos", type=int, default=5)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    config = load_mvp_config(ROOT / "configs/qwen_mvp.yaml")
    runtime = QwenRuntime.load(
        checkpoint=str((ROOT / config["model"]["adapter"]).resolve()),
        model_name=config["model"]["name"],
        model_revision=config["model"]["revision"],
        frame_num_tokens=int(config["streaming"]["frame_num_tokens"]),
        generation_prefix_mode="direct",
    )
    val_path = (
        STREAM_GEMMA
        / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl"
    )
    rows = read_jsonl(args.validation_manifest or val_path)
    # Identical deterministic length-balanced assignment as feature extraction.
    bins: list[list[dict]] = [[] for _ in range(world)]
    loads = [0] * world
    for row in sorted(
        rows, key=lambda item: (-int(item["inference_frame_count"]), item["video_id"])
    ):
        target = min(range(world), key=lambda item: (loads[item], item))
        bins[target].append(row)
        loads[target] += int(row["inference_frame_count"])
    assigned = sorted(bins[rank], key=lambda row: row["video_id"])
    if args.max_videos_per_rank:
        assigned = assigned[: args.max_videos_per_rank]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"pace_token.rank{rank:02d}.jsonl"
    output = read_jsonl(path) if path.exists() else []
    completed = {row["video_id"] for row in output}
    if completed - {row["video_id"] for row in assigned}:
        raise RuntimeError(f"existing token shard has videos outside current assignment: {path}")

    def save() -> None:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
        os.replace(temporary, path)

    records = annotation_records(STREAM_GEMMA / "datasets/markii/val.jsonl")
    metadata = json.loads((STREAM_GEMMA / "datasets/markii/metadata.json").read_text())
    started = time.monotonic()
    for number, source in enumerate(assigned, 1):
        if source["video_id"] in completed:
            continue
        timeline = state_timeline(records[int(source["raw_index"])], metadata)
        count = int(source["inference_frame_count"])
        emits, states = absolute_timeline(timeline, count)
        images = runtime._resize(
            decode_indices(
                source["resolved_video_path"],
                [int(value) for value in source["inference_source_indices"]],
            )
        )
        cache = FrameVisionCache(runtime, limit=max(32, count))
        cache.prefetch(images)
        for index in range(count):
            messages, start = gold_messages(
                question=source["question"],
                images=images,
                emits=emits,
                states=states,
                index=index,
            )
            text = runtime.generate_turn(
                messages,
                max_new_tokens=4,
                vision=cache.window(images, start, index + 1),
            )
            parsed = parse_v24_emit(text)
            predicted = (
                Action.Hold
                if parsed and parsed.state == "Standby"
                else Action[parsed.state]
                if parsed
                else Action.Silence
            )
            gold = Action.Hold if states[index] == "Standby" else Action[states[index]]
            output.append(
                {
                    "video_id": source["video_id"],
                    "frame_index": index,
                    "timestamp": float(index),
                    "gold_action": gold.name,
                    "gold_action_id": int(gold),
                    "predicted_action": predicted.name,
                    "predicted_action_id": int(predicted),
                    "raw_emit": text,
                    "malformed": parsed is None,
                }
            )
        completed.add(source["video_id"])
        if len(completed) % args.save_every_videos == 0:
            save()
        print(
            f"[rank {rank}] {number}/{len(assigned)} videos {len(output)} decisions "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
    save()
    print(f"[rank {rank}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
