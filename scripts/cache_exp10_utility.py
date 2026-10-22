#!/usr/bin/env python3
"""Generate answers and frozen decision features for E09/E10 training candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG / "src")]

from extract_feature_shards import assigned_video_ids, load_runtime
from generate_selected_responses import forced_response
from stream_gemma.inference.qwen_stream import FrameVisionCache

from pace_clm.config import load_mvp_config
from pace_clm.experiment import (
    absolute_timeline,
    decode_indices,
    direct_inputs,
    gold_messages,
    state_timeline,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/exp10/e09_utility_candidates.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pace-clm-exp10/cache/utility")
    p.add_argument("--max-videos-per-rank", type=int, default=0)
    p.add_argument("--save-every-videos", type=int, default=10)
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    selected = read_jsonl(a.manifest)
    if len({(r["video_id"], int(r["frame_index"])) for r in selected}) != len(selected):
        raise RuntimeError("duplicate utility candidate")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        groups[row["video_id"]].append(row)
    rollout_rows = {
        row["video_id"]: row
        for row in read_jsonl(ROOT / "manifests/exp10/training_rollout.jsonl")
    }
    assigned = assigned_video_ids(groups, rank, world)
    if a.max_videos_per_rank:
        assigned = assigned[:a.max_videos_per_rank]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    shard = a.output_dir / f"utility.rank{rank:02d}.pt"
    source_sha = sha(a.manifest)
    existing = torch.load(shard, map_location="cpu", weights_only=False) if shard.exists() else None
    if existing and (existing["source_sha256"] != source_sha or existing["world_size"] != world or
                     existing["max_videos_per_rank"] != a.max_videos_per_rank):
        raise RuntimeError("incompatible utility cache shard")
    completed = set(existing["completed_video_ids"]) if existing else set()
    output = list(existing["rows"]) if existing else []
    pending = [video for video in assigned if video not in completed]
    if not pending:
        print(f"[rank {rank}] utility cache complete", flush=True)
        return
    wanted_raw = {int(groups[v][0]["raw_index"]): v for v in pending}
    annotations = {}
    with (SG / "datasets/markii/train.jsonl").open() as f:
        for raw, line in enumerate(f):
            if raw in wanted_raw:
                annotations[wanted_raw[raw]] = json.loads(line)
    if set(annotations) != set(pending):
        raise RuntimeError("missing utility training annotation")
    metadata = json.loads((SG / "datasets/markii/metadata.json").read_text())
    runtime, extractor = load_runtime(load_mvp_config(ROOT / "configs/qwen_mvp.yaml"))
    started = time.monotonic()

    def save() -> None:
        temp = shard.with_suffix(".pt.tmp")
        torch.save({"source_sha256": source_sha, "world_size": world,
                    "max_videos_per_rank": a.max_videos_per_rank,
                    "completed_video_ids": sorted(completed), "rows": output}, temp)
        os.replace(temp, shard)

    for ordinal, video in enumerate(pending, 1):
        timeline = state_timeline(annotations[video], metadata)
        if timeline["video_id"] != video:
            raise RuntimeError("utility video mismatch")
        # The inference video can continue past the annotated event range. Candidate
        # times are frozen in rollout coordinates, so retain the rollout frame count.
        count = int(rollout_rows[video]["inference_frame_count"])
        emits, states = absolute_timeline(timeline, count)
        candidates = sorted(groups[video], key=lambda r: int(r["frame_index"]))
        needed = sorted({frame for row in candidates for frame in
                         range(max(0, int(row["frame_index"]) - 31), int(row["frame_index"]) + 1)})
        video_path = metadata[video].get("path") or str(SG / "datasets/markii/videos" / f"{video}.mp4")
        decoded = runtime._resize(decode_indices(video_path, needed))
        images = [None] * count
        for frame, image in zip(needed, decoded, strict=True):
            images[frame] = image
        cache = FrameVisionCache(runtime, limit=32)
        for row in candidates:
            frame = int(row["frame_index"])
            if not 0 <= frame < count:
                raise RuntimeError("candidate outside video")
            messages, start = gold_messages(question=timeline["question"], images=images,
                                            emits=emits, states=states, index=frame)
            vision = cache.window(images, start, frame + 1)
            runtime.encode_frames_features(vision)
            with runtime._cached_image_features(vision), torch.inference_mode():
                feature = extractor.extract(direct_inputs(runtime, messages, vision)).squeeze(0).half().cpu()
            answer = forced_response(runtime, messages, vision, 96)
            if not answer.strip():
                raise RuntimeError(f"empty candidate answer: {video}:{frame}")
            output.append({"video_id": video, "frame_index": frame,
                           "category": row["category"], "feature": feature, "answer": answer})
        completed.add(video)
        if len(completed) % a.save_every_videos == 0:
            save()
        print(f"[rank {rank}] {ordinal}/{len(pending)} {video} "
              f"candidates={len(candidates)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    save()


if __name__ == "__main__":
    main()
