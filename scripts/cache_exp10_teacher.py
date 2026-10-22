#!/usr/bin/env python3
"""Cache frozen PACE action-tag sequence scores on run5 training positions."""

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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG / "src")]

from extract_feature_shards import assigned_video_ids, load_runtime
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import FrameVisionCache

from pace_clm.config import load_mvp_config
from pace_clm.experiment import (
    absolute_timeline,
    decode_indices,
    direct_inputs,
    gold_messages,
    state_timeline,
)

TAGS = (STATE_TAGS["Response"], STATE_TAGS["Silence"], STATE_TAGS["Standby"])


def tag_log_scores(runtime, extractor, messages: list[dict], vision) -> torch.Tensor:
    """Log probability of each full tag string under the frozen token generator."""
    prompt = direct_inputs(runtime, messages, vision)
    output_head = runtime.model.get_output_embeddings()
    scores = []
    for tag in TAGS:
        targets = runtime.tokenizer(tag, add_special_tokens=False, return_tensors="pt")["input_ids"].to(runtime.device)
        if targets.ndim != 2 or targets.shape[0] != 1 or targets.shape[1] < 1:
            raise RuntimeError(f"invalid action tag tokens: {tag}")
        context = targets[:, :-1]
        inputs = dict(prompt)
        inputs["input_ids"] = torch.cat([prompt["input_ids"], context], dim=1)
        inputs["attention_mask"] = torch.cat([prompt["attention_mask"], torch.ones_like(context)], dim=1)
        if "mm_token_type_ids" in prompt:
            inputs["mm_token_type_ids"] = torch.cat(
                [prompt["mm_token_type_ids"], torch.zeros_like(context)], dim=1)
        with torch.inference_mode(), runtime._cached_image_features(vision):
            hidden = extractor.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            last = hidden[:, -targets.shape[1]:, :]
            logits = output_head(last).float()
            logp = F.log_softmax(logits, dim=-1)
            score = logp.gather(-1, targets.unsqueeze(-1)).sum()
        scores.append(score.detach().cpu())
    return torch.stack(scores)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pace-clm-exp10/cache/teacher")
    p.add_argument("--max-videos-per-rank", type=int, default=0, help="Training-only preflight")
    p.add_argument("--save-every-videos", type=int, default=10)
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    if len(rows) != len({(r["video_id"], int(r["frame_index"])) for r in rows}):
        raise RuntimeError("duplicate training position")
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    assigned = assigned_video_ids(by_video, rank, world)
    if a.max_videos_per_rank:
        assigned = assigned[:a.max_videos_per_rank]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    shard = a.output_dir / f"teacher.rank{rank:02d}.pt"
    existing = torch.load(shard, map_location="cpu", weights_only=False) if shard.exists() else None
    source_sha = sha(a.manifest)
    if existing and (existing["source_sha256"] != source_sha or existing["world_size"] != world or
                     existing["max_videos_per_rank"] != a.max_videos_per_rank):
        raise RuntimeError("incompatible teacher cache shard")
    completed = set(existing["completed_video_ids"]) if existing else set()
    output = list(existing["rows"]) if existing else []
    pending = [video for video in assigned if video not in completed]
    if not pending:
        print(f"[rank {rank}] cache complete", flush=True)
        return
    wanted_raw = {int(by_video[v][0]["raw_index"]): v for v in pending}
    records = {}
    with (SG / "datasets/markii/train.jsonl").open() as f:
        for raw_index, line in enumerate(f):
            if raw_index in wanted_raw:
                records[wanted_raw[raw_index]] = json.loads(line)
    if set(records) != set(pending):
        raise RuntimeError("missing training annotation")
    metadata = json.loads((SG / "datasets/markii/metadata.json").read_text())
    config = load_mvp_config(ROOT / "configs/qwen_mvp.yaml")
    runtime, extractor = load_runtime(config)
    started = time.monotonic()

    def save() -> None:
        temp = shard.with_suffix(".pt.tmp")
        torch.save({"source_sha256": source_sha, "world_size": world,
                    "max_videos_per_rank": a.max_videos_per_rank,
                    "completed_video_ids": sorted(completed), "rows": output,
                    "tag_strings": TAGS,
                    "model_revision": config["model"]["revision"]}, temp)
        os.replace(temp, shard)

    for ordinal, video_id in enumerate(pending, 1):
        timeline = state_timeline(records[video_id], metadata)
        if timeline["video_id"] != video_id:
            raise RuntimeError("annotation video mismatch")
        selections = sorted(by_video[video_id], key=lambda r: int(r["frame_index"]))
        count = int(timeline["source_stop"])
        emits, states = absolute_timeline(timeline, count)
        source_indices = sorted({f for row in selections
                                 for f in range(max(0, int(row["frame_index"]) - 31), int(row["frame_index"]) + 1)})
        video_path = metadata[video_id].get("path") or str(SG / "datasets/markii/videos" / f"{video_id}.mp4")
        decoded = runtime._resize(decode_indices(video_path, source_indices))
        images = [None] * count
        for index, image in zip(source_indices, decoded, strict=True):
            images[index] = image
        cache = FrameVisionCache(runtime, limit=32)
        for row in selections:
            index = int(row["frame_index"])
            messages, start = gold_messages(question=timeline["question"], images=images,
                                            emits=emits, states=states, index=index)
            vision = cache.window(images, start, index + 1)
            runtime.encode_frames_features(vision)
            scores = tag_log_scores(runtime, extractor, messages, vision)
            if scores.shape != (3,) or not torch.isfinite(scores).all():
                raise RuntimeError("invalid teacher action scores")
            output.append({"video_id": video_id, "frame_index": index,
                           "gold_action_id": int(row["gold_action_id"]),
                           "log_scores": scores})
        completed.add(video_id)
        if len(completed) % a.save_every_videos == 0:
            save()
        print(f"[rank {rank}] {ordinal}/{len(pending)} {video_id} "
              f"positions={len(selections)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    save()


if __name__ == "__main__":
    main()
