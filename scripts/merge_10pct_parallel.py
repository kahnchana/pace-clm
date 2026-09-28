#!/usr/bin/env python3
"""Merge disjoint head and tail extraction shards against the frozen manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def key(row):
    return (row["video_id"], int(row["frame_index"]), int(row["gold_action_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=450)
    args = parser.parse_args()
    paths = [args.input_dir / f"train{suffix}.rank{rank:02d}.pt"
             for suffix in ("", ".tail") for rank in range(8)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    manifest = [json.loads(line) for line in
                (ROOT / "manifests/pace_train_10pct_decisions.jsonl").read_text().splitlines() if line]
    groups = defaultdict(list)
    for row in manifest:
        groups[row["video_id"]].append(row)
    bins = [[] for _ in range(8)]
    loads = [0] * 8
    for video_id in sorted(groups, key=lambda name: (-len(groups[name]), name)):
        rank = min(range(8), key=lambda item: (loads[item], item))
        bins[rank].append(video_id)
        loads[rank] += len(groups[video_id])
    rows = []
    for rank in range(8):
        head_ids = set(bins[rank][:args.split_index])
        tail_ids = set(bins[rank][args.split_index:])
        if not head_ids.issubset(shards[rank]["completed_video_ids"]):
            raise RuntimeError(f"head rank {rank} incomplete")
        if set(shards[rank + 8]["completed_video_ids"]) != tail_ids:
            raise RuntimeError(f"tail rank {rank} incomplete")
        rows.extend(row for row in shards[rank]["rows"] if row["video_id"] in head_ids)
        rows.extend(shards[rank + 8]["rows"])
    rows.sort(key=key)
    if [key(row) for row in rows] != [key(row) for row in manifest]:
        raise RuntimeError("merged feature rows do not match the frozen decision manifest")
    actions = next(shard["action_features"] for shard in shards if shard["action_features"] is not None)
    if any(shard["hidden_size"] != shards[0]["hidden_size"] for shard in shards):
        raise RuntimeError("feature width mismatch")
    features = torch.stack([row.pop("feature") for row in rows])
    if not torch.isfinite(features).all() or not torch.isfinite(actions).all():
        raise RuntimeError("non-finite frozen features")
    output = args.input_dir / "train.pt"
    torch.save({
        "split": "train",
        "hidden_size": shards[0]["hidden_size"],
        "features": features,
        "targets": torch.tensor([row["gold_action_id"] for row in rows]),
        "metadata": rows,
        "action_features": actions,
        "config": shards[0]["config"],
        "extract_gpu_seconds": sum(float(shard["seconds"]) for shard in shards),
    }, output)
    summary = {
        "output": str(output), "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "class_distribution": dict(Counter(row["gold_action"] for row in rows)),
        "extract_gpu_seconds": sum(float(shard["seconds"]) for shard in shards),
        "split_index": args.split_index,
        "shards": [str(path) for path in paths],
    }
    (args.input_dir / "train.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
