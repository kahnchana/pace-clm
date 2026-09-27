#!/usr/bin/env python3
"""Merge distributed feature shards into one deterministic artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "pace200"), required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    paths = [args.input_dir / f"{args.split}.rank{rank:02d}.pt" for rank in range(args.world_size)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: (row["video_id"], row["frame_index"], row["gold_action_id"]))
    action_features = next(
        shard["action_features"] for shard in shards if shard["action_features"] is not None
    )
    output = args.input_dir / f"{args.split}.pt"
    torch.save(
        {
            "split": args.split,
            "hidden_size": shards[0]["hidden_size"],
            "features": torch.stack([row.pop("feature") for row in rows]),
            "targets": torch.tensor([row["gold_action_id"] for row in rows]),
            "metadata": rows,
            "action_features": action_features,
            "config": shards[0]["config"],
            "extract_gpu_seconds": sum(float(shard["seconds"]) for shard in shards),
        },
        output,
    )
    summary = {
        "output": str(output),
        "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "class_distribution": dict(Counter(row["gold_action"] for row in rows)),
        "extract_gpu_seconds": sum(float(shard["seconds"]) for shard in shards),
    }
    (args.input_dir / f"{args.split}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
