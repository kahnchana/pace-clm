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
    parser.add_argument("--split", choices=("train", "pace200", "selection", "final"), required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--shard-suffix", default="")
    parser.add_argument("--output-name")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    suffix = f".{args.shard_suffix}" if args.shard_suffix else ""
    paths = [args.input_dir / f"{args.split}{suffix}.rank{rank:02d}.pt" for rank in range(args.world_size)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    rows = [row for shard in shards for row in shard["rows"]]
    rows.sort(key=lambda row: (row["video_id"], row["frame_index"], row["gold_action_id"]))
    if args.manifest is not None:
        expected = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
        observed_keys = [(r["video_id"], int(r["frame_index"]), int(r["gold_action_id"])) for r in rows]
        expected_keys = sorted((r["video_id"], int(r["frame_index"]), int(r["gold_action_id"])) for r in expected)
        if observed_keys != expected_keys:
            raise RuntimeError("extracted features do not match frozen manifest")
    action_features = next(
        shard["action_features"] for shard in shards if shard["action_features"] is not None
    )
    output_name = args.output_name or args.split
    output = args.input_dir / f"{output_name}.pt"
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
    (args.input_dir / f"{output_name}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
