#!/usr/bin/env python3
"""Validate and merge complete WEB/EGO rollout shards for official scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    manifest = json.loads((root / "manifest.json").read_text())
    wanted = manifest["question_ids"]
    if len(wanted) != len(set(wanted)):
        raise RuntimeError("duplicate question IDs in frozen manifest")
    records = {}
    for shard in sorted(root.glob("predictions.rank*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["question_id"]
            if key in records:
                raise RuntimeError(f"duplicate prediction: {key}")
            # The worker records this counter cumulatively across its shard;
            # it is not a per-video diagnostic and is not part of benchmark IO.
            record.pop("pace_clm_actions", None)
            records[key] = record
    if set(records) != set(wanted):
        raise RuntimeError(f"incomplete prediction set: {len(records)} / {len(wanted)}; missing={list(set(wanted)-set(records))[:5]}")
    if any(not isinstance(records[key]["model_response_list"], list) for key in wanted):
        raise RuntimeError("malformed model_response_list")
    output = root / "predictions.jsonl"
    output.write_text("".join(json.dumps(records[key], sort_keys=True) + "\n" for key in wanted))
    fires = sum(len(records[key]["model_response_list"]) for key in wanted)
    print(json.dumps({"benchmark": manifest["benchmark"], "videos": len(wanted), "fires": fires, "fires_per_video": fires / len(wanted), "predictions": str(output)}, indent=2))


if __name__ == "__main__":
    main()
