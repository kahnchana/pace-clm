#!/usr/bin/env python3
"""Merge complete rank shards for the unchanged WEB/EGO scorers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    wanted = manifest["question_ids"]
    rows = []
    for rank in range(args.world_size):
        shard = args.input_dir / f"predictions.rank{rank:02d}.jsonl"
        rows.extend(json.loads(line) for line in shard.read_text().splitlines() if line.strip())
    by_id = {row["question_id"]: row for row in rows}
    if len(rows) != len(wanted) or len(by_id) != len(rows) or set(by_id) != set(wanted):
        raise ValueError("missing, duplicated, or unexpected benchmark prediction")
    output = args.input_dir / "predictions.jsonl"
    body = "".join(json.dumps(by_id[question], sort_keys=True) + "\n" for question in wanted)
    if output.exists() and output.read_text() != body:
        raise ValueError("existing merged benchmark predictions differ")
    output.write_text(body)
    print(json.dumps({"split": manifest["split"], "videos": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
