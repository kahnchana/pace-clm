#!/usr/bin/env python3
"""Group similar-length decisions in bounded windows without dropping rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def estimated_tokens(row: dict) -> int:
    # Each image occupies 144 language slots. Text is a smaller correction.
    turns = row["turns"]
    text_chars = len(row["question"])
    text_chars += sum(len(str(turn.get("content", ""))) for turn in turns)
    if row.get("memory"):
        text_chars += len(str(row["memory"].get("content", "")))
    return 144 * len(turns) + text_chars // 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, default=384)
    parser.add_argument("--batches-per-window", type=int, default=20)
    args = parser.parse_args()
    if args.global_batch < 1 or args.batches_per_window < 1:
        parser.error("batch and window counts must be positive")
    window = args.global_batch * args.batches_per_window
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    count = 0
    with args.source.open("rb") as source, args.output.open("wb") as output:
        while True:
            lines = [source.readline() for _ in range(window)]
            lines = [line for line in lines if line]
            if not lines:
                break
            for line in lines:
                source_hash.update(line)
            ranked = sorted(lines, key=lambda line: estimated_tokens(json.loads(line)))
            for line in ranked:
                output.write(line)
                output_hash.update(line)
            count += len(lines)
    summary = {
        "source": str(args.source), "output": str(args.output),
        "presentations": count, "global_batch": args.global_batch,
        "batches_per_window": args.batches_per_window,
        "source_sha256": source_hash.hexdigest(),
        "output_sha256": output_hash.hexdigest(),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
