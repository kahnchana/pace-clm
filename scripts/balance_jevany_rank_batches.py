#!/usr/bin/env python3
"""Redistribute each fixed global batch across ranks for a short GPU probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bucket_jevany_manifest import estimated_tokens


def cost(row: dict) -> int:
    tokens = estimated_tokens(row)
    return tokens + tokens * tokens // 8192 + 144 * len(row["turns"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, default=384)
    parser.add_argument("--world", type=int, default=8)
    args = parser.parse_args()
    if args.global_batch < 1 or args.world < 1 or args.global_batch % args.world:
        parser.error("global batch must be a positive multiple of world")
    per_rank = args.global_batch // args.world
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_counts = Counter()
    output_counts = Counter()
    before_spreads = []
    after_spreads = []
    batches = 0
    with args.source.open("rb") as source, args.output.open("wb") as output:
        while True:
            lines = [source.readline() for _ in range(args.global_batch)]
            if not lines[0]:
                break
            if any(not line for line in lines):
                raise ValueError("source ends inside a global batch")
            rows = [json.loads(line) for line in lines]
            weights = [cost(row) for row in rows]
            original_loads = [sum(weights[rank::args.world]) for rank in range(args.world)]
            before_spreads.append(max(original_loads) / min(original_loads))
            assigned = [[] for _ in range(args.world)]
            loads = [0] * args.world
            for index in sorted(range(args.global_batch), key=lambda i: (-weights[i], i)):
                rank = min((r for r in range(args.world) if len(assigned[r]) < per_rank),
                           key=lambda r: (loads[r], len(assigned[r]), r))
                assigned[rank].append(index)
                loads[rank] += weights[index]
            for indexes in assigned:
                indexes.sort()
            after_spreads.append(max(loads) / min(loads))
            for position in range(per_rank):
                for rank in range(args.world):
                    line = lines[assigned[rank][position]]
                    output.write(line)
                    output_counts[hashlib.sha256(line).digest()] += 1
            input_counts.update(hashlib.sha256(line).digest() for line in lines)
            batches += 1
    if input_counts != output_counts:
        raise ValueError("rebalanced manifest changed the row multiset")
    summary = {
        "source": str(args.source), "output": str(args.output),
        "global_batch": args.global_batch, "world": args.world,
        "batches": batches, "rows": batches * args.global_batch,
        "estimated_rank_load_spread_before_mean": sum(before_spreads) / batches,
        "estimated_rank_load_spread_after_mean": sum(after_spreads) / batches,
        "row_multiset_verified": True,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
