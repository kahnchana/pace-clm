#!/usr/bin/env python3
"""Plan synchronized eight-rank batch sizes from a length-grouped manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from itertools import islice
from pathlib import Path

from bucket_jevany_manifest import estimated_tokens


# Per-rank capacities require full-context stress checks on the target node.
# Keep a smaller fallback for unusually long text with 32 frames.
TIERS = (
    (4, 700, 256),
    (8, 1300, 160),
    (16, 2500, 96),
    (24, 3600, 64),
    (32, 4700, 52),
    (32, float("inf"), 40),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=8)
    args = parser.parse_args()
    if args.world != 8:
        parser.error("capacity tiers are calibrated for eight ranks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows_total = 0
    buffer = deque()
    eof = False
    max_global = args.world * TIERS[0][2]
    with args.manifest.open() as source, args.output.open("w") as output:
        while True:
            while len(buffer) < max_global and not eof:
                line = source.readline()
                if not line:
                    eof = True
                    break
                row = json.loads(line)
                buffer.append((len(row["turns"]), estimated_tokens(row)))
            if not buffer:
                break
            chosen = None
            for max_turns, max_tokens, per_rank in TIERS:
                size = per_rank * args.world
                if len(buffer) < size:
                    continue
                prefix = list(islice(buffer, size))
                if all(turns <= max_turns and tokens <= max_tokens for turns, tokens in prefix):
                    chosen = (size, max(turns for turns, _ in prefix),
                              max(tokens for _, tokens in prefix))
                    break
            if chosen is None:
                size = min(len(buffer) // args.world * args.world, TIERS[-1][2] * args.world)
                if size == 0:
                    raise ValueError("manifest row count is not divisible by eight")
                prefix = list(islice(buffer, size))
                chosen = (size, max(turns for turns, _ in prefix),
                          max(tokens for _, tokens in prefix))
            size, max_turns, max_tokens = chosen
            output.write(json.dumps({"start_row": rows_total,
                                     "global_batch": size,
                                     "max_turns": max_turns,
                                     "max_estimated_tokens": max_tokens}) + "\n")
            counts[size // args.world] += 1
            for _ in range(size):
                buffer.popleft()
            rows_total += size
    summary = {"manifest": str(args.manifest), "rows": rows_total,
               "optimizer_updates": sum(counts.values()),
               "per_rank_batch_counts": dict(sorted(counts.items())),
               "tiers": [[turns, None if tokens == float("inf") else tokens, batch]
                         for turns, tokens, batch in TIERS]}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
