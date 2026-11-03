#!/usr/bin/env python3
"""Place the frozen v2 mined rows in their source replacement slots."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_jevany_j1sr import SOURCE, rows, sha


def row_key(row):
    return row["episode_id"], row["turn_index"]


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    retained = set()
    mined = defaultdict(list)
    original_digests = Counter()
    for row in rows(args.v2_manifest):
        original_digests[row_digest(row)] += 1
        if "mining" in row:
            mined[row["label"]].append(row)
        else:
            retained.add(row_key(row))
    if {label: len(value) for label, value in mined.items()} != {"Silence": 512, "Response": 128}:
        raise ValueError("unexpected mined quota")
    rng = random.Random(args.seed)
    for group in mined.values():
        rng.shuffle(group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    output_digests = Counter()
    mined_positions = []
    with args.output.open("x") as handle:
        for index, source in enumerate(rows(SOURCE)):
            if row_key(source) in retained:
                row = source
            else:
                label = source["label"]
                if not mined[label]:
                    raise ValueError(f"no mined replacement for {label}")
                row = mined[label].pop()
                if row["label"] != label:
                    raise ValueError("replacement changed target")
                mined_positions.append(index)
            counts[row["label"]] += 1
            output_digests[row_digest(row)] += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if any(mined.values()) or output_digests != original_digests or sum(counts.values()) != 263760:
        args.output.unlink()
        raise ValueError("interleaved manifest changed the v2 presentation multiset")
    audit = {"source_manifest_sha256": sha(SOURCE), "v2_manifest_sha256": sha(args.v2_manifest),
             "interleaved_manifest_sha256": sha(args.output), "seed": args.seed,
             "counts": dict(counts), "mined_positions": mined_positions,
             "mined_last_640": sum(index >= 263120 for index in mined_positions)}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "mined_positions"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
