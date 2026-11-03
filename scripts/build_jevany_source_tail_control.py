#!/usr/bin/env python3
"""Replace only the appended mined rows with their removed source rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_jevany_j1sr import SOURCE, rows, sha


def key(row):
    return row["episode_id"], row["turn_index"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    retained = {key(row) for row in rows(args.v2_manifest) if "mining" not in row}
    replacements = defaultdict(list)
    for row in rows(SOURCE):
        if key(row) not in retained:
            replacements[row["label"]].append(row)
    if {label: len(group) for label, group in replacements.items()} != {"Silence": 512, "Response": 128}:
        raise ValueError("removed source quota differs")
    rng = random.Random(args.seed)
    for group in replacements.values():
        rng.shuffle(group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    replaced = 0
    with args.output.open("x") as handle:
        for index, row in enumerate(rows(args.v2_manifest)):
            if "mining" in row:
                if index < 263120:
                    raise ValueError("mined row before the appended block")
                row = replacements[row["label"]].pop()
                replaced += 1
            counts[row["label"]] += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if replaced != 640 or any(replacements.values()) or sum(counts.values()) != 263760:
        args.output.unlink()
        raise ValueError("incorrect source tail")
    audit = {"source_manifest_sha256": sha(SOURCE), "v2_manifest_sha256": sha(args.v2_manifest),
             "control_manifest_sha256": sha(args.output), "seed": args.seed,
             "replaced_mined_rows": replaced, "counts": dict(counts),
             "prefix_rows_identical_to_v2": 263120}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
