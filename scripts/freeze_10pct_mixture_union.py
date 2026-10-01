#!/usr/bin/env python3
"""Freeze the union of selected positions so each Qwen feature is extracted once."""

import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--manifest-dir", type=Path, required=True)
a = p.parse_args()
rows = {}
for name in ("run3", "run4", "run5"):
    for line in (a.manifest_dir / f"{name}.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["video_id"], row["frame_index"]
        if key in rows and rows[key]["gold_action_id"] != row["gold_action_id"]:
            raise RuntimeError(f"conflicting label at {key}")
        rows[key] = row
ordered = [rows[key] for key in sorted(rows)]
path = a.manifest_dir / "union.jsonl"
rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
if path.exists() and path.read_text() != rendered:
    raise FileExistsError(path)
path.write_text(rendered)
print(json.dumps({"rows": len(ordered), "videos": len({r['video_id'] for r in ordered}),
                  "sha256": hashlib.sha256(rendered.encode()).hexdigest()}))
