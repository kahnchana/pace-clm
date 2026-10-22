#!/usr/bin/env python3
"""Merge and validate the shared frozen PACE teacher scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl")
    p.add_argument("--input-dir", type=Path, default=ROOT / "outputs/pace-clm-exp10/cache/teacher")
    p.add_argument("--world-size", type=int, default=8)
    a = p.parse_args()
    manifest = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    source_sha = sha(a.manifest)
    shards = [torch.load(a.input_dir / f"teacher.rank{rank:02d}.pt", map_location="cpu", weights_only=False)
              for rank in range(a.world_size)]
    if any(s["source_sha256"] != source_sha or s["world_size"] != a.world_size or
           s["max_videos_per_rank"] != 0 for s in shards):
        raise RuntimeError("teacher shard provenance mismatch")
    if any(s["tag_strings"] != shards[0]["tag_strings"] for s in shards):
        raise RuntimeError("teacher action strings differ across ranks")
    cached = [row for shard in shards for row in shard["rows"]]
    lookup = {(r["video_id"], int(r["frame_index"])): r for r in cached}
    if len(lookup) != len(cached) or len(lookup) != len(manifest):
        raise RuntimeError("missing or duplicate teacher positions")
    scores = []
    for source in manifest:
        key = (source["video_id"], int(source["frame_index"]))
        row = lookup[key]
        if int(row["gold_action_id"]) != int(source["gold_action_id"]):
            raise RuntimeError(f"teacher label mismatch: {key}")
        value = row["log_scores"].float()
        if value.shape != (3,) or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid teacher scores: {key}")
        scores.append(value)
    out = a.input_dir / "teacher.pt"
    if out.exists():
        existing = torch.load(out, map_location="cpu", weights_only=False)
        if (existing["manifest_sha256"] != source_sha or
            existing["keys"] != [(r["video_id"], int(r["frame_index"])) for r in manifest] or
            not torch.equal(existing["gold_action_ids"], torch.tensor([r["gold_action_id"] for r in manifest])) or
            not torch.equal(existing["log_scores"], torch.stack(scores))):
            raise RuntimeError("existing teacher merge differs from verified shards")
        print(json.dumps({"positions": len(scores), "output": str(out),
                          "sha256": sha(out), "reused": True}, indent=2), flush=True)
        return
    torch.save({"manifest_sha256": source_sha, "keys": [
        (r["video_id"], int(r["frame_index"])) for r in manifest],
        "gold_action_ids": torch.tensor([r["gold_action_id"] for r in manifest]),
        "log_scores": torch.stack(scores), "tag_strings": shards[0]["tag_strings"],
        "model_revision": shards[0]["model_revision"]}, out)
    print(json.dumps({"positions": len(scores), "output": str(out),
                      "sha256": sha(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
