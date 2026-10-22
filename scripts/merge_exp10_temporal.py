#!/usr/bin/env python3
"""Assemble causal four-state tensors from the frozen E07/E08 feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10/cache/temporal"
FROZEN = ROOT / "manifests/exp10"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cache_path = OUT / "temporal.pt"
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    expected = read_jsonl(FROZEN / "temporal_features.jsonl")
    observed = [(r["video_id"], int(r["frame_index"]), int(r["gold_action_id"]))
                for r in data["metadata"]]
    wanted = [(r["video_id"], int(r["frame_index"]), int(r["gold_action_id"])) for r in expected]
    if observed != wanted:
        raise RuntimeError("temporal cache does not match frozen feature manifest")
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(data["metadata"])}
    result = {"source_sha256": sha(cache_path),
              "feature_manifest_sha256": sha(FROZEN / "temporal_features.jsonl"),
              "hidden_size": data["hidden_size"]}
    for split in ("train", "dev"):
        rows = read_jsonl(FROZEN / f"temporal_{split}_keys.jsonl")
        sequences = []
        for row in rows:
            current = int(row["frame_index"])
            frames = [int(frame) for frame in row["sequence"]]
            if len(frames) != 4 or frames != sorted(frames) or frames[-1] != current or any(f > current for f in frames):
                raise RuntimeError("noncausal temporal sequence")
            sequences.append(torch.stack([data["features"][lookup[(row["video_id"], frame)]]
                                          for frame in frames]))
        result[split] = {"keys": [(r["video_id"], int(r["frame_index"])) for r in rows],
                         "features": torch.stack(sequences),
                         "targets": torch.tensor([r["gold_action_id"] for r in rows]),
                         "keys_sha256": sha(FROZEN / f"temporal_{split}_keys.jsonl")}
    out = OUT / "sequences.pt"
    if out.exists():
        raise FileExistsError(out)
    torch.save(result, out)
    print(json.dumps({"train": len(result["train"]["keys"]),
                      "dev": len(result["dev"]["keys"]),
                      "output": str(out), "sha256": sha(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
