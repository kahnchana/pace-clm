#!/usr/bin/env python3
"""Freeze four-state causal sequences for the E07/E08 temporal pair."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
OUT = ROOT / "manifests/exp10"
OLD = Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features/train.pt")
sys.path[:0] = [str(ROOT / "src"), str(SG / "src")]
from pace_clm.experiment import absolute_timeline, state_timeline
from pace_clm.labels import Action


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen temporal artifact differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    run5_path = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    run5 = [json.loads(line) for line in run5_path.read_text().splitlines() if line.strip()]
    old = torch.load(OLD, map_location="cpu", weights_only=False)
    dev = [r for r in old["metadata"] if r["selection_split"] == "dev"]
    if len(run5) != 24300 or len(dev) != 2700:
        raise RuntimeError("run5/dev size mismatch")
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for split, rows in (("train", run5), ("dev", dev)):
        for row in rows:
            groups[row["video_id"]].append((split, row))
    if any(len({split for split, _ in group}) != 1 for group in groups.values()):
        raise RuntimeError("train/dev video overlap")
    raw_by_video = {video: int(group[0][1]["raw_index"]) for video, group in groups.items()}
    raw_to_video = {raw: video for video, raw in raw_by_video.items()}
    if len(raw_to_video) != len(raw_by_video):
        raise RuntimeError("multiple videos per annotation")
    records = {}
    with (SG / "datasets/markii/train.jsonl").open() as f:
        for raw, line in enumerate(f):
            if raw in raw_to_video:
                records[raw_to_video[raw]] = json.loads(line)
    if set(records) != set(groups):
        raise RuntimeError("missing temporal annotation")
    metadata = json.loads((SG / "datasets/markii/metadata.json").read_text())
    features = {}
    keys_by_split = {"train": [], "dev": []}
    for video in sorted(groups):
        timeline = state_timeline(records[video], metadata)
        count = int(timeline["source_stop"])
        _, states = absolute_timeline(timeline, count)
        split = groups[video][0][0]
        for _, source in groups[video]:
            index = int(source["frame_index"])
            if not 0 <= index < count:
                raise RuntimeError("temporal current frame outside timeline")
            sequence = [max(0, index - offset) for offset in (3, 2, 1, 0)]
            keys_by_split[split].append({"video_id": video, "frame_index": index,
                                         "gold_action_id": int(source["gold_action_id"]),
                                         "sequence": sequence})
            for frame in sequence:
                action = "Hold" if states[frame] == "Standby" else states[frame]
                key = (video, frame)
                existing = features.get(key)
                if existing and existing["split"] != split:
                    raise RuntimeError("temporal train/dev feature overlap")
                features[key] = {"split": split, "raw_index": raw_by_video[video],
                                 "video_id": video, "frame_index": frame,
                                 "gold_action": action, "gold_action_id": int(Action[action]),
                                 "stratum": "temporal"}
    for split in ("train", "dev"):
        write_frozen(OUT / f"temporal_{split}_keys.jsonl",
                     "".join(json.dumps(r, sort_keys=True) + "\n" for r in keys_by_split[split]))
    rows = sorted(features.values(), key=lambda r: (r["video_id"], r["frame_index"]))
    write_frozen(OUT / "temporal_features.jsonl",
                 "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    summary = {"train_positions": len(keys_by_split["train"]),
               "dev_positions": len(keys_by_split["dev"]), "feature_positions": len(rows),
               "videos": len(groups), "source_run5_sha256": sha(run5_path),
               "source_dev_sha256": sha(OLD),
               "feature_manifest_sha256": sha(OUT / "temporal_features.jsonl")}
    write_frozen(OUT / "temporal.freeze.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
