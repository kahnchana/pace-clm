#!/usr/bin/env python3
"""Freeze E05/E06 half-replay positions and aligned induced-history features."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10"
FROZEN = ROOT / "manifests/exp10"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen replay artifact differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    source_path = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    original = read_jsonl(source_path)
    if len(original) != 24300:
        raise RuntimeError("run5 size mismatch")
    rollout = OUT / "train_rollout"
    predictions = read_jsonl(rollout / "official/generation/predictions.jsonl")
    by_prediction = {r["video_id"]: r for r in predictions}
    gold = json.loads((FROZEN / "training_rollout.gold.json").read_text())["records"]
    by_gold = {r["video_id"]: r for r in gold}
    videos = {r["video_id"] for r in original}
    if set(by_prediction) != videos or set(by_gold) != videos:
        raise RuntimeError("replay rollout incomplete")
    cache = torch.load(ROOT / "outputs/pace-clm-mixtures/features/union.pt", map_location="cpu", weights_only=False)
    cache_lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(cache["metadata"])}
    replay = {}
    capture_sha = sha(source_path)
    replay_dir = rollout / "replay_features"
    for video in sorted(videos):
        path = replay_dir / (hashlib.sha256(video.encode()).hexdigest() + ".pt")
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["video_id"] != video or item["capture_manifest_sha256"] != capture_sha:
            raise RuntimeError("replay feature provenance mismatch")
        for row in item["rows"]:
            key = (row["video_id"], int(row["frame_index"]))
            if key in replay:
                raise RuntimeError("duplicate replay feature")
            replay[key] = row["feature"].float()
    original_keys = {(r["video_id"], int(r["frame_index"])) for r in original}
    if set(replay) != original_keys:
        raise RuntimeError("replay feature keys do not match run5")
    ambiguous = set()
    for row in original:
        if row["gold_action"] != "Response":
            continue
        video, index = row["video_id"], int(row["frame_index"])
        window = next((w for w in by_gold[video]["windows"]
                       if w["window_start"] <= index < w["window_end"]), None)
        if window is None:
            raise RuntimeError("run5 Response lies outside frozen gold window")
        if any(window["window_start"] <= float(f["time"]) < index
               for f in by_prediction[video]["model_response_list"]):
            ambiguous.add((video, index))
    kept = [r for r in original if (r["video_id"], int(r["frame_index"])) not in ambiguous]
    counts = Counter(r["video_id"] for r in kept)
    extras = [r for r in cache["metadata"]
              if (r["video_id"], int(r["frame_index"])) not in original_keys and
              r["video_id"] in videos and r["gold_action"] == "Response"]
    extras.sort(key=lambda r: digest(f"e05-refill:{r['video_id']}:{r['frame_index']}"))
    refill = []
    for row in extras:
        if len(refill) == len(ambiguous):
            break
        if counts[row["video_id"]] >= 14:
            continue
        refill.append(row)
        counts[row["video_id"]] += 1
    if len(refill) != len(ambiguous):
        raise RuntimeError(f"only {len(refill)}/{len(ambiguous)} capped Response refills")
    train = sorted(kept + refill, key=lambda r: (r["video_id"], int(r["frame_index"])))
    if len(train) != 24300 or max(counts.values()) > 14:
        raise RuntimeError("E05/E06 position budget or video cap mismatch")
    if Counter(r["gold_action"] for r in train) != Counter(r["gold_action"] for r in original):
        raise RuntimeError("E05/E06 class distribution changed")
    replay_eligible = [r for r in train
                       if (r["video_id"], int(r["frame_index"])) in original_keys]
    replay_eligible.sort(key=lambda r: digest(f"e05-replay:{r['video_id']}:{r['frame_index']}"))
    replay_keys = {(r["video_id"], int(r["frame_index"])) for r in replay_eligible[:12150]}
    if len(replay_keys) != 12150:
        raise RuntimeError("insufficient replay-eligible positions")
    records = []
    features = []
    for row in train:
        key = (row["video_id"], int(row["frame_index"]))
        source = "generated_history" if key in replay_keys else "gold_history"
        records.append({"video_id": key[0], "frame_index": key[1],
                        "gold_action_id": int(row["gold_action_id"]),
                        "gold_action": row["gold_action"], "feature_source": source})
        features.append(replay[key] if key in replay_keys else cache["features"][cache_lookup[key]].float())
    manifest_path = FROZEN / "e05_train.jsonl"
    write_frozen(manifest_path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    output = OUT / "cache/replay/e05.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save({"keys": [(r["video_id"], r["frame_index"]) for r in records],
                "features": torch.stack(features),
                "targets": torch.tensor([r["gold_action_id"] for r in records]),
                "manifest_sha256": sha(manifest_path),
                "source_run5_sha256": sha(source_path)}, output)
    summary = {"train_positions": len(train), "ambiguous_excluded": len(ambiguous),
               "refilled_positions": len(refill), "generated_history_views": len(replay_keys),
               "gold_history_views": len(train) - len(replay_keys),
               "class_counts": dict(Counter(r["gold_action"] for r in train)),
               "manifest_sha256": sha(manifest_path), "cache_sha256": sha(output),
               "rollout_predictions_sha256": sha(rollout / "official/generation/predictions.jsonl")}
    write_frozen(FROZEN / "e05.freeze.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
