#!/usr/bin/env python3
"""Freeze 4,000 training-only answer-utility candidate times for E09/E10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests/exp10"
ROLLOUT = ROOT / "outputs/pace-clm-exp10/train_rollout"


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
            raise RuntimeError(f"frozen utility candidate artifact differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    source = read_jsonl(OUT / "training_rollout.jsonl")
    gold = json.loads((OUT / "training_rollout.gold.json").read_text())["records"]
    predictions_path = ROLLOUT / "official/generation/predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    by_source = {r["video_id"]: r for r in source}
    by_gold = {r["video_id"]: r for r in gold}
    by_pred = {r["video_id"]: r for r in predictions}
    if set(by_source) != set(by_gold) or set(by_source) != set(by_pred):
        raise RuntimeError("incomplete training rollout for utility selection")
    pools = {"missed_valid_window": [], "current_head_fire": [], "near_window_negative": []}
    for video in sorted(by_source):
        row, reference, prediction = by_source[video], by_gold[video], by_pred[video]
        count = int(row["inference_frame_count"])
        windows = reference["windows"]
        fires = {int(float(f["time"])) for f in prediction["model_response_list"]}
        def window_for(frame: int, windows: list[dict] = windows):
            return next((w for w in windows if w["window_start"] <= frame < w["window_end"]), None)
        for window in windows:
            start, end = float(window["window_start"]), float(window["window_end"])
            if end <= start:
                continue
            if not any(start <= fire < end for fire in fires):
                eligible = [int(t) for t in window["decision_timestamps"]
                            if 0 <= int(t) < count]
                if eligible:
                    pools["missed_valid_window"].append((video, eligible[0]))
            for frame in (int(start) - 1, int(end)):
                if 0 <= frame < count and window_for(frame) is None:
                    pools["near_window_negative"].append((video, frame))
        for fire in fires:
            if 0 <= fire < count:
                pools["current_head_fire"].append((video, fire))
    selected = []
    used = set()
    targets = (("missed_valid_window", 2000), ("current_head_fire", 1000),
               ("near_window_negative", 1000))
    for category, target in targets:
        ordered = sorted(set(pools[category]), key=lambda key: digest(f"e09:{category}:{key[0]}:{key[1]}"))
        category_rows = []
        for video, frame in ordered:
            if (video, frame) in used:
                continue
            window = next((w for w in by_gold[video]["windows"]
                           if w["window_start"] <= frame < w["window_end"]), None)
            if category == "missed_valid_window" and window is None:
                raise RuntimeError("missed-window candidate outside gold window")
            if category == "near_window_negative" and window is not None:
                raise RuntimeError("near-window negative inside gold window")
            source_row = by_source[video]
            category_rows.append({
                "category": category, "video_id": video, "frame_index": frame,
                "question_id": source_row["question_id"], "question": source_row["question"],
                "raw_index": source_row["raw_index"],
                "gold_window_index": window["window_index"] if window else None,
                "gold_text": window["gold_text"] if window else None,
            })
            used.add((video, frame))
            if len(category_rows) == target:
                break
        if len(category_rows) != target:
            raise RuntimeError(f"only {len(category_rows)}/{target} {category} candidates")
        selected.extend(category_rows)
    selected.sort(key=lambda r: (r["video_id"], r["frame_index"]))
    manifest = OUT / "e09_utility_candidates.jsonl"
    write_frozen(manifest, "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
    summary = {"candidates": len(selected), "category_counts": {name: target for name, target in targets},
               "inside_gold_window": sum(r["gold_window_index"] is not None for r in selected),
               "training_rollout_sha256": sha(predictions_path),
               "candidate_manifest_sha256": sha(manifest)}
    write_frozen(OUT / "e09_utility_candidates.freeze.json",
                 json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
