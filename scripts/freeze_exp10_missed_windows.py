#!/usr/bin/env python3
"""Freeze E01/E02 missed-window replacements and matched ranking triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
OUT = ROOT / "manifests/exp10"
sys.path[:0] = [str(ROOT / "src"), str(SG / "src")]
from pace_clm.experiment import absolute_timeline, state_timeline
from pace_clm.labels import Action


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen E01 artifact differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, default=ROOT / "outputs/pace-clm-exp10/train_rollout")
    a = p.parse_args()
    source_path = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    selected = read_jsonl(source_path)
    if len(selected) != 24300:
        raise RuntimeError("run5 size mismatch")
    response = [r for r in selected if r["gold_action"] == "Response"]
    replace_count = round(len(response) * 0.2)
    removed = sorted(response, key=lambda r: digest(f"e01-remove:{r['video_id']}:{r['frame_index']}"))[:replace_count]
    removed_keys = {(r["video_id"], int(r["frame_index"])) for r in removed}
    kept = [r for r in selected if (r["video_id"], int(r["frame_index"])) not in removed_keys]
    counts = Counter(r["video_id"] for r in kept)
    gold = json.loads((OUT / "training_rollout.gold.json").read_text())["records"]
    manifest = read_jsonl(OUT / "training_rollout.jsonl")
    by_video = {r["video_id"]: r for r in manifest}
    predictions = read_jsonl(a.rollout_dir / "official/generation/predictions.jsonl")
    by_pred = {r["video_id"]: r for r in predictions}
    if set(by_pred) != set(by_video):
        raise RuntimeError("training rollout incomplete or unexpected")
    metadata = json.loads((SG / "datasets/markii/metadata.json").read_text())
    raw_by_video = {r["video_id"]: int(r["raw_index"]) for r in manifest}
    raw_to_video = {raw: video for video, raw in raw_by_video.items()}
    records = {}
    with (SG / "datasets/markii/train.jsonl").open() as f:
        for raw, line in enumerate(f):
            if raw in raw_to_video:
                records[raw_to_video[raw]] = json.loads(line)
    if set(records) != set(by_video):
        raise RuntimeError("missing training annotation")
    states_by_video = {}
    for video in by_video:
        timeline = state_timeline(records[video], metadata)
        _, states = absolute_timeline(timeline, int(by_video[video]["inference_frame_count"]))
        states_by_video[video] = states
    existing_keys = {(r["video_id"], int(r["frame_index"])) for r in selected}
    candidates = []
    for record in gold:
        video = record["video_id"]
        fires = [float(f["time"]) for f in by_pred[video]["model_response_list"]]
        states = states_by_video[video]
        for window in record["windows"]:
            start, end = float(window["window_start"]), float(window["window_end"])
            if end <= start or any(start <= fire < end for fire in fires):
                continue
            eligible = [int(t) for t in window["decision_timestamps"]
                        if 0 <= int(t) < len(states) and states[int(t)] == "Response"
                        and (video, int(t)) not in existing_keys]
            if not eligible:
                continue
            positive = eligible[0]
            before = int(start) - 1
            after = int(end)
            if before < 0 or after >= len(states) or before >= positive or after <= positive:
                continue
            if states[before] == "Response" or states[after] == "Response":
                continue
            candidates.append((video, positive, before, after, int(window["window_index"])))
    candidates.sort(key=lambda c: digest(f"e01-add:{c[0]}:{c[1]}"))
    additions = []
    triples = []
    used = set()
    for video, positive, before, after, window_index in candidates:
        if len(additions) == replace_count:
            break
        if counts[video] >= 14 or (video, positive) in used:
            continue
        row = {"split": "train", "raw_index": raw_by_video[video], "video_id": video,
               "frame_index": positive, "gold_action": "Response",
               "gold_action_id": int(Action.Response), "stratum": "response",
               "mask_source": "exp10_e01_missed_window"}
        additions.append(row)
        triples.append({"video_id": video, "window_index": window_index,
                        "positive": positive, "before": before, "after": after})
        used.add((video, positive))
        counts[video] += 1
    if len(additions) != replace_count:
        raise RuntimeError(f"only {len(additions)}/{replace_count} distinct capped missed windows")
    train = sorted(kept + additions, key=lambda r: (r["video_id"], int(r["frame_index"])))
    if len(train) != 24300 or max(counts.values()) > 14:
        raise RuntimeError("E01 training budget or video cap mismatch")
    if Counter(r["gold_action"] for r in train) != Counter(r["gold_action"] for r in selected):
        raise RuntimeError("E01 class counts changed")
    feature_rows = {(r["video_id"], int(r["frame_index"])): r for r in train}
    for triple in triples:
        video = triple["video_id"]
        states = states_by_video[video]
        for key in ("before", "after"):
            frame = triple[key]
            action = "Hold" if states[frame] == "Standby" else states[frame]
            feature_rows.setdefault((video, frame), {
                "split": "train", "raw_index": raw_by_video[video], "video_id": video,
                "frame_index": frame, "gold_action": action,
                "gold_action_id": int(Action[action]), "stratum": "e01_ranking_negative"})
    features = sorted(feature_rows.values(), key=lambda r: (r["video_id"], int(r["frame_index"])))
    write_frozen(OUT / "e01_train.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in train))
    write_frozen(OUT / "e01_rank_triples.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in triples))
    write_frozen(OUT / "e01_features.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in features))
    summary = {"train_positions": len(train), "replaced_responses": len(additions),
               "ranking_triples": len(triples), "feature_positions": len(features),
               "class_counts": dict(Counter(r["gold_action"] for r in train)),
               "maximum_per_video": max(counts.values()),
               "run5_sha256": sha(source_path),
               "rollout_predictions_sha256": sha(a.rollout_dir / "official/generation/predictions.jsonl"),
               "train_sha256": sha(OUT / "e01_train.jsonl"),
               "triples_sha256": sha(OUT / "e01_rank_triples.jsonl"),
               "features_sha256": sha(OUT / "e01_features.jsonl")}
    write_frozen(OUT / "e01.freeze.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
