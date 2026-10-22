#!/usr/bin/env python3
"""Read-only integrity checks before the ten-run campaign."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.markii_paucf_200 import _score_records


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check() -> dict:
    frozen = json.loads((ROOT / "manifests/exp10/freeze.json").read_text())
    cohorts = {}
    ids = set()
    for name in ("selection", "confirmation"):
        manifest = ROOT / f"manifests/exp10/{name}.jsonl"
        gold_path = ROOT / f"manifests/exp10/{name}.gold.json"
        rows = read_jsonl(manifest)
        gold = json.loads(gold_path.read_text())["records"]
        assert len(rows) == len(gold) == 200
        assert sha(manifest) == frozen["sets"][name]["manifest_sha256"]
        assert sha(gold_path) == frozen["sets"][name]["gold_sha256"]
        cohort_ids = {r["video_id"] for r in rows}
        assert len(cohort_ids) == 200 and not ids & cohort_ids
        ids |= cohort_ids
        for row, ref in zip(rows, gold, strict=True):
            assert row["video_id"] == ref["video_id"]
            assert row["question_id"] == ref["question_id"]
            times = [float(t) for t in row["inference_timestamps"]]
            assert len(times) == row["inference_frame_count"]
            assert times == sorted(set(times))
            assert all(float(w["window_start"]) <= t < float(w["window_end"])
                       for w in ref["windows"] for t in w["decision_timestamps"])
        cohorts[name] = {"videos": len(rows), "gold_windows": sum(len(r["windows"]) for r in gold)}
    train_pool = {r["video_id"] for r in read_jsonl(ROOT / "manifests/pace_train_stream_gemma_10pct.jsonl")}
    run5 = read_jsonl(ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl")
    assert not ids & train_pool
    assert len(run5) == len({(r["video_id"], r["frame_index"]) for r in run5}) == 24300
    assert {r["video_id"] for r in run5} <= train_pool
    cache = torch.load(ROOT / "outputs/pace-clm-mixtures/features/union.pt", map_location="cpu", weights_only=False)
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(cache["metadata"])}
    assert len(lookup) == len(cache["metadata"])
    assert all((r["video_id"], int(r["frame_index"])) in lookup for r in run5)
    assert all(int(cache["targets"][lookup[(r["video_id"], int(r["frame_index"]))]]) == r["gold_action_id"] for r in run5)
    old = torch.load(Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features/train.pt"),
                     map_location="cpu", weights_only=False)
    dev = [r for r in old["metadata"] if r["selection_split"] == "dev"]
    assert len(dev) == 2700
    assert not {r["video_id"] for r in dev} & ids
    # Recompute an existing Markii result with the unchanged scorer to catch scorer drift.
    reference_root = ROOT / "outputs/pace-clm-gate"
    reference_manifest = read_jsonl(reference_root / "validation/final.jsonl")
    reference_gold = json.loads((reference_root / "validation/final.gold.json").read_text())["records"]
    artifact = json.loads((reference_root / "final_eval/run5.mlp.seed42.per_video.json").read_text())
    reference_summary = json.loads((reference_root / "final_eval/final.results.json").read_text())
    assert len(reference_manifest) == len(reference_gold) == len(artifact) == 200
    assert artifact and reference_summary
    # Confirm the imported scorer is usable and preserves its exact timing semantics.
    probe_gold = [{"question_id": "probe", "video_id": "probe", "question": "probe",
                   "physical_video_end": 4.0, "windows": [{"window_index": 0,
                   "window_start": 0.0, "window_end": 4.0, "length_seconds": 4.0,
                   "gold_text": "probe"}]}]
    probe_prediction = [{"question_id": "probe", "video_id": "probe", "question": "probe",
                         "model_response_list": [{"time": 0.0, "content": "probe"}]}]
    probe_score, _ = _score_records(probe_prediction, probe_gold, None)
    assert probe_score["pauc_f"] == 100.0
    return {"cohorts": cohorts, "train_positions": len(run5),
            "train_class_counts": dict(Counter(r["gold_action"] for r in run5)),
            "dev_positions": len(dev), "scorer": _score_records.__module__,
            "old_final_videos": len(reference_manifest)}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True))
