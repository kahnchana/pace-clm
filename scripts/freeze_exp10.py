#!/usr/bin/env python3
"""Freeze the previously unused Markii selection and confirmation cohorts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
OUT = ROOT / "manifests/exp10"
SOURCE = SG / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen artifact changed: {path}")
    else:
        path.write_text(content)


def gold(rows_: list[dict]) -> dict:
    records = []
    for row in rows_:
        runs = row["exact_caption_response_runs"]
        windows = []
        end_of_video = float(row["physical_duration"])
        for i, run in enumerate(runs):
            start = float(run["response_mark"])
            following = float(runs[i + 1]["response_mark"]) if i + 1 < len(runs) else math.inf
            end = min(start + 4.0, following, end_of_video)
            windows.append({
                "window_index": i, "response_mark": start, "window_start": start,
                "window_end": end, "end_inclusive": False,
                "length_seconds": max(0.0, end - start), "gold_text": run["gold_text"],
                "annotation_end_time": run["annotation_end_time"],
                "decision_timestamps": [t for t in row["inference_timestamps"] if start <= float(t) < end],
            })
        records.append({
            "question_id": row["question_id"], "video_id": row["video_id"],
            "question": row["question"], "physical_video_end": end_of_video,
            "windows": windows,
        })
    return {"protocol_version": "markii-paucf-200-zss7-yesno-v1", "window_seconds": 4.0,
            "boundary_semantics": "[start,end)", "records": records}


def main() -> None:
    population_path = SOURCE / "manifests/valid_population.jsonl"
    old_paths = [
        SOURCE / "subset/subset_200_manifest.jsonl",
        ROOT / "outputs/pace-clm-mixtures/validation/selection.jsonl",
        ROOT / "outputs/pace-clm-mixtures/validation/final.jsonl",
        ROOT / "outputs/pace-clm-gate/validation/selection.jsonl",
        ROOT / "outputs/pace-clm-gate/validation/final.jsonl",
    ]
    excluded = set()
    counts = {}
    for path in old_paths:
        ids = {r["video_id"] for r in rows(path)}
        if excluded & ids:
            raise RuntimeError(f"previous cohorts overlap: {path}")
        excluded |= ids
        counts[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = len(ids)
    population = rows(population_path)
    if len(population) != len({r["video_id"] for r in population}):
        raise RuntimeError("duplicate validation video")
    available = [r for r in population if r["video_id"] not in excluded]
    if len(available) != 710:
        raise RuntimeError(f"expected 710 unused validation videos; found {len(available)}")
    train_path = ROOT / "manifests/pace_train_stream_gemma_10pct.jsonl"
    train_ids = {r["video_id"] for r in rows(train_path)}
    if train_ids & {r["video_id"] for r in available}:
        raise RuntimeError("training/validation video overlap")
    available.sort(key=lambda r: hashlib.sha256(
        ("pace-clm-exp10-20261010:" + r["video_id"]).encode()).digest())
    sets = {"selection": available[:200], "confirmation": available[200:400]}
    for name, cohort in sets.items():
        frozen_write(OUT / f"{name}.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in cohort))
        frozen_write(OUT / f"{name}.gold.json", json.dumps(gold(cohort), indent=2, sort_keys=True) + "\n")
    manifest = {
        "plan": "docs/exp_plan.md", "hash_salt": "pace-clm-exp10-20261010:",
        "source_sha256": sha(population_path), "old_cohort_sha256": {str(p): sha(p) for p in old_paths},
        "training_pool_sha256": sha(train_path), "old_cohort_counts": counts,
        "available_count": len(available), "untouched_count": 310,
        "sets": {name: {"videos": len(cohort),
                         "frames": sum(int(r["inference_frame_count"]) for r in cohort),
                         "gold_windows": sum(len(r["exact_caption_response_runs"]) for r in cohort),
                         "manifest_sha256": sha(OUT / f"{name}.jsonl"),
                         "gold_sha256": sha(OUT / f"{name}.gold.json")}
                 for name, cohort in sets.items()},
    }
    frozen_write(OUT / "freeze.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
