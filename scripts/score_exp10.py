#!/usr/bin/env python3
"""Score exp10 with the unchanged Markii PAUC-F functions and truthful provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.markii_paucf_200 import (
    JUDGE_MODEL,
    OMEGA,
    SCORER_VERSION,
    _bootstrap,
    _sanity_checks,
    _score_records,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_frozen(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise RuntimeError(f"scoring artifact differs: {path}")
    else:
        path.write_text(rendered)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.rollout_dir / "official"
    predictions = read_jsonl(root / "generation/predictions.jsonl")
    gold = json.loads((root / "gold/paucf_4s_gold.json").read_text())["records"]
    judge = json.loads((root / "judge/pauc_llm_scores.json").read_text())
    if len(predictions) != len(gold) or {r["question_id"] for r in predictions} != {r["question_id"] for r in gold}:
        raise RuntimeError("prediction/gold mismatch")
    if not _sanity_checks(gold)["passed"]:
        raise RuntimeError("official scorer sanity checks failed")
    judged, per_video = _score_records(predictions, gold, judge["scores_by_request_key"])
    timing, _ = _score_records(predictions, gold, None)
    judged.update({
        "metric": "judged PAUC-F@4s", "headline": True,
        "scorer_version": SCORER_VERSION, "judge_model": JUDGE_MODEL,
        "omega": OMEGA, "window_seconds": 4.0,
        "fires_per_video": judged["total_physical_fires"] / len(predictions),
        "per_video_macro_score": mean(r["pauc_f"] for r in per_video),
        "bootstrap_95_percent_ci": _bootstrap(per_video),
    })
    timing.update({"metric": "timing-only PAUC-F@4s", "headline": False,
                   "omega": OMEGA})
    results = root / "results"
    results.mkdir(exist_ok=True)
    write_frozen(results / "exp10_judged_summary.json", judged)
    write_frozen(results / "exp10_per_video.json", per_video)
    write_frozen(results / "exp10_timing_summary.json", timing)
    print(json.dumps({"videos": len(predictions), "judged_pauc_f": judged["pauc_f"],
                      "judged_fire_precision": judged["fire_precision"],
                      "timing_pauc_f": timing["pauc_f"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
