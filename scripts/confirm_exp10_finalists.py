#!/usr/bin/env python3
"""Freeze confirmation decisions before any WEB/EGO finalist evaluation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10"


def score(name: str) -> dict:
    path = OUT / "confirmation" / name / "official/results/exp10_judged_summary.json"
    value = json.loads(path.read_text())
    if not value.get("headline") or value["judge_model"] != "gpt-4.1" or value["num_sampled_videos"] != 200:
        raise RuntimeError(f"incomplete judged confirmation: {path}")
    return value


def main() -> None:
    selection = json.loads((ROOT / "manifests/exp10/selection_decision.json").read_text())
    finalists = selection["finalists"]
    if len(finalists) > 2 or len(set(finalists)) != len(finalists):
        raise RuntimeError("invalid frozen finalists")
    baseline = score("run5") if finalists else None
    pace = score("PACE") if finalists else None
    results = {run: score(run) for run in finalists}
    advanced = [run for run in finalists if results[run]["pauc_f"] > baseline["pauc_f"]]
    decision = {"selection_finalists": finalists, "confirmation_run5_pauc_f": baseline["pauc_f"] if baseline else None,
                "confirmation_pace_pauc_f": pace["pauc_f"] if pace else None,
                "candidate_pauc_f": {run: value["pauc_f"] for run, value in results.items()},
                "advanced_to_web_ego": advanced,
                "rule": "strictly improve on run5 judged PAUC-F on untouched confirmation cohort"}
    path = ROOT / "manifests/exp10/confirmation_decision.json"
    content = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError("frozen confirmation decision differs")
    path.write_text(content)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
