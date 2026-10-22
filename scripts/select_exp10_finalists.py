#!/usr/bin/env python3
"""Apply the frozen selection and confirmation gates without WEB/EGO access."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10"
RUNS = {f"E{i:02d}": ("clm" if i % 2 else "mlp") for i in range(1, 11)}


def score(cohort: str, name: str) -> dict:
    path = OUT / cohort / name / "official/results/exp10_judged_summary.json"
    value = json.loads(path.read_text())
    if not value.get("headline") or value.get("judge_model") != "gpt-4.1":
        raise RuntimeError(f"not a complete judged result: {path}")
    if value["num_sampled_videos"] != 200:
        raise RuntimeError(f"incomplete cohort: {path}")
    return value


def write_frozen(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen finalist selection differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    baseline = score("selection", "run5")
    pace = score("selection", "PACE")
    selection = {run: score("selection", run) for run in RUNS}
    pair_differences = {}
    for odd in range(1, 11, 2):
        clm, mlp = f"E{odd:02d}", f"E{odd+1:02d}"
        pair_differences[f"{clm}/{mlp}"] = selection[clm]["pauc_f"] - selection[mlp]["pauc_f"]
    eligible = {run: {
        "judged_pauc_f": value["pauc_f"],
        "judged_fire_precision": value["fire_precision"],
        "passes": value["pauc_f"] >= baseline["pauc_f"] and
                  value["fire_precision"] >= baseline["fire_precision"] - 1.0,
    } for run, value in selection.items()}
    finalists = []
    for method in ("clm", "mlp"):
        candidates = [run for run in RUNS if RUNS[run] == method and eligible[run]["passes"]]
        if candidates:
            finalists.append(max(candidates, key=lambda run: (selection[run]["pauc_f"], run)))
    result = {"selection_baseline_run5": baseline["pauc_f"],
              "selection_baseline_pace": pace["pauc_f"],
              "pair_clm_minus_mlp_points": pair_differences,
              "candidate_gates": eligible, "finalists": finalists,
              "selection_rule": "highest judged PAUC-F eligible CLM and eligible MLP; no worse than run5 PAUC-F and at most one point fire-precision loss"}
    write_frozen(ROOT / "manifests/exp10/selection_decision.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
