#!/usr/bin/env python3
"""Paired video bootstrap for frozen WEB/EGO finalists and raw reference outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.pauc_f import (
    OFFICIAL_SCORER,
    _load_official_scorer,
    assign_value_adding_fires,
    fire_times,
)

OUT = ROOT / "outputs/pace-clm-exp10/full"
OFFICIAL = _load_official_scorer(OFFICIAL_SCORER)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_video(root: Path) -> dict[str, tuple[float, int, int, int]]:
    predictions = {r["question_id"]: r for r in read_jsonl(root / "predictions.jsonl")}
    scored = json.loads((root / "pauc_llm_scores-gpt-4.1.json").read_text())
    if len(predictions) != len(scored):
        raise RuntimeError("paired bootstrap predictions/scored count mismatch")
    result = {}
    for record in scored:
        key = record["question_id"]
        fires = fire_times(predictions[key])
        useful = len(assign_value_adding_fires(fires, record["answer"]))
        auc_sum = 0.0
        for turn in record["answer"]:
            if not turn["judge_scores"]:
                auc_sum += 0.25
                continue
            points = [[float(t) - turn["reply_timespan"][0], score]
                      for t, score in turn["judge_scores"].items()]
            auc_sum += OFFICIAL.area_under_line_ratio(
                points, turn["reply_timespan"][1] - turn["reply_timespan"][0], 2, 0.5, 0.5)
        result[key] = (auc_sum, len(record["answer"]), useful, len(fires))
    if set(result) != set(predictions):
        raise RuntimeError("paired bootstrap question IDs differ")
    arrays = np.asarray(list(result.values()), dtype=np.float64)
    recall = arrays[:, 0].sum() / arrays[:, 1].sum()
    precision = arrays[:, 2].sum() / arrays[:, 3].sum() if arrays[:, 3].sum() else 0.0
    point = 100 * 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    official_point = json.loads((root / "pauc_f_summary-gpt-4.1-v1.json").read_text())["pauc_f"]
    if abs(point - official_point) > 1e-7:
        raise RuntimeError(f"bootstrap point estimate disagrees with official scorer: {point} vs {official_point}")
    return result


def resampled_scores(rows: np.ndarray, draws: np.ndarray) -> np.ndarray:
    sampled = rows[draws].sum(axis=1)
    recall = sampled[:, 0] / sampled[:, 1]
    precision = np.divide(sampled[:, 2], sampled[:, 3], out=np.zeros_like(recall), where=sampled[:, 3] > 0)
    return np.divide(200 * recall * precision, recall + precision,
                     out=np.zeros_like(recall), where=recall + precision > 0)


def main() -> None:
    advanced = json.loads((ROOT / "manifests/exp10/confirmation_decision.json").read_text())["advanced_to_web_ego"]
    results = {}
    for benchmark, expected in (("web", 500), ("ego", 326)):
        if not advanced:
            results[benchmark] = {"advanced_finalists": [], "paired_intervals": {}}
            continue
        roots = {"run5": ROOT / "outputs/pace-clm-web-ego-full" / benchmark,
                 "PACE": OUT / benchmark / "PACE"}
        roots.update({run: OUT / benchmark / run for run in advanced})
        data = {name: per_video(root) for name, root in roots.items()}
        ids = sorted(data["run5"])
        if len(ids) != expected or any(set(value) != set(ids) for value in data.values()):
            raise RuntimeError("paired WEB/EGO videos differ")
        arrays = {name: np.asarray([value[key] for key in ids], dtype=np.float64)
                  for name, value in data.items()}
        rng = np.random.default_rng(42)
        draws = rng.integers(0, len(ids), size=(10000, len(ids)))
        boot = {name: resampled_scores(value, draws) for name, value in arrays.items()}
        split = {}
        for run in advanced:
            split[run] = {}
            for reference in ("run5", "PACE"):
                difference = boot[run] - boot[reference]
                point = json.loads((roots[run] / "pauc_f_summary-gpt-4.1-v1.json").read_text())["pauc_f"] - \
                        json.loads((roots[reference] / "pauc_f_summary-gpt-4.1-v1.json").read_text())["pauc_f"]
                low, high = np.quantile(difference, [0.025, 0.975])
                split[run][reference] = {"difference_points": point, "paired_95_percent_ci": [float(low), float(high)]}
        results[benchmark] = {"videos": len(ids), "bootstrap_draws": len(draws), "seed": 42,
                              "paired_intervals": split}
    path = OUT / "paired_intervals.json"
    content = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError("paired interval report differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
