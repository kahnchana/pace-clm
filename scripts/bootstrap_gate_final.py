#!/usr/bin/env python3
"""Paired video bootstrap for timing PAUC-F comparisons on the frozen final set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def vectors(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    rows = payload["per_video"] if isinstance(payload, dict) else payload
    by_id = {row["video_id"]: row for row in rows}
    if set(by_id) != set(ids):
        raise ValueError(f"video mismatch: {path}")
    ordered = [by_id[video_id] for video_id in ids]
    return tuple(np.asarray(values, dtype=np.float64) for values in (
        [row["total_physical_fires"] for row in ordered],
        [row["value_adding_fires"] for row in ordered],
        [row["num_gold_windows"] for row in ordered],
        [sum(window["auc_recall"] / 100.0 for window in row["windows"]) for row in ordered],
    ))


def pauc_f(data: tuple[np.ndarray, ...], draws: np.ndarray) -> np.ndarray:
    fires, useful, windows, auc_sum = data
    total_fires = fires[draws].sum(axis=1)
    precision = np.divide(useful[draws].sum(axis=1), total_fires,
                          out=np.zeros(len(draws)), where=total_fires > 0)
    total_windows = windows[draws].sum(axis=1)
    recall = np.divide(auc_sum[draws].sum(axis=1), total_windows,
                       out=np.zeros(len(draws)), where=total_windows > 0)
    return np.divide(200 * precision * recall, precision + recall,
                     out=np.zeros(len(draws)), where=precision + recall > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/pace-clm-gate"))
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in (args.root / "validation/final.jsonl").read_text().splitlines()]
    ids = [row["video_id"] for row in manifest]
    draws = np.random.default_rng(20261002).integers(0, len(ids), size=(args.replicates, len(ids)))
    token = vectors(args.root / "token_score/timing_paucf/pace_token_generation.json", ids)
    token_scores = pauc_f(token, draws)
    conditions = {}
    for condition in ("control", "run5", "run5_early", "run5_hold2"):
        seed_scores = [pauc_f(vectors(args.root / f"final_eval/{condition}.mlp.seed{seed}.per_video.json", ids), draws)
                       for seed in (42, 43, 44)]
        conditions[condition] = np.mean(seed_scores, axis=0)
    comparisons = {
        "run5_minus_token": conditions["run5"] - token_scores,
        "run5_minus_control": conditions["run5"] - conditions["control"],
        "run5_early_minus_run5": conditions["run5_early"] - conditions["run5"],
        "run5_hold2_minus_run5": conditions["run5_hold2"] - conditions["run5"],
    }
    result = {name: {"mean": float(values.mean()),
                     "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
                     "probability_positive": float((values > 0).mean())}
              for name, values in comparisons.items()}
    result["replicates"] = args.replicates
    result["seed"] = 20261002
    result["resampling_unit"] = "video"
    (args.root / "results/bootstrap_final.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
