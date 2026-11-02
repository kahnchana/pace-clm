#!/usr/bin/env python3
"""Collect completed seed-42 training and judged rollout metrics."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/jevany"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    train = read(BASE / "full-j1-seed42/summary.rank0.json")
    validation = {
        split: read(BASE / f"evaluation/validation/{split}/official/results/exp10_judged_summary.json")
        for split in ("selection", "final")
    }
    benchmarks = {
        split: read(BASE / f"evaluation/benchmarks/{split}/pauc_f_summary-gpt-4.1-v1.json")
        for split in ("web", "ego")
    }
    if train["global_presentations"] != 263760 or train["world"] != 8 or train["seed"] != 42:
        raise ValueError("full training exposure, GPU count, or seed mismatch")
    for split, expected in (("web", 500), ("ego", 326)):
        if benchmarks[split]["num_videos"] != expected:
            raise ValueError(f"incomplete {split} benchmark")
    summary = {
        "train": {key: train[key] for key in (
            "global_presentations", "optimizer_steps", "seconds", "first_loss",
            "last_loss", "mean_loss_first_nine", "mean_loss_last_nine",
        )},
        "validation": {split: {key: result[key] for key in (
            "pauc_f", "fire_precision", "total_physical_fires")}
            for split, result in validation.items()},
        "benchmarks": {split: {key: result[key] for key in (
            "pauc_f", "fire_precision", "num_fires", "num_videos", "scorer_version")}
            for split, result in benchmarks.items()},
    }
    path = BASE / "reports/full-results.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# JevAny streaming controller — single-seed results", "",
        f"Training: {train['global_presentations']:,} presentations across eight GPUs, "
        f"seed 42, {train['optimizer_steps']:,} optimizer updates.", "",
        "| Split | Judged PAUC-F | Fire precision | Fires | Videos |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("selection", "final"):
        result = validation[split]
        lines.append(f"| Reused validation {split} | {result['pauc_f']:.2f} | "
                     f"{result['fire_precision']:.2f} | {result['total_physical_fires']} | 200 |")
    for split in ("web", "ego"):
        result = benchmarks[split]
        lines.append(f"| {split.upper()} | {result['pauc_f']:.2f} | "
                     f"{result['fire_precision']:.2f} | {result['num_fires']} | "
                     f"{result['num_videos']} |")
    lines += ["", "All rollout scores use GPT-4.1 and the existing PAUC-F scorers. "
              "Selection and final validation are reused cohorts, not untouched confirmation.", ""]
    (BASE / "reports/FINAL_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
