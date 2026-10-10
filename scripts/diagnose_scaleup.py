#!/usr/bin/env python3
"""Summarize dev trajectories and paired final-set errors for the scale-up."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/pace-clm-mixtures"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.root / "results/final.results.json").read_text())
    references = json.loads((args.root / "results/final_reference/final.results.json").read_text())
    report = []
    for method in ("mlp", "clm_short_256"):
        for seed in (42, 43, 44):
            artifact = torch.load(args.root / f"scaleup/selected.{method}.seed{seed}.pt",
                                  map_location="cpu", weights_only=False)
            run = artifact["run"]
            history = run["dev_history"]
            best = min(history, key=lambda row: row["nll"])
            scaled = next(r for r in records if r["condition"] == "selected"
                          and r["method"] == method and r["seed"] == seed)
            reference = next(r for r in references if r["condition"] == "run5"
                             and r["method"] == method and r["seed"] == seed)
            pairs = {}
            for key in scaled["breakdown"]:
                a = scaled["breakdown"][key]
                b = reference["breakdown"][key]
                pairs[key] = {"n": a["n"],
                              "macro_f1_delta_points": 100 * (a["macro_f1"] - b["macro_f1"])}
            report.append({
                "method": method, "seed": seed,
                "best_logged_dev_step": best["step"], "best_logged_dev_nll": best["nll"],
                "final_dev_nll": history[-1]["nll"],
                "final_minus_best_dev_nll": history[-1]["nll"] - best["nll"],
                "reference_best_epoch": reference["best_epoch_from_dev"],
                "pauc_f_reference": reference["no_judge_pauc_f"]["pauc_f"],
                "pauc_f_scaled": scaled["no_judge_pauc_f"]["pauc_f"],
                "pauc_f_delta": scaled["no_judge_pauc_f"]["pauc_f"] - reference["no_judge_pauc_f"]["pauc_f"],
                "response_recall_4s_delta": scaled["behavior"]["response_recall_4s"] - reference["behavior"]["response_recall_4s"],
                "fire_precision_delta": scaled["no_judge_pauc_f"]["fire_precision"] - reference["no_judge_pauc_f"]["fire_precision"],
                "fires_per_video_delta": scaled["behavior"]["fires_per_video"] - reference["behavior"]["fires_per_video"],
                "repeat_fires_delta": scaled["behavior"]["repeat_fires_on_hold"] - reference["behavior"]["repeat_fires_on_hold"],
                "breakdown": pairs,
            })
    summary = {
        "runs": report,
        "mean_best_logged_dev_step": statistics.mean(r["best_logged_dev_step"] for r in report),
        "mean_final_minus_best_dev_nll": statistics.mean(r["final_minus_best_dev_nll"] for r in report),
        "mean_pauc_f_delta": statistics.mean(r["pauc_f_delta"] for r in report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
