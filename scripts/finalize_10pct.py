#!/usr/bin/env python3
"""Summarize the scaled comparison and apply the prespecified CLM gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, default=Path("manifests/pace_train_10pct_summary.json"))
    args = parser.parse_args()
    rows = json.loads((args.result_dir / "pace200_results.json").read_text())
    curve = json.loads((args.result_dir / "curve.json").read_text())
    serving = json.loads((args.result_dir / "serving_benchmark.json").read_text())
    manifest = json.loads(args.manifest_summary.read_text())
    token_baseline = json.loads((ROOT / "outputs/results/pace200_v1/timing_paucf/pace_token_generation.json").read_text())
    by_method = {name: sorted((r for r in rows if r["method"] == name), key=lambda r: r["seed"])
                 for name in ("linear", "mlp", "clm_128", "clm_256", "clm_short_256")}
    if any(len(items) != 3 for items in by_method.values()):
        raise RuntimeError("missing full-size seed results")
    mlp = by_method["mlp"]
    gate = {}
    for name in ("clm_128", "clm_256", "clm_short_256"):
        checks = []
        for clm, baseline in zip(by_method[name], mlp, strict=True):
            if clm["seed"] != baseline["seed"]:
                raise RuntimeError("seed mismatch")
            checks.append({
                "seed": clm["seed"],
                "delta_pauc_f_points": clm["no_judge_pauc_f"]["pauc_f"] - baseline["no_judge_pauc_f"]["pauc_f"],
                "delta_macro_f1_points": 100 * (clm["classification"]["macro_f1"] - baseline["classification"]["macro_f1"]),
                "delta_fires_per_video": clm["behavior"]["fires_per_video"] - baseline["behavior"]["fires_per_video"],
            })
        delta_by_size = {}
        for size in (768, 3000, 12000, 24300):
            clm_dev = [r["dev_calibrated"]["weighted_macro_f1"] for r in curve if r["method"] == name and r["train_size"] == size]
            mlp_dev = [r["dev_calibrated"]["weighted_macro_f1"] for r in curve if r["method"] == "mlp" and r["train_size"] == size]
            delta_by_size[size] = 100 * (mean(clm_dev) - mean(mlp_dev))
        grows = delta_by_size[24300] > delta_by_size[768]
        gate[name] = {
            "seeds": checks,
            "dev_delta_macro_f1_points_by_size": delta_by_size,
            "advantage_grows": grows,
            "passes": grows and all(
                r["delta_pauc_f_points"] >= 2
                and r["delta_macro_f1_points"] >= 1
                and r["delta_fires_per_video"] <= 0 for r in checks
            ),
        }
    summary = {
        "manifest": manifest,
        "methods": {
            name: {
                "pauc_f_mean": mean([r["no_judge_pauc_f"]["pauc_f"] for r in items]),
                "macro_f1_mean_percent": 100 * mean([r["classification"]["macro_f1"] for r in items]),
                "fires_per_video_mean": mean([r["behavior"]["fires_per_video"] for r in items]),
                "timing_f1_4s_mean_percent": 100 * mean([r["behavior"]["timing_f1_4s"] for r in items]),
            } for name, items in by_method.items()
        },
        "clm_gate": gate,
        "continue_clm": any(value["passes"] for value in gate.values()),
        "pace_token_baseline_pauc_f": token_baseline["pauc_f"],
        "serving": serving["methods"],
    }
    summary["deployment_candidates"] = [name for name in gate if gate[name]["passes"]
        and token_baseline["pauc_f"] - summary["methods"][name]["pauc_f_mean"] <= 3
        and serving["methods"][name]["median_ms"] < serving["methods"]["pace_token"]["median_ms"]]
    output = args.result_dir / "decision_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PACE-CLM 10% scaling result", "",
        f"Pool: {manifest['pool_size']} exact Stream-Gemma stride-10 videos; {manifest['decision_count']} decisions.",
        "", "| Method | PAUC-F | Macro-F1 | Fires/video |", "|---|---:|---:|---:|",
    ]
    for name, result in summary["methods"].items():
        lines.append(f"| {name} | {result['pauc_f_mean']:.2f} | {result['macro_f1_mean_percent']:.2f} | {result['fires_per_video_mean']:.2f} |")
    lines += ["", f"PACE token baseline PAUC-F: {token_baseline['pauc_f']:.2f}.",
              f"CLM gate: {'PASS' if summary['continue_clm'] else 'FAIL'}.",
              f"Deployment candidates: {', '.join(summary['deployment_candidates']) or 'none'}.", ""]
    for name, result in gate.items():
        lines.append(f"- {name}: {'pass' if result['passes'] else 'fail'}; "
                     + ", ".join(f"seed {r['seed']} ΔPAUC-F {r['delta_pauc_f_points']:+.2f}, "
                                 f"Δmacro-F1 {r['delta_macro_f1_points']:+.2f}, "
                                 f"Δfires/video {r['delta_fires_per_video']:+.2f}"
                                 for r in result["seeds"]))
    lines += ["", "## Serving latency", ""]
    for name, timing in summary["serving"].items():
        lines.append(f"- {name}: median {timing['median_ms']:.1f} ms, mean {timing['mean_ms']:.1f} ms ({timing['count']} paired decisions).")
    (args.result_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"continue_clm": summary["continue_clm"], "methods": summary["methods"]}, indent=2))


if __name__ == "__main__":
    main()
