#!/usr/bin/env python3
"""Summarize the frozen six-condition ablation and selected-mixture scale-up."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
selection_path = a.root / "results/selection.results.json"
final_path = a.root / "results/final.results.json"
choice_path = a.root / "results/selected_mixture.json"
selection = json.loads(selection_path.read_text())
final = json.loads(final_path.read_text())
reference_path = a.root / "results/final_reference/final.results.json"
reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
choice = json.loads(choice_path.read_text())
preflight = json.loads((a.root / "manifests/preflight.json").read_text())


def mean_sd(values):
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def table(runs, conditions):
    lines = ["| Condition | Head | PAUC-F | Macro F1 | Silence F1 | Hold F1 | Response F1 | NLL | ECE | Timing F1@4s | Fires/video | Repeats/video |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by_key = defaultdict(list)
    for run in runs:
        by_key[run["condition"], run["method"]].append(run)
    for condition in conditions:
        for method in ("mlp", "clm_short_256"):
            rows = sorted(by_key[condition, method], key=lambda r: r["seed"])
            if len(rows) != 3:
                raise RuntimeError(f"missing seeds for {condition}/{method}")
            fields = [mean_sd([r["no_judge_pauc_f"]["pauc_f"] for r in rows]),
                      mean_sd([100*r["classification"]["macro_f1"] for r in rows])]
            fields += [mean_sd([100*r["classification"]["per_class"][label]["f1"] for r in rows])
                       for label in ("Silence", "Hold", "Response")]
            fields += [mean_sd([r["classification"]["nll"] for r in rows]),
                       mean_sd([r["classification"]["ece"] for r in rows]),
                       mean_sd([100*r["behavior"]["timing_f1_4s"] for r in rows]),
                       mean_sd([r["behavior"]["fires_per_video"] for r in rows]),
                       mean_sd([r["behavior"]["repeat_fires_on_hold"] / r["no_judge_pauc_f"]["num_sampled_videos"] for r in rows])]
            lines.append("| " + " | ".join([condition, method, *fields]) + " |")
    return "\n".join(lines)


report = ["# PACE-CLM 10% data-mixture results", "",
          "The six-condition comparison used the fresh 200-video mixture-selection set. Values are mean ± sample SD across seeds 42/43/44; PAUC-F and F1 values are percentages.", "",
          table(selection, ("control", "run1", "run2", "run3", "run4", "run5")), "",
          f"Selected mixture: **{choice['selected_mixture']}**. Mean PAUC-F beat the balanced control: **{choice['control_beaten']}**.", "",
          "## Manifest counts", "",
          "| Condition | Decisions | Silence | Hold | Response | Unique videos | Repeated positions |",
          "|---|---:|---:|---:|---:|---:|---:|"]
for condition in ("run3", "run4", "run5"):
    row = preflight[condition]
    report.append(f"| {condition} | {row['decisions']} | {row['states']['Silence']} | {row['states']['Hold']} | {row['states']['Response']} | {row['videos']} | {row['repeated_positions']} |")
effective = {
    condition: torch.load(a.root / f"checkpoints/{condition}.mlp.seed42.pt", map_location="cpu", weights_only=False)["run"]["effective_sample_size"]
    for condition in ("run1", "run2")
}
report += ["",
           f"Weighted-loss effective sample size: run 1 **{effective['run1']:.0f}**, run 2 **{effective['run2']:.0f}** out of 24,300 selected decisions. Run 5 replaced **{preflight['run5']['replaced_silence']:,}** Silence decisions with long response-free Silence; all three mask manifests have zero repeated positions."]
report += ["", "## PACE-like exposure on the untouched final set", "",
           "The selected and balanced regimes used the same selected positions, 263,760 decision presentations, and 3,260 optimizer updates per model/seed.", "",
           table(final, ("selected", "balanced")), "",
           "Because run 5 uses unweighted loss, selected and balanced-loss scale-up runs are the same recipe on the same positions and give identical scores.", ""]
if reference is not None:
    report += ["## Frozen 24.3k references on the same final set", "",
               "These checkpoints were trained before the final set was examined and were scored only to put the scale-up result on the same held-out videos.", "",
               table(reference, ("control", "run5")), ""]
    def average_pauc(rows):
        return statistics.mean(r["no_judge_pauc_f"]["pauc_f"] for r in rows)
    selected_mean = average_pauc([r for r in final if r["condition"] == "selected"])
    run5_mean = average_pauc([r for r in reference if r["condition"] == "run5"])
    control_mean = average_pauc([r for r in reference if r["condition"] == "control"])
    report += [f"Across both heads and three seeds, 24.3k run 5 scored {run5_mean:.2f} PAUC-F versus {control_mean:.2f} for the original balanced control; scale-up scored {selected_mean:.2f}. Extra exposure changed PAUC-F by {selected_mean-run5_mean:+.2f} points on this final set.", ""]
selected_rank = next(row for row in choice["ranking"] if row["condition"] == choice["selected_mixture"])
control_rank = next(row for row in choice["ranking"] if row["condition"] == "control")
report += ["## Selection tradeoffs and extraction note", "",
           f"On the selection set, the selected mixture changed fires/video from {control_rank['mean_fires_per_video']:.2f} to {selected_rank['mean_fires_per_video']:.2f}, mean repeat fires per 200 videos from {control_rank['mean_repeat_fires']:.2f} to {selected_rank['mean_repeat_fires']:.2f}, and Hold recall by {100*selected_rank['recall_delta_vs_control']['Hold']:+.2f} percentage points. It therefore did not meet the plan's no-extra-fires/repeats and per-state-recall preferences.", "",
           "The raw training-union extraction differed materially from the prior cache at 9 of 3,185 overlapping positions because OpenCV seek placement changed with denser frame requests. The raw cache is preserved, and the working union cache reuses the prior frozen vector at every overlapping position; see feature_overlap.json.", "",
           "Detailed per-seed metrics, PAUC-F components, per-video scores, and timing/offset breakdowns are in the JSON artifacts beside this report.", ""]
out = a.root / "results/RESULTS.md"
out.write_text("\n".join(report))
print(out)
