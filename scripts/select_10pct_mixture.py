#!/usr/bin/env python3
"""Freeze one mixture from fresh-selection results, never from PACE-200."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--selection-results", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
runs = json.loads(a.selection_results.read_text())
conditions = ("control", "run1", "run2", "run3", "run4", "run5")
ranked = []
for condition in conditions:
    rows = [r for r in runs if r["condition"] == condition]
    if len(rows) != 6 or {r["method"] for r in rows} != {"mlp", "clm_short_256"}:
        raise RuntimeError(f"incomplete six-seed condition: {condition}")
    scores = [r["no_judge_pauc_f"]["pauc_f"] for r in rows]
    fires = [r["behavior"]["fires_per_video"] for r in rows]
    repeats = [r["behavior"]["repeat_fires_on_hold"] for r in rows]
    recalls = {name: statistics.mean(r["classification"]["per_class"][name]["recall"] for r in rows)
               for name in ("Silence", "Hold", "Response")}
    ranked.append({"condition": condition, "mean_pauc_f": statistics.mean(scores),
                   "sd_pauc_f": statistics.stdev(scores), "mean_fires_per_video": statistics.mean(fires),
                   "mean_repeat_fires": statistics.mean(repeats), "mean_state_recall": recalls,
                   "by_method": {method: statistics.mean(r["no_judge_pauc_f"]["pauc_f"] for r in rows if r["method"] == method)
                                 for method in ("mlp", "clm_short_256")}})
control = next(row for row in ranked if row["condition"] == "control")
eligible = [row for row in ranked if row["condition"] != "control"]
# PAUC-F is the prespecified primary rank. Near ties (<=0.5 point) use fires,
# then seed spread. This is a deterministic rule frozen before final-set access.
eligible.sort(key=lambda row: -row["mean_pauc_f"])
top_score = eligible[0]["mean_pauc_f"]
close = [row for row in eligible if top_score - row["mean_pauc_f"] <= 0.5]
winner = min(close, key=lambda row: (row["mean_fires_per_video"], row["sd_pauc_f"], row["condition"]))
for row in ranked:
    row["delta_pauc_f_vs_control"] = row["mean_pauc_f"] - control["mean_pauc_f"]
    row["delta_fires_per_video_vs_control"] = row["mean_fires_per_video"] - control["mean_fires_per_video"]
    row["recall_delta_vs_control"] = {name: row["mean_state_recall"][name] - control["mean_state_recall"][name]
                                       for name in ("Silence", "Hold", "Response")}
selected = {"selected_mixture": winner["condition"], "ranking": ranked,
            "selection_rule": "mean no-judge PAUC-F across MLP and CLM, three seeds each; within 0.5 point lower fires then lower variance",
            "control_beaten": winner["mean_pauc_f"] > control["mean_pauc_f"],
            "selection_results_sha256": hashlib.sha256(a.selection_results.read_bytes()).hexdigest()}
rendered = json.dumps(selected, indent=2, sort_keys=True) + "\n"
if a.output.exists() and a.output.read_text() != rendered:
    raise FileExistsError(f"selection is already frozen: {a.output}")
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(rendered)
print(json.dumps({"selected_mixture": winner["condition"], "control_beaten": selected["control_beaten"],
                  "mean_pauc_f": winner["mean_pauc_f"], "control_mean_pauc_f": control["mean_pauc_f"]}))
