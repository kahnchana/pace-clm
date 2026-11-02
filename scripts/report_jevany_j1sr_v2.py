#!/usr/bin/env python3
"""Summarize frozen J1-S-R validation and full WEB/EGO confirmation."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from report_jevany_j1s import pooled_score, read_jsonl, video_stats as base_video_stats

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
OUT = ROOT / "outputs/jevany-j1sr-v2"


def video_stats(prediction, score):
    result = base_video_stats(prediction, score)
    turns = score["answer"]
    result["first_events"] = int(bool(turns))
    result["first_event_covered"] = int(bool(turns and turns[0].get("preds")))
    return result


def main():
    validation = {}
    for name in ("j1s", "j1sr"):
        path = OUT / f"evaluation/validation/{name}/official/results/exp10_judged_summary.json"
        validation[name] = json.loads(path.read_text())
    benchmarks = {}
    diagnostics = {}
    for split in ("web", "ego"):
        folders = {"J1-S": ROOT / f"outputs/jevany-j1s/evaluation/benchmarks/{split}",
                   "J1-S-R": OUT / f"evaluation/benchmarks/{split}"}
        benchmarks[split] = {name: json.loads((folder / "pauc_f_summary-gpt-4.1-v1.json").read_text())
                             for name, folder in folders.items()}
        duration = {str(r["question_id"]): float(r["duration"]) for r in json.loads(
            (SG / f"datasets/proactive-videoqa-{split}/anno.json").read_text())}
        cells = {}
        for name, folder in folders.items():
            predictions = {str(r["question_id"]): r for r in read_jsonl(folder / "predictions.jsonl")}
            scores = {str(r["question_id"]): r for r in json.loads(
                (folder / "pauc_llm_scores-gpt-4.1.json").read_text())}
            if predictions.keys() != scores.keys() or predictions.keys() != duration.keys():
                raise ValueError(f"incomplete paired {split} artifacts for {name}")
            cells[name] = {qid: video_stats(predictions[qid], scores[qid]) for qid in duration}
        delta = Counter()
        buckets = defaultdict(lambda: {"videos": 0, "J1-S": Counter(), "J1-S-R": Counter()})
        for qid in duration:
            old, new = cells["J1-S"][qid], cells["J1-S-R"][qid]
            for metric in ("fires", "useful", "wasted", "repeat_fires_in_gold_windows",
                           "first_events", "first_event_covered"):
                delta[metric] += new[metric] - old[metric]
            bucket = "<=60s" if duration[qid] <= 60 else "60-180s" if duration[qid] <= 180 else ">180s"
            buckets[bucket]["videos"] += 1
            for name, row in (("J1-S", old), ("J1-S-R", new)):
                buckets[bucket][name].update({k: row[k] for k in
                    ("fires", "useful", "wasted", "repeat_fires_in_gold_windows",
                     "first_events", "first_event_covered")})
        ids = sorted(duration)
        rng = random.Random(20261029)
        paired = []
        for _ in range(5000):
            sample = rng.choices(ids, k=len(ids))
            paired.append(pooled_score([cells["J1-S-R"][qid] for qid in sample]) -
                          pooled_score([cells["J1-S"][qid] for qid in sample]))
        paired.sort()
        diagnostics[split] = {"delta_j1sr_minus_j1s": dict(delta),
                              "paired_video_bootstrap_95_ci": [paired[125], paired[4874]],
                              "bootstrap_samples": 5000, "bootstrap_seed": 20261029,
                              "duration_buckets": {b: {"videos": v["videos"],
                                  "J1-S": dict(v["J1-S"]), "J1-S-R": dict(v["J1-S-R"])}
                                  for b, v in buckets.items()}}
    report = {"validation": validation, "benchmarks": benchmarks,
              "promoted_pace_historical_pauc_f": {"web": 57.47, "ego": 46.05},
              "paired_diagnostics": diagnostics,
              "validation_delta": validation["j1sr"]["pauc_f"]-validation["j1s"]["pauc_f"],
              "benchmark_deltas": {split: rows["J1-S-R"]["pauc_f"]-rows["J1-S"]["pauc_f"]
                                   for split, rows in benchmarks.items()}}
    report["advance"] = (report["validation_delta"] > 0 and
                         all(report["benchmark_deltas"][s] > 0 and
                             benchmarks[s]["J1-S-R"]["num_value_adding_fires"] >=
                             benchmarks[s]["J1-S"]["num_value_adding_fires"]
                             for s in ("web", "ego")))
    path = OUT / "reports/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    audit = json.loads((OUT / "manifests/audit.json").read_text())
    lines = ["# J1-S-R closed-loop error exposure results", "",
             f"Manifest: 263,760 presentations; 41,220 Response; quota {audit['quota']}; "
             f"selected videos {audit['selected_Silence']['videos']} Silence, "
             f"{audit['selected_Response']['videos']} Response.", "",
             "Frozen video-disjoint MarkII validation (GPT-4.1 judged PAUC-F):", "",
             "| Controller | PAUC-F | Useful fires | Wasted fires |", "| --- | ---: | ---: | ---: |"]
    for name in ("j1s", "j1sr"):
        row = validation[name]
        lines.append(f"| {name.upper()} | {row['pauc_f']:.2f} | {row['value_adding_fires']} | {row['wasted_fires']} |")
    lines += ["", "Full GPT-4.1 `pauc-f-v1`, `omega=0.5` confirmation:", "",
              "| Split | Controller | P | R | PAUC-F | Fires | Useful | Wasted |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("web", "ego"):
        for name in ("J1-S", "J1-S-R"):
            row = benchmarks[split][name]
            lines.append(f"| {split.upper()} | {name} | {row['fire_precision']:.2f}% | "
                         f"{row['pauc_recall_component']:.2f} | {row['pauc_f']:.2f} | "
                         f"{row['num_fires']} | {row['num_value_adding_fires']} | "
                         f"{row['num_wasted_fires']} |")
    lines += ["", f"Validation PAUC-F delta: {report['validation_delta']:+.2f} points."]
    for split in ("web", "ego"):
        ci = diagnostics[split]["paired_video_bootstrap_95_ci"]
        lines.append(f"{split.upper()} delta: {report['benchmark_deltas'][split]:+.2f} points; "
                     f"paired video bootstrap 95% interval [{ci[0]:+.2f}, {ci[1]:+.2f}].")
        delta = diagnostics[split]["delta_j1sr_minus_j1s"]
        lines.append(f"{split.upper()} first-event coverage delta: "
                     f"{delta['first_event_covered']:+d} videos; "
                     f"repeat fires in gold windows delta: "
                     f"{delta['repeat_fires_in_gold_windows']:+d}.")
    lines += ["", f"Advance decision under the prespecified coverage and score rule: "
              f"**{'yes' if report['advance'] else 'no'}**.", "",
              "See `results.json` for paired fire counts and duration buckets, "
              "and `../source/preflight.json` for hashes and video-disjointness checks.", ""]
    (OUT / "reports/results.md").write_text("\n".join(lines))
    print(json.dumps({"validation_delta": report["validation_delta"],
                      "benchmark_deltas": report["benchmark_deltas"],
                      "advance": report["advance"]}, sort_keys=True))


if __name__ == "__main__":
    main()
