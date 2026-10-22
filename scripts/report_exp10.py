#!/usr/bin/env python3
"""Write the factual ten-run campaign report from completed judged artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10"
RUNS = [f"E{i:02d}" for i in range(1, 11)]
IDEAS = ["Missed-window contrastive mining", "PACE decision distillation",
         "Generated-history replay", "Causal temporal head", "Answer-aware utility"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selection(name: str) -> dict:
    path = OUT / "selection" / name / "official/results/exp10_judged_summary.json"
    score = json.loads(path.read_text())
    if score["num_sampled_videos"] != 200 or not score.get("headline"):
        raise RuntimeError(f"incomplete judged selection: {path}")
    return score


def diagnostics(name: str) -> dict:
    predictions = read_jsonl(OUT / "selection" / name / "official/generation/predictions.jsonl")
    duplicate = 0
    wall = []
    head = []
    for record in predictions:
        answers = [str(fire["content"]) for fire in record["model_response_list"]]
        duplicate += len(answers) - len(set(answers))
        frames = int(record["exp10_frame_count"])
        wall.append(float(record["exp10_video_wall_seconds"]) / frames)
        if "exp10_head_seconds" in record:
            head.append(1000 * float(record["exp10_head_seconds"]) / frames)
    return {"duplicate_response_texts": duplicate,
            "median_end_to_end_seconds_per_frame": statistics.median(wall),
            "median_head_milliseconds_per_frame": statistics.median(head) if head else None}


def main() -> None:
    reference = {name: selection(name) for name in ("run5", "PACE")}
    scored = {run: selection(run) for run in RUNS}
    decision = json.loads((ROOT / "manifests/exp10/selection_decision.json").read_text())
    confirmation = json.loads((ROOT / "manifests/exp10/confirmation_decision.json").read_text())
    paired_path = OUT / "full/paired_intervals.json"
    paired = json.loads(paired_path.read_text()) if paired_path.exists() else {}
    results = {}
    table = []
    detail_table = []
    for i, run in enumerate(RUNS):
        checkpoint = torch.load(OUT / run / "checkpoint.pt", map_location="cpu", weights_only=False)
        metrics = scored[run]
        diagnostic = diagnostics(run)
        if not (OUT / run / "completion.json").is_file():
            raise RuntimeError(f"missing completed judged run marker: {run}")
        item = {"idea": IDEAS[i // 2], "method": "CLM" if i % 2 == 0 else "MLP",
                "selection": {"judged_pauc_f": metrics["pauc_f"],
                              "recall_component": metrics["pauc_recall_component"],
                              "judged_fire_precision": metrics["fire_precision"],
                              "fires_per_video": metrics["fires_per_video"],
                              "wasted_fires": metrics["wasted_fires"],
                              "value_adding_fires": metrics["value_adding_fires"]},
                "dev_per_state_f1": {key: value["f1"] for key, value in
                                     checkpoint["run"]["dev_calibrated"]["per_class"].items()},
                "parameters": checkpoint["run"]["parameters"], "latency": diagnostic,
                "selection_gate_pass": decision["candidate_gates"][run]["passes"]}
        if run in confirmation["candidate_pauc_f"]:
            item["confirmation_judged_pauc_f"] = confirmation["candidate_pauc_f"][run]
        results[run] = item
        row = (f"| {run} | {item['method']} | {item['idea']} | {metrics['pauc_f']:.3f} | "
               f"{metrics['pauc_recall_component']:.3f} | {metrics['fire_precision']:.3f} | "
               f"{metrics['fires_per_video']:.2f} | {metrics['wasted_fires']} | "
               f"{diagnostic['duplicate_response_texts']} | "
               f"{diagnostic['median_end_to_end_seconds_per_frame']:.3f} |")
        table.append(row)
        f1 = item["dev_per_state_f1"]
        head_ms = diagnostic["median_head_milliseconds_per_frame"]
        detail_table.append(f"| {run} | {item['parameters']:,} | {f1['Response']:.3f} | "
                            f"{f1['Silence']:.3f} | {f1['Hold']:.3f} | {head_ms:.3f} |")
    full = {}
    old = {"web": {"run5": 53.498394500663316, "PACE": 57.47497434486915},
           "ego": {"run5": 43.95297997027136, "PACE": 46.05065873842904}}
    for benchmark in ("web", "ego"):
        full[benchmark] = {"recorded_baselines": old[benchmark], "finalists": {}}
        for run in confirmation["advanced_to_web_ego"]:
            path = OUT / "full" / benchmark / run / "pauc_f_summary-gpt-4.1-v1.json"
            score = json.loads(path.read_text())
            expected = 500 if benchmark == "web" else 326
            if score["num_videos"] != expected or not score["headline_eligible"]:
                raise RuntimeError(f"incomplete judged full split: {path}")
            full[benchmark]["finalists"][run] = score
    summary = {"selection_references": {name: value["pauc_f"] for name, value in reference.items()},
               "runs": results, "pair_clm_minus_mlp_points": decision["pair_clm_minus_mlp_points"],
               "selection_finalists": decision["finalists"],
               "confirmation_advanced": confirmation["advanced_to_web_ego"],
               "full_splits": full, "paired_intervals": paired,
               "scope_note": "One seed per run; WEB/EGO were inspected during earlier model selection."}
    report_json = OUT / "report.json"
    report_md = ROOT / "agent_notes/experiments/pace_clm_exp10.md"
    intro = ["# Ten-run PACE-CLM experiment", "",
             "Status: all ten trained checkpoints and judged selection evaluations complete.", "",
             "The fresh selection cohort contains 200 videos. These are judged GPT-4.1 PAUC-F scores; the 5-video portal smoke score is excluded.", "",
             f"Selection references: run 5 MLP {reference['run5']['pauc_f']:.3f}; PACE token {reference['PACE']['pauc_f']:.3f}.", "",
             "| Run | Head | Intervention | Judged PAUC-F | Recall | Fire precision | Fires/video | Wasted fires | Duplicate texts | Seconds/frame |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|", *table, "",
             "Internal dev F1 and selection head inference latency:", "",
             "| Run | Parameters | Response F1 | Silence F1 | Hold F1 | Head ms/frame |",
             "|---|---:|---:|---:|---:|---:|", *detail_table, "",
             "## Matched pairs", ""]
    for pair, difference in decision["pair_clm_minus_mlp_points"].items():
        intro.append(f"- {pair}: CLM minus MLP {difference:+.3f} PAUC-F points.")
    best = max(RUNS, key=lambda run: scored[run]["pauc_f"])
    beat_run5 = [run for run in RUNS if scored[run]["pauc_f"] > reference["run5"]["pauc_f"]]
    intro += ["", "## Findings", "",
              (f"- {best} had the highest selection judged PAUC-F ({scored[best]['pauc_f']:.3f}), "
               f"{scored[best]['pauc_f'] - reference['run5']['pauc_f']:+.3f} points versus run 5."),
              (f"- {len(beat_run5)} of ten runs exceeded run 5 on selection judged PAUC-F; "
               f"{len(decision['finalists'])} passed the joint score and precision gate.")]
    for odd in range(1, 11, 2):
        clm, mlp = f"E{odd:02d}", f"E{odd+1:02d}"
        intro.append(f"- {IDEAS[(odd-1)//2]}: CLM versus MLP changed judged recall by "
                     f"{scored[clm]['pauc_recall_component'] - scored[mlp]['pauc_recall_component']:+.3f} "
                     f"and fire precision by {scored[clm]['fire_precision'] - scored[mlp]['fire_precision']:+.3f} points.")
    intro += ["", "## Promotion", "",
              f"Selection finalists: {', '.join(decision['finalists']) or 'none'}.",
              f"Confirmation advances: {', '.join(confirmation['advanced_to_web_ego']) or 'none'}.", ""]
    for benchmark in ("web", "ego"):
        intro += [f"### {benchmark.upper()}", "",
                  f"Recorded PACE {old[benchmark]['PACE']:.3f}; recorded run 5 MLP {old[benchmark]['run5']:.3f}.", ""]
        for run, score in full[benchmark]["finalists"].items():
            intro.append(f"- {run}: judged PAUC-F {score['pauc_f']:.3f}; recall {score['pauc_recall_component']:.3f}; "
                         f"fire precision {score['fire_precision']:.3f}; useful/wasted fires "
                         f"{score['num_value_adding_fires']}/{score['num_wasted_fires']}.")
            intervals = paired.get(benchmark, {}).get("paired_intervals", {}).get(run, {})
            for baseline, value in intervals.items():
                low, high = value["paired_95_percent_ci"]
                intro.append(f"  - Paired difference versus {baseline}: {value['difference_points']:+.3f} "
                             f"[95% CI {low:+.3f}, {high:+.3f}].")
        intro.append("")
    successful = [run for run in confirmation["advanced_to_web_ego"]
                  if full["web"]["finalists"][run]["pauc_f"] >= 57.475 and
                     full["ego"]["finalists"][run]["pauc_f"] >= 46.051]
    parity = [run for run in confirmation["advanced_to_web_ego"]
              if full["web"]["finalists"][run]["pauc_f"] >= 56.475 and
                 full["ego"]["finalists"][run]["pauc_f"] >= 45.051 and
                 results[run]["latency"]["median_end_to_end_seconds_per_frame"] <
                 diagnostics("PACE")["median_end_to_end_seconds_per_frame"]]
    intro += ["## Outcome", "",
              f"- Met the preregistered full-split success threshold: {', '.join(successful) or 'none'}.",
              ("- Within one point on both full splits with faster selection serving latency: "
               f"{', '.join(parity) or 'none'}."), ""]
    intro += ["## Limits", "", ("One seed per run supports screening, not a repeatability claim. "
              "WEB/EGO were used in earlier model selection; confirmation is the untouched within-project test."), ""]
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown = "\n".join(intro)
    for path, content in ((report_json, rendered), (report_md, markdown)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"frozen report differs: {path}")
        path.write_text(content)
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
