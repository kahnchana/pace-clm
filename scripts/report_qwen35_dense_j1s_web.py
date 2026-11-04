#!/usr/bin/env python3
"""Write a completed dense Qwen3.5 J1-S WEB result and summary."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def paired_bootstrap(root: Path, new_predictions: list[dict],
                     baseline_root: Path | None = None) -> dict | None:
    baseline_dir = (baseline_root or root.parent / "jevany-j1s") / "evaluation/benchmarks/web"
    old_predictions_path = baseline_dir / "predictions.jsonl"
    old_scores_path = baseline_dir / "pauc_llm_scores-gpt-4.1.json"
    new_scores_path = root / "evaluation/benchmarks/web/pauc_llm_scores-gpt-4.1.json"
    if not all(path.is_file() for path in (old_predictions_path, old_scores_path, new_scores_path)):
        return None
    from zs_infer.pauc_f import OFFICIAL_SCORER, _load_official_scorer, assign_value_adding_fires, fire_times
    official = _load_official_scorer(OFFICIAL_SCORER)

    def components(predictions: list[dict], scores_path: Path) -> dict[str, tuple[int, int, float, int]]:
        scored = {row["question_id"]: row for row in json.loads(scores_path.read_text())}
        values = {}
        for row in predictions:
            question_id = row["question_id"]
            turns = scored[question_id]["answer"]
            fires = fire_times(row)
            useful = len(assign_value_adding_fires(fires, turns))
            auc_total = 0.0
            for turn in turns:
                judged = turn.get("judge_scores") or {}
                if judged:
                    points = [[float(time) - turn["reply_timespan"][0], value]
                              for time, value in judged.items()]
                    auc_total += official.area_under_line_ratio(
                        points, turn["reply_timespan"][1] - turn["reply_timespan"][0],
                        2, 0.5, 0.5)
                else:
                    auc_total += 0.25
            values[question_id] = (len(fires), useful, auc_total, len(turns))
        return values

    old_predictions = [json.loads(line) for line in old_predictions_path.read_text().splitlines()]
    old = components(old_predictions, old_scores_path)
    new = components(new_predictions, new_scores_path)
    ids = sorted(new)
    if len(ids) != 500 or set(ids) != set(old):
        raise ValueError("paired bootstrap requires the same 500 WEB IDs")

    def metric(values: dict, sample: list[str]) -> float:
        fires = useful = turns = 0
        auc = 0.0
        for question_id in sample:
            item = values[question_id]
            fires += item[0]; useful += item[1]; auc += item[2]; turns += item[3]
        precision = useful / fires if fires else 0.0
        recall = auc / turns
        return 200 * precision * recall / (precision + recall) if precision + recall else 0.0

    randomizer = random.Random(42)
    # Draw one shared sample per replicate for a paired global comparison.
    deltas = []
    for _ in range(1000):
        sample = [ids[randomizer.randrange(500)] for _ in range(500)]
        deltas.append(metric(new, sample) - metric(old, sample))
    deltas.sort()
    return {"replicates": 1000, "seed": 42, "delta_ci95": [deltas[25], deltas[974]],
            "baseline_recomputed_pauc_f": metric(old, ids),
            "new_recomputed_pauc_f": metric(new, ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    train = root / "full-seed42"
    web = root / "evaluation/benchmarks/web"
    report = root / "reports/web_result.md"
    score = json.loads((web / "pauc_f_summary-gpt-4.1-v1.json").read_text())
    protocol = json.loads((web / "manifest.json").read_text())
    model_name = protocol["controller_model"]["model_name"]
    size = model_name.rsplit("-", 1)[-1]
    predictions = [json.loads(line) for line in (web / "predictions.jsonl").read_text().splitlines()]
    if len(predictions) != 500 or len({row["question_id"] for row in predictions}) != 500:
        raise ValueError("WEB predictions must contain 500 unique IDs")
    if score["num_videos"] != 500 or not score["headline_eligible"]:
        raise ValueError("judged WEB score is incomplete")
    latest = json.loads((train / "latest.json").read_text())
    if latest["step"] != 8243:
        raise ValueError("WEB must use the final one-epoch checkpoint")
    train_summaries = [json.loads((train / f"summary.rank{rank}.json").read_text()) for rank in range(8)]
    started_path, completed_path = train / "started_epoch.txt", train / "completed_epoch.txt"
    train_wall_seconds = (int(completed_path.read_text()) - int(started_path.read_text())
                          if started_path.is_file() and completed_path.is_file() else None)
    decisions = sum(sum(row["jevany_actions"].values()) for row in predictions)
    decision_seconds = sum(row["jevany_decision_seconds"] for row in predictions)
    generation_seconds = sum(row["jevany_generation_seconds"] for row in predictions)
    result = {
        "checkpoint": latest["checkpoint"],
        "controller_model": protocol["controller_model"],
        "controller_size": size,
        "generator_adapter_sha256": protocol["original_generator_adapter_sha256"],
        "manifest_sha256": "89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222",
        "train_global_presentations": 263760,
        "train_optimizer_steps": 8243,
        "train_seconds": max(row["seconds"] for row in train_summaries),
        "train_wall_seconds_including_resumes": train_wall_seconds,
        "train_peak_gpu_memory_gib": max(row["peak_gpu_memory_gib"] for row in train_summaries),
        "eval_peak_gpu_memory_gib": max(row["jevany_peak_gpu_memory_gib"] for row in predictions),
        "web_videos": 500,
        "web_decisions": decisions,
        "decision_seconds_total": decision_seconds,
        "decision_seconds_per_decision": decision_seconds / decisions,
        "generation_seconds_total": generation_seconds,
        "generation_seconds_per_fire": generation_seconds / score["num_fires"] if score["num_fires"] else None,
        "fires_per_video": score["num_fires"] / 500,
        "score": score,
        "delta_pauc_f_vs_j1s": score["pauc_f"] - 56.51082070394108,
        "delta_pauc_f_vs_promoted_pace": score["pauc_f"] - 57.47,
        "delta_pauc_f_vs_2b": score["pauc_f"] - 54.665128505970486,
        "reaches_j1s_56_51": score["pauc_f"] >= 56.51,
    }
    result["paired_bootstrap"] = paired_bootstrap(root, predictions)
    report.parent.mkdir(parents=True, exist_ok=True)
    (report.parent / "web_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        f"# Qwen3.5-{size} J1-S WEB result", "",
        f"Final checkpoint: `{latest['checkpoint']}` (8,243 updates, seed 42).", "",
        f"The controller starts from the official post-trained Qwen3.5-{size} checkpoint; historical MoE J1-S starts from a PACE-trained adapter. The generator is the same frozen promoted 35B MoE.", "",
        f"| Metric | {size} controller | Historical J1-S MoE |", "|---|---:|---:|",
        f"| Judged PAUC-F | {score['pauc_f']:.2f} | 56.51 |",
        f"| Fire precision | {score['fire_precision']:.2f}% | 65.36% |",
        f"| PAUC recall | {score['pauc_recall_component']:.2f} | 49.77 |",
        f"| Total fires | {score['num_fires']} | 1,357 |",
        f"| Useful fires | {score['num_value_adding_fires']} | 887 |",
        f"| Wasted fires | {score['num_wasted_fires']} | 470 |", "",
        f"PAUC-F delta versus 2B: {result['delta_pauc_f_vs_2b']:+.2f}; versus J1-S: {result['delta_pauc_f_vs_j1s']:+.2f}; versus promoted PACE (57.47): {result['delta_pauc_f_vs_promoted_pace']:+.2f}. Reaches 56.51: **{'yes' if result['reaches_j1s_56_51'] else 'no'}**.", "",
        f"Fires/video: {result['fires_per_video']:.3f}. Decision latency: {result['decision_seconds_per_decision']:.3f} s/decision; generator latency: {result['generation_seconds_per_fire']:.3f} s/fire.", "",
        f"Peak training GPU memory: {result['train_peak_gpu_memory_gib']:.2f} GiB; peak evaluation GPU memory: {result['eval_peak_gpu_memory_gib']:.2f} GiB. Training wall duration: {(train_wall_seconds or result['train_seconds'])/3600:.2f} h.", "",
        "Evaluation: all 500 unique WEB videos, response margin 0, 32 causal frames, 96-token greedy promoted MoE answers; GPT-4.1, `pauc-f-v1`, omega 0.5.", "",
        f"Predictions: `{web / 'predictions.jsonl'}`. Judged summary: `{web / 'pauc_f_summary-gpt-4.1-v1.json'}`.", "",
    ]
    if result["paired_bootstrap"]:
        lower, upper = result["paired_bootstrap"]["delta_ci95"]
        lines += [f"Paired 1,000-replicate bootstrap delta versus J1-S, global PAUC-F: 95% interval [{lower:+.2f}, {upper:+.2f}] points.", ""]
    report.write_text("\n".join(lines))
    print(json.dumps({"report": str(report), "pauc_f": score["pauc_f"]}))


if __name__ == "__main__":
    main()
