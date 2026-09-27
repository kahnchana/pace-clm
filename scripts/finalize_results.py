#!/usr/bin/env python3
"""Merge token baseline results and render the PACE-200 result tables."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from train_and_eval_heads import behavioral_metrics, classification_metrics


def mean_sd(values: list[float], scale: float = 100.0) -> str:
    mean = statistics.mean(values) * scale
    if len(values) == 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {statistics.stdev(values) * scale:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/results/pace200_v1"))
    parser.add_argument("--token-dir", type=Path, default=Path("outputs/results/pace_token_v1"))
    args = parser.parse_args()
    payload = json.loads((args.result_dir / "results.json").read_text())
    token_rows = []
    for path in sorted(args.token_dir.glob("pace_token.rank*.jsonl")):
        token_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if token_rows:
        token_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
        prediction_dir = args.result_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir / "pace_token_generation.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in token_rows)
        )
        token_targets = torch.tensor([row["gold_action_id"] for row in token_rows])
        token_predictions = torch.tensor([row["predicted_action_id"] for row in token_rows])
        token_logits = torch.full((len(token_rows), 3), -20.0)
        token_logits.scatter_(1, token_predictions[:, None], 20.0)
        token_classification = classification_metrics(token_logits, token_targets)
        token_behavior = behavioral_metrics(token_predictions, token_rows)
        token_result = {
            "method": "pace_token_generation",
            "seed": None,
            "classification": token_classification,
            "behavior": token_behavior,
            "prediction_distribution": dict(Counter(row["predicted_action"] for row in token_rows)),
            "malformed": sum(row["malformed"] for row in token_rows),
        }
        (args.result_dir / "pace_token_results.json").write_text(
            json.dumps(token_result, indent=2, sort_keys=True) + "\n"
        )
    else:
        token_result = None

    timing_path = args.result_dir / "timing_paucf" / "summary.json"
    timing_paucf = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    grouped = {}
    for run in payload["runs"]:
        grouped.setdefault(run["method"], []).append(run)
    rows = []
    if token_result:
        rows.append(("PACE token generation", [token_result]))
    labels = {
        "raw_cosine": "Raw cosine",
        "linear": "Linear classifier",
        "mlp": "Capacity-matched MLP",
        "clm_128": "CLM, 128-d",
        "clm_256": "CLM, 256-d",
        "clm_512": "CLM, 512-d",
        "clm_shared_512": "CLM, shared head",
        "clm_short_512": "CLM, short labels",
    }
    rows.extend((labels[name], runs) for name, runs in grouped.items())
    header = (
        "| Method | Accuracy | Macro-F1 | Response F1 | Hold F1 | Silence F1 | "
        "Timing F1@4s | Timing PAUC-F | Fires/video | Repeat rate | Latency (s) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = []
    for label, runs in rows:
        accuracy = [float(run["classification"]["accuracy"]) for run in runs]
        macro_f1 = [float(run["classification"]["macro_f1"]) for run in runs]
        response_f1 = [float(run["classification"]["per_class"]["Response"]["f1"]) for run in runs]
        hold_f1 = [float(run["classification"]["per_class"]["Hold"]["f1"]) for run in runs]
        silence_f1 = [float(run["classification"]["per_class"]["Silence"]["f1"]) for run in runs]
        timing_f1 = [float(run["behavior"]["timing_f1_4s"]) for run in runs]
        fires_per_video = [float(run["behavior"]["fires_per_video"]) for run in runs]
        repeat_rate = [float(run["behavior"]["repeat_rate"]) for run in runs]
        latencies = [run["behavior"]["mean_latency_seconds"] for run in runs]
        latencies = [float(value) for value in latencies if value is not None]
        method = str(runs[0]["method"])
        pauc_f = timing_paucf.get(method, {}).get("pauc_f")
        body.append(
            "| "
            + " | ".join(
                [
                    label,
                    mean_sd(accuracy),
                    mean_sd(macro_f1),
                    mean_sd(response_f1),
                    mean_sd(hold_f1),
                    mean_sd(silence_f1),
                    mean_sd(timing_f1),
                    f"{float(pauc_f):.2f}" if pauc_f is not None else "—",
                    mean_sd(fires_per_video, scale=1.0),
                    mean_sd(repeat_rate),
                    mean_sd(latencies, scale=1.0) if latencies else "—",
                ]
            )
            + " |"
        )
    notes = f"""

Tiny-set overfit gate: **{payload["tiny_overfit"]["accuracy"] * 100:.1f}%** on {payload["tiny_overfit"]["examples"]} examples.

Train distribution: `{payload["train_distribution"]}`. PACE-200 frame-state distribution: `{payload["validation_distribution"]}`.

Classifier rows show mean ± sample standard deviation over three seeds. Raw cosine and PACE token generation are deterministic single runs. Timing PAUC-F uses seed 42 for trained methods and the unchanged Stream-Gemma scorer. Inputs use the same strictly causal 32-frame window and common gold prior-action history; this table therefore isolates the decision rule rather than compounding free-running history drift.
"""
    report = "# PACE-200 CLM pilot results\n\n" + header + "\n" + "\n".join(body) + notes
    (args.result_dir / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
