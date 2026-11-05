#!/usr/bin/env python3
"""Compare judged WEB results for official dense Qwen3.5 J1-S controllers."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from report_qwen35_dense_j1s_web import paired_bootstrap


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SIZES = ("0p8b", "2b", "4b")
LABELS = {"0p8b": "0.8B", "2b": "2B", "4b": "4B"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    output = root / "outputs/jevany-qwen35-size-sweep-j1s/reports"
    results = {}
    prediction_rows = {}
    ids = None
    for size in SIZES:
        base = root / f"outputs/jevany-qwen35-{size}-j1s"
        result = json.loads((base / "reports/web_result.json").read_text())
        predictions = [json.loads(line) for line in
                       (base / "evaluation/benchmarks/web/predictions.jsonl").read_text().splitlines()]
        current_ids = {row["question_id"] for row in predictions}
        if len(predictions) != 500 or len(current_ids) != 500 or result["score"]["num_videos"] != 500:
            raise ValueError(f"incomplete WEB result for {size}")
        if ids is None:
            ids = current_ids
        elif current_ids != ids:
            raise ValueError("controller runs used different WEB question IDs")
        if results and result["generator_adapter_sha256"] != next(iter(results.values()))["generator_adapter_sha256"]:
            raise ValueError("controller runs used different generators")
        if results and result["manifest_sha256"] != next(iter(results.values()))["manifest_sha256"]:
            raise ValueError("controller runs used different training manifests")
        results[size] = result
        prediction_rows[size] = predictions
    baseline_root = root / "outputs/jevany-qwen35-2b-j1s"
    paired_vs_2b = {}
    for size in ("0p8b", "4b"):
        comparison = paired_bootstrap(
            root / f"outputs/jevany-qwen35-{size}-j1s",
            prediction_rows[size], baseline_root=baseline_root,
        )
        if comparison is None:
            raise ValueError("stored 2B judged predictions are required for paired comparison")
        paired_vs_2b[size] = comparison
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "sizes": {size: {
            "model_name": item["controller_model"]["model_name"],
            "model_revision": item["controller_model"]["model_revision"],
            "checkpoint": item["checkpoint"],
            "score": item["score"],
            "fires_per_video": item["fires_per_video"],
            "decision_seconds_per_decision": item["decision_seconds_per_decision"],
            "generation_seconds_per_fire": item["generation_seconds_per_fire"],
            "eval_peak_gpu_memory_gib": item["eval_peak_gpu_memory_gib"],
            "train_wall_seconds": item.get("train_wall_seconds_including_resumes"),
        } for size, item in results.items()},
        "web_question_ids": 500,
        "generator_adapter_sha256": results["2b"]["generator_adapter_sha256"],
        "training_manifest_sha256": results["2b"]["manifest_sha256"],
        "paired_bootstrap_vs_2b": paired_vs_2b,
    }
    (output / "web_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Qwen3.5 dense J1-S controller size sweep: WEB", "",
        "All three controllers trained on the same 263,760 J1-S presentations, seed 42, one epoch, with the same LoRA, pointer head, input window, and global batch 32. Each uses the same frozen promoted Qwen3.5-35B-A3B generator for answers. All 500 WEB IDs were judged with GPT-4.1 and `pauc-f-v1` at omega 0.5.", "",
        "| Controller | PAUC | PAUC-F | Precision | Fires | Useful | Wasted | Fires/video | Decision s/decision | Train h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, item in results.items():
        score = item["score"]
        lines.append(
            f"| {LABELS[size]} | {score['pauc_recall_component']:.2f} | {score['pauc_f']:.2f} | "
            f"{score['fire_precision']:.2f}% | {score['num_fires']} | "
            f"{score['num_value_adding_fires']} | {score['num_wasted_fires']} | "
            f"{item['fires_per_video']:.3f} | {item['decision_seconds_per_decision']:.3f} | "
            f"{(item.get('train_wall_seconds_including_resumes') or item['train_seconds']) / 3600:.2f} |"
        )
    lines += ["", "Historical MoE J1-S scored 56.51 PAUC-F and promoted PACE scored 57.47 on WEB. The dense controllers start from official post-trained weights; historical MoE J1-S started from a PACE-trained adapter, so differences from it cannot be attributed solely to model size.", ""]
    for size in ("0p8b", "4b"):
        delta = results[size]["score"]["pauc_f"] - results["2b"]["score"]["pauc_f"]
        lower, upper = paired_vs_2b[size]["delta_ci95"]
        lines.append(f"{LABELS[size]} versus 2B: {delta:+.2f} PAUC-F points; paired 1,000-replicate 95% interval [{lower:+.2f}, {upper:+.2f}].")
    lines += ["", "Individual reports and judged summaries:", ""]
    for size in SIZES:
        base = root / f"outputs/jevany-qwen35-{size}-j1s"
        lines.append(f"- {LABELS[size]}: `{base / 'reports/web_result.md'}`; `{base / 'evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json'}`")
    lines.append("")
    (output / "web_comparison.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(output / 'web_comparison.md'),
                      "pauc_f": {size: results[size]["score"]["pauc_f"] for size in SIZES}}))


if __name__ == "__main__":
    main()
