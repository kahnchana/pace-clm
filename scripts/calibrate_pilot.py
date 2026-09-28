#!/usr/bin/env python3
"""Diagnostic equal-budget margin sweep over existing PACE-200 pilot logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(STREAM_GEMMA)]

from train_and_eval_heads import behavioral_metrics, classification_metrics
from score_timing_paucf import prediction_records
from zs_infer.markii_paucf_200 import _score_records, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "outputs/results/pace200_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    manifest = read_jsonl(source / "subset/subset_200_manifest.jsonl")
    gold = read_json(source / "gold/paucf_4s_gold.json")["records"]
    results = {}
    for method in ("linear", "mlp", "clm_128", "clm_256", "clm_short_512"):
        artifact = torch.load(args.pilot_dir / f"{method}.seed42.pt", map_location="cpu", weights_only=False)
        logits = artifact["logits"].float()
        targets = artifact["targets"].long()
        metadata = artifact["metadata"]
        margin = logits[:, 0] - logits[:, 1:].max(-1).values
        thresholds = sorted({float(x) for x in torch.quantile(margin, torch.linspace(0, 1, 41))} | {0.0})
        sweep = []
        for threshold in thresholds:
            adjusted = logits.clone()
            adjusted[:, 0] -= threshold
            predictions = adjusted.argmax(-1)
            records = prediction_records(predictions.tolist(), metadata, manifest)
            paucf, _ = _score_records(records, gold, None)
            sweep.append({
                "response_margin": threshold,
                "macro_f1": classification_metrics(adjusted, targets)["macro_f1"],
                "fires_per_video": behavioral_metrics(predictions, metadata)["fires_per_video"],
                "no_judge_pauc_f": paucf,
            })
        results[method] = sweep
        print(f"{method}: {len(sweep)} thresholds", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
