#!/usr/bin/env python3
"""One frozen PACE-200 evaluation for all prespecified full-size models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(STREAM_GEMMA)]

from pace_clm.labels import Action
from score_timing_paucf import prediction_records
from train_and_eval_heads import behavioral_metrics, build_model, classification_metrics, logits_for
from zs_infer.markii_paucf_200 import _score_records, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    val = torch.load(args.feature_dir / "pace200.pt", map_location="cpu", weights_only=False)
    variants = torch.load(args.feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    source = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    manifest = read_jsonl(source / "subset/subset_200_manifest.jsonl")
    gold = read_json(source / "gold/paucf_4s_gold.json")["records"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = val["features"].float().to(device)
    targets = val["targets"].long()
    descriptive = val["action_features"].float().to(device)
    short = variants["short_labels"].float().to(device)
    runs = []
    for name in ("linear", "mlp", "clm_128", "clm_256", "clm_short_256"):
        for seed in (42, 43, 44):
            checkpoint = torch.load(
                args.result_dir / f"{name}.n24300.seed{seed}.pt", map_location="cpu", weights_only=False
            )
            action_vectors = short if name == "clm_short_256" else descriptive
            dim = 128 if name == "clm_128" else 256 if name.startswith("clm") else 0
            model = build_model(name, int(val["hidden_size"]), dim).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with torch.no_grad():
                logits = torch.cat([
                    logits_for(model, batch, action_vectors).cpu()
                    for batch in features.split(512)
                ])
            margin = float(checkpoint["run"]["response_margin"])
            logits[:, int(Action.Response)] -= margin
            predictions = logits.argmax(-1)
            records = prediction_records(predictions.tolist(), val["metadata"], manifest)
            paucf, per_video = _score_records(records, gold, None)
            result = {
                "method": name,
                "seed": seed,
                "train_size": 24300,
                "response_margin_from_dev": margin,
                "best_epoch_from_dev": checkpoint["run"]["best_epoch"],
                "classification": classification_metrics(logits, targets),
                "behavior": behavioral_metrics(predictions, val["metadata"]),
                "no_judge_pauc_f": paucf,
            }
            runs.append(result)
            torch.save({"logits": logits, "predictions": predictions,
                        "targets": targets, "metadata": val["metadata"], "run": result},
                       args.result_dir / f"{name}.pace200.seed{seed}.pt")
            (args.result_dir / "pace200_results.json").write_text(
                json.dumps(runs, indent=2, sort_keys=True) + "\n"
            )
            (args.result_dir / f"{name}.pace200.seed{seed}.per_video.json").write_text(
                json.dumps(per_video, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({"method": name, "seed": seed,
                              "macro_f1": result["classification"]["macro_f1"],
                              "pauc_f": paucf["pauc_f"],
                              "fires_per_video": result["behavior"]["fires_per_video"]}), flush=True)


if __name__ == "__main__":
    main()
