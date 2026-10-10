#!/usr/bin/env python3
"""Select a simple MLP firing gate on development videos and score a frozen test."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG)]

from pace_clm.labels import Action
from score_timing_paucf import prediction_records
from train_and_eval_heads import behavioral_metrics, build_model, classification_metrics, logits_for
from zs_infer.markii_paucf_200 import _score_records, read_jsonl


def gated(logits: torch.Tensor, metadata: list[dict], margin: float, cooldown: int) -> torch.Tensor:
    adjusted = logits.clone()
    adjusted[:, int(Action.Response)] -= margin
    result = adjusted.argmax(-1)
    if cooldown:
        last: dict[str, int] = {}
        for index, row in enumerate(metadata):
            if int(result[index]) != int(Action.Response):
                continue
            video = row["video_id"]
            frame = int(row["frame_index"])
            if video in last and frame - last[video] <= cooldown:
                result[index] = adjusted[index, 1:].argmax() + 1
            else:
                last[video] = frame
    return result


def evaluate(predictions: torch.Tensor, targets: torch.Tensor, metadata: list[dict],
             manifest: list[dict], gold: list[dict]) -> dict:
    records = prediction_records(predictions.tolist(), metadata, manifest)
    paucf, _ = _score_records(records, gold, None)
    logits = torch.full((len(predictions), 3), -20.0)
    logits.scatter_(1, predictions[:, None], 20.0)
    cls = classification_metrics(logits, targets)
    return {"pauc_f": paucf["pauc_f"], "pauc_recall": paucf["pauc_recall_component"],
            "fire_precision": paucf["fire_precision"],
            "fires_per_video": behavioral_metrics(predictions, metadata)["fires_per_video"],
            "repeat_fires": behavioral_metrics(predictions, metadata)["repeat_fires_on_hold"],
            "macro_f1": cls["macro_f1"],
            "hold_recall": cls["per_class"]["Hold"]["recall"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "test"), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--old-result-dir", type=Path, required=True)
    parser.add_argument("--mixture-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--early-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hold2-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = "selection" if args.phase == "select" else "final"
    val = torch.load(args.feature_dir / f"{split}.pt", map_location="cpu", weights_only=False)
    manifest = read_jsonl(args.validation_dir / f"{split}.jsonl")
    gold = json.loads((args.validation_dir / f"{split}.gold.json").read_text())["records"]
    targets = val["targets"].long()
    features = val["features"].float()
    vectors = val["action_features"].float()
    logits_by_model = {}
    margins = {}
    for condition in ("control", "run5", "run5_early", "run5_hold2"):
        for seed in (42, 43, 44):
            path = (args.old_result_dir / f"mlp.n24300.seed{seed}.pt" if condition == "control"
                    else args.early_checkpoint_dir / f"run5_early.mlp.seed{seed}.pt" if condition == "run5_early"
                    else args.hold2_checkpoint_dir / f"run5_hold2.mlp.seed{seed}.pt" if condition == "run5_hold2"
                    else args.mixture_checkpoint_dir / f"run5.mlp.seed{seed}.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model("mlp", int(val["hidden_size"]), 0)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with torch.no_grad():
                logits_by_model[(condition, seed)] = torch.cat(
                    [logits_for(model, batch, vectors).cpu() for batch in features.split(512)])
            margins[(condition, seed)] = float(checkpoint["run"]["response_margin"])
    if args.phase == "test":
        config = json.loads((args.output_dir / "selected_gate.json").read_text())
        candidates = [{"condition": config["condition"], "margin_offset": config["margin_offset"],
                       "cooldown": config["cooldown"]},
                      {"condition": "control", "margin_offset": 0.0, "cooldown": 0},
                      {"condition": "run5", "margin_offset": 0.0, "cooldown": 0},
                      {"condition": "run5_early", "margin_offset": 0.0, "cooldown": 0},
                      {"condition": "run5_hold2", "margin_offset": 0.0, "cooldown": 0}]
    else:
        candidates = [{"condition": condition, "margin_offset": offset, "cooldown": cooldown}
                      for condition in ("control", "run5", "run5_early", "run5_hold2")
                      for offset in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
                      for cooldown in (0, 2, 4, 8)]
    all_results = []
    for config in candidates:
        scores = []
        for seed in (42, 43, 44):
            key = (config["condition"], seed)
            pred = gated(logits_by_model[key], val["metadata"],
                         margins[key] + config["margin_offset"], config["cooldown"])
            scores.append(evaluate(pred, targets, val["metadata"], manifest, gold))
        all_results.append({**config, "seeds": scores,
                            "mean": {metric: statistics.mean(s[metric] for s in scores)
                                     for metric in scores[0]}})
    (args.output_dir / f"gate_{args.phase}_results.json").write_text(
        json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    if args.phase == "select":
        baseline = next(r for r in all_results if r["condition"] == "control"
                        and r["margin_offset"] == 0 and r["cooldown"] == 0)
        eligible = [r for r in all_results
                    if r["mean"]["fires_per_video"] <= baseline["mean"]["fires_per_video"] + 0.05
                    and r["mean"]["repeat_fires"] <= baseline["mean"]["repeat_fires"] + 2
                    and r["mean"]["hold_recall"] >= baseline["mean"]["hold_recall"] - 0.02]
        winner = max(eligible, key=lambda r: (r["mean"]["pauc_f"], r["mean"]["macro_f1"],
                                              -r["mean"]["fires_per_video"]))
        selected = {k: winner[k] for k in ("condition", "margin_offset", "cooldown", "mean")}
        selected["baseline_mean"] = baseline["mean"]
        selected["eligible_candidates"] = len(eligible)
        (args.output_dir / "selected_gate.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        print(json.dumps(selected, indent=2))
    else:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
