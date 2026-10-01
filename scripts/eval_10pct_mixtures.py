#!/usr/bin/env python3
"""Evaluate frozen mixture heads on a fresh Markii validation cohort."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG)]
from pace_clm.labels import Action
from score_timing_paucf import prediction_records
from train_and_eval_heads import behavioral_metrics, build_model, classification_metrics, logits_for
from zs_infer.markii_paucf_200 import _score_records, read_jsonl


def breakdown(logits: torch.Tensor, targets: torch.Tensor, metadata: list[dict]) -> dict:
    by_video = defaultdict(list)
    for i, row in enumerate(metadata):
        by_video[row["video_id"]].append((int(row["frame_index"]), i, int(targets[i])))
    groups = defaultdict(list)
    for rows in by_video.values():
        rows.sort()
        response = [frame for frame, _, target in rows if target == int(Action.Response)]
        previous_response = None
        hold_start = None
        length = len(rows)
        for frame, i, target in rows:
            groups[f"video_length/{'short' if length < 32 else 'medium' if length < 96 else 'long'}"].append(i)
            nearest = min((abs(frame - r) for r in response), default=9999)
            groups[f"response_distance/{'0-4' if nearest <= 4 else '5-16' if nearest <= 16 else '17+'}"].append(i)
            if previous_response is not None:
                offset = frame - previous_response
                groups[f"post_response/{'1-4' if offset <= 4 else '5-16' if offset <= 16 else '17+'}"].append(i)
            if target == int(Action.Hold):
                if hold_start is None:
                    hold_start = frame
                offset = frame - hold_start
                groups[f"hold_run/{'0-2' if offset < 3 else '3-7' if offset < 8 else '8+'}"].append(i)
            else:
                hold_start = None
            if target == int(Action.Response):
                previous_response = frame
    return {key: {"n": len(ids), **classification_metrics(logits[ids], targets[ids])}
            for key, ids in sorted(groups.items()) if ids}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("selection", "final"), required=True)
    p.add_argument("--conditions", nargs="+", choices=("control", "run1", "run2", "run3", "run4", "run5", "selected", "balanced"), required=True)
    p.add_argument("--scaleup", action="store_true")
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--validation-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--old-result-dir", type=Path, required=True)
    p.add_argument("--old-feature-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.scaleup and (a.set != "final" or set(a.conditions) != {"selected", "balanced"}):
        raise ValueError("scale-up evaluation uses only selected/balanced on the untouched final set")
    if not a.scaleup and set(a.conditions) & {"selected", "balanced"}:
        raise ValueError("selected/balanced regimes require --scaleup")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    val = torch.load(a.feature_dir / f"{a.set}.pt", map_location="cpu", weights_only=False)
    variants = torch.load(a.old_feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    manifest = read_jsonl(a.validation_dir / f"{a.set}.jsonl")
    gold = json.loads((a.validation_dir / f"{a.set}.gold.json").read_text())["records"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = val["features"].float().to(device)
    targets = val["targets"].long()
    descriptive = val["action_features"].float().to(device)
    short = variants["short_labels"].float().to(device)
    runs = []
    for condition in a.conditions:
        for method in ("mlp", "clm_short_256"):
            for seed in (42, 43, 44):
                path = (a.old_result_dir / f"{method}.n24300.seed{seed}.pt" if condition == "control"
                        else a.checkpoint_dir / f"{condition}.{method}.seed{seed}.pt")
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model = build_model(method, int(val["hidden_size"]), 256 if method == "clm_short_256" else 0).to(device)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                vectors = short if method == "clm_short_256" else descriptive
                with torch.no_grad():
                    logits = torch.cat([logits_for(model, batch, vectors).cpu() for batch in features.split(512)])
                margin = float(checkpoint["run"]["response_margin"])
                logits[:, int(Action.Response)] -= margin
                predictions = logits.argmax(-1)
                pred_records = prediction_records(predictions.tolist(), val["metadata"], manifest)
                paucf, per_video = _score_records(pred_records, gold, None)
                result = {"condition": condition, "method": method, "seed": seed,
                          "checkpoint": str(path), "response_margin_from_dev": margin,
                          "best_epoch_from_dev": checkpoint["run"].get("best_epoch"),
                          "classification": classification_metrics(logits, targets),
                          "behavior": behavioral_metrics(predictions, val["metadata"]),
                          "no_judge_pauc_f": paucf,
                          "breakdown": breakdown(logits, targets, val["metadata"])}
                runs.append(result)
                (a.output_dir / f"{condition}.{method}.seed{seed}.per_video.json").write_text(json.dumps(per_video, indent=2, sort_keys=True) + "\n")
                (a.output_dir / f"{a.set}.results.json").write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"condition": condition, "method": method, "seed": seed,
                                  "pauc_f": paucf["pauc_f"], "macro_f1": result["classification"]["macro_f1"],
                                  "fires_per_video": result["behavior"]["fires_per_video"]}), flush=True)
    summary = []
    for condition in a.conditions:
        for method in ("mlp", "clm_short_256"):
            rows = [r for r in runs if r["condition"] == condition and r["method"] == method]
            summary.append({"condition": condition, "method": method,
                            "mean_pauc_f": statistics.mean(r["no_judge_pauc_f"]["pauc_f"] for r in rows),
                            "sd_pauc_f": statistics.stdev(r["no_judge_pauc_f"]["pauc_f"] for r in rows),
                            "mean_macro_f1": statistics.mean(r["classification"]["macro_f1"] for r in rows),
                            "mean_fires_per_video": statistics.mean(r["behavior"]["fires_per_video"] for r in rows),
                            "mean_repeats_per_video": statistics.mean(r["behavior"]["repeat_fires_on_hold"] / len(manifest) for r in rows)})
    (a.output_dir / f"{a.set}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
