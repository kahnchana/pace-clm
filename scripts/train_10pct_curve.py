#!/usr/bin/env python3
"""Train the frozen-feature nested curve with video-disjoint early stopping."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pace_clm.labels import Action
from train_and_eval_heads import build_model, classification_metrics, logits_for


def order_key(row: dict) -> bytes:
    return hashlib.sha256(f"curve:{row['video_id']}:{row['frame_index']}".encode()).digest()


def weighted_macro_f1(
    predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> float:
    confusion = torch.bincount(
        3 * targets.long() + predictions.long(), weights=weights, minlength=9
    ).reshape(3, 3)
    scores = []
    for action in Action:
        i = int(action)
        tp = float(confusion[i, i])
        denominator = float(confusion[i, :].sum() + confusion[:, i].sum())
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def calibrate(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> tuple[float, dict]:
    # Fixed budget for every learned method: 41 thresholds of the Response margin.
    margin = logits[:, 0] - logits[:, 1:].max(-1).values
    thresholds = torch.quantile(margin.float(), torch.linspace(0, 1, 41))
    candidates = sorted({float(x) for x in thresholds})
    best = None
    for threshold in candidates:
        adjusted = logits.clone()
        adjusted[:, 0] -= threshold
        metrics = classification_metrics(adjusted, targets)
        metrics["weighted_macro_f1"] = weighted_macro_f1(adjusted.argmax(-1), targets, weights)
        key = (metrics["weighted_macro_f1"], -abs(threshold))
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker and torch.cuda.device_count() >= 8:
        subprocess.run([
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            str(Path(__file__).resolve()), "--feature-dir", str(args.feature_dir),
            "--output-dir", str(args.output_dir), "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience), "--worker",
        ], check=True)
        merged = []
        for rank in range(8):
            path = args.output_dir / f"curve.rank{rank:02d}.json"
            merged.extend(json.loads(path.read_text()))
        if len(merged) != 60:
            raise RuntimeError(f"expected 60 curve runs, found {len(merged)}")
        merged.sort(key=lambda run: (run["train_size"], run["method"], run["seed"]))
        (args.output_dir / "curve.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print("merged 60 learning-curve runs", flush=True)
        return
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    data = torch.load(args.feature_dir / "train.pt", map_location="cpu", weights_only=False)
    variants = torch.load(args.feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    manifest_rows = [json.loads(line) for line in
                     (ROOT / "manifests/pace_train_10pct_decisions.jsonl").read_text().splitlines() if line]
    expected = [(r["video_id"], int(r["frame_index"]), int(r["gold_action_id"]))
                for r in manifest_rows]
    observed = [(r["video_id"], int(r["frame_index"]), int(r["gold_action_id"]))
                for r in metadata]
    if expected != observed:
        raise RuntimeError("merged features do not match the frozen ordered decision manifest")
    manifest_summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    train_ids = [i for i, row in enumerate(metadata) if row["selection_split"] == "train"]
    dev_ids = [i for i, row in enumerate(metadata) if row["selection_split"] == "dev"]
    if len(train_ids) != 24300 or len(dev_ids) != 2700:
        raise RuntimeError(f"unexpected train/dev size: {len(train_ids)}/{len(dev_ids)}")
    if {metadata[i]["video_id"] for i in train_ids} & {metadata[i]["video_id"] for i in dev_ids}:
        raise RuntimeError("train/dev video overlap")
    by_action = {action: sorted((i for i in train_ids if int(data["targets"][i]) == int(action)), key=lambda i: order_key(metadata[i])) for action in Action}
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
    features = data["features"].float().to(device)
    targets = data["targets"].long().to(device)
    descriptive = data["action_features"].float().to(device)
    short = variants["short_labels"].float().to(device)
    dev_indices = torch.tensor(dev_ids, device=device)
    selected_distribution = manifest_summary["distribution"]
    candidate_distribution = manifest_summary["candidate_distribution"]
    dev_weights = torch.tensor([
        candidate_distribution[f"dev/{metadata[i]['gold_action']}/{metadata[i]['selection_stratum']}"]
        / selected_distribution[f"dev/{metadata[i]['gold_action']}/{metadata[i]['selection_stratum']}"]
        for i in dev_ids
    ], dtype=torch.float32)
    methods = (("linear", 0, descriptive), ("mlp", 0, descriptive),
               ("clm_128", 128, descriptive), ("clm_256", 256, descriptive),
               ("clm_short_256", 256, short))
    results = []
    run_number = 0
    for size in (768, 3000, 12000, 24300):
        if size > len(train_ids):
            raise RuntimeError("subset larger than train split")
        selected = sorted(i for action in Action for i in by_action[action][:size // 3])
        index = torch.tensor(selected, device=device)
        for name, output_dim, action_vectors in methods:
            for seed in (42, 43, 44):
                assigned = run_number % world == rank
                run_number += 1
                if not assigned:
                    continue
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                model = build_model(name, int(data["hidden_size"]), output_dim).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
                generator = torch.Generator(device=device).manual_seed(seed)
                best_loss = float("inf")
                best_epoch = 0
                best_state = None
                stale = 0
                for epoch in range(1, args.max_epochs + 1):
                    model.train()
                    permutation = index[torch.randperm(len(index), generator=generator, device=device)]
                    for batch in permutation.split(128):
                        loss = F.cross_entropy(logits_for(model, features[batch], action_vectors), targets[batch])
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        dev_logits = logits_for(model, features[dev_indices], action_vectors)
                        dev_loss = float(F.cross_entropy(dev_logits, targets[dev_indices]))
                    if dev_loss < best_loss - 1e-4:
                        best_loss = dev_loss
                        best_epoch = epoch
                        best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
                        stale = 0
                    else:
                        stale += 1
                    if stale >= args.patience:
                        break
                assert best_state is not None
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    dev_logits = logits_for(model, features[dev_indices], action_vectors).cpu()
                dev_y = targets[dev_indices].cpu()
                margin, calibrated = calibrate(dev_logits, dev_y, dev_weights)
                result = {
                    "method": name, "seed": seed, "train_size": size,
                    "train_distribution": dict(Counter(Action(int(targets[i])).name for i in selected)),
                    "train_videos": len({metadata[i]["video_id"] for i in selected}),
                    "dev_size": len(dev_ids), "best_epoch": best_epoch,
                    "dev_nll": best_loss, "response_margin": margin,
                    "dev_calibrated": calibrated,
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                checkpoint = args.output_dir / f"{name}.n{size}.seed{seed}.pt"
                torch.save({"model": best_state, "run": result, "dev_logits": dev_logits,
                            "dev_targets": dev_y}, checkpoint)
                results.append(result)
                curve_path = args.output_dir / (f"curve.rank{rank:02d}.json" if world > 1 else "curve.json")
                curve_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"rank": rank, **{k: result[k] for k in ("method", "seed", "train_size", "best_epoch", "dev_nll", "response_margin")}}), flush=True)


if __name__ == "__main__":
    main()
