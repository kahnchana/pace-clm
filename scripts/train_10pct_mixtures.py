#!/usr/bin/env python3
"""Train the prespecified 24.3k mixture heads with a fixed dev protocol."""

from __future__ import annotations

import argparse
import copy
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_10pct_curve import calibrate
from train_and_eval_heads import build_model, logits_for

PRIOR = {"run1": (0.1563, 0.8170, 0.0267), "run2": (0.067, 0.917, 0.016)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-feature-dir", type=Path, required=True)
    p.add_argument("--mixture-feature-dir", type=Path)
    p.add_argument("--manifest-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", choices=("run1", "run2", "run3", "run4", "run5"), required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker and torch.cuda.device_count() > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={min(8,torch.cuda.device_count())}",
                   str(Path(__file__).resolve()), "--old-feature-dir", str(args.old_feature_dir),
                   "--output-dir", str(args.output_dir), "--conditions", *args.conditions, "--worker"]
        if args.mixture_feature_dir:
            command += ["--mixture-feature-dir", str(args.mixture_feature_dir)]
        if args.manifest_dir:
            command += ["--manifest-dir", str(args.manifest_dir)]
        subprocess.run(command, check=True)
        return
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    old = torch.load(args.old_feature_dir / "train.pt", map_location="cpu", weights_only=False)
    variants = torch.load(args.old_feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "dev"]
    train_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "train"]
    if len(dev_ids) != 2700 or len(train_ids) != 24300:
        raise RuntimeError("frozen old feature split mismatch")
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    dev_weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    dev_x = old["features"][dev_ids].float().to(device)
    dev_y = old["targets"][dev_ids].long().to(device)
    descriptive = old["action_features"].float().to(device)
    short = variants["short_labels"].float().to(device)
    selected = None
    if any(c in ("run3", "run4", "run5") for c in args.conditions):
        if not args.mixture_feature_dir or not args.manifest_dir:
            raise ValueError("new-selection conditions require mixture features and manifests")
        selected = torch.load(args.mixture_feature_dir / "union.pt", map_location="cpu", weights_only=False)
        lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(selected["metadata"])}
    run_no = 0
    for condition in args.conditions:
        if condition in PRIOR:
            train_x = old["features"][train_ids].float().to(device)
            train_y = old["targets"][train_ids].long().to(device)
            train_meta = [old["metadata"][i] for i in train_ids]
            class_weights = torch.tensor(PRIOR[condition], device=device) * 3
        else:
            rows = [json.loads(line) for line in (args.manifest_dir / f"{condition}.jsonl").read_text().splitlines()]
            indices = [lookup[(r["video_id"], int(r["frame_index"]))] for r in rows]
            train_x = selected["features"][indices].float().to(device)
            train_y = selected["targets"][indices].long().to(device)
            train_meta = rows
            class_weights = torch.ones(3, device=device)
        if len(train_y) != 24300:
            raise RuntimeError(f"{condition}: expected 24300 positions, got {len(train_y)}")
        sample_weights = class_weights[train_y]
        effective_n = float(sample_weights.sum().square() / sample_weights.square().sum())
        for method in ("mlp", "clm_short_256"):
            vectors = short if method == "clm_short_256" else descriptive
            for seed in (42, 43, 44):
                assigned = run_no % world == rank
                run_no += 1
                if not assigned:
                    continue
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                model = build_model(method, int(old["hidden_size"]), 256 if method == "clm_short_256" else 0).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
                generator = torch.Generator(device=device).manual_seed(seed)
                best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
                for epoch in range(1, 61):
                    model.train()
                    order = torch.randperm(len(train_y), generator=generator, device=device)
                    for batch in order.split(128):
                        per_item = F.cross_entropy(logits_for(model, train_x[batch], vectors), train_y[batch], reduction="none")
                        loss = (per_item * class_weights[train_y[batch]]).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        dev_loss = float(F.cross_entropy(logits_for(model, dev_x, vectors), dev_y))
                    if dev_loss < best_loss - 1e-4:
                        best_loss, best_epoch = dev_loss, epoch
                        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                        stale = 0
                    else:
                        stale += 1
                    if stale >= 6:
                        break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    dev_logits = logits_for(model, dev_x, vectors).cpu()
                margin, calibrated = calibrate(dev_logits, dev_y.cpu(), dev_weights)
                result = {"condition": condition, "method": method, "seed": seed,
                          "train_size": len(train_y), "train_distribution": dict(Counter(r["gold_action"] for r in train_meta)),
                          "train_videos": len({r["video_id"] for r in train_meta}), "effective_sample_size": effective_n,
                          "class_weights": class_weights.cpu().tolist(), "dev_size": len(dev_y),
                          "best_epoch": best_epoch, "dev_nll": best_loss, "response_margin": margin,
                          "dev_calibrated": calibrated, "parameters": sum(p.numel() for p in model.parameters())}
                checkpoint = args.output_dir / f"{condition}.{method}.seed{seed}.pt"
                torch.save({"model": best_state, "run": result, "dev_logits": dev_logits, "dev_targets": dev_y.cpu()}, checkpoint)
                print(json.dumps({k: result[k] for k in ("condition", "method", "seed", "best_epoch", "dev_nll", "response_margin")}), flush=True)


if __name__ == "__main__":
    main()
