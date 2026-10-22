#!/usr/bin/env python3
"""Train the matched E09/E10 answer-aware utility pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_10pct_curve import calibrate
from train_and_eval_heads import build_model

from pace_clm.utility import UtilityHead

OLD = Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features")
OUT = ROOT / "outputs/pace-clm-exp10"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=("E09", "E10"), required=True)
    a = p.parse_args()
    cfg_path = ROOT / "manifests/exp10/config.v1.json"
    cfg = json.loads(cfg_path.read_text())
    method = cfg["runs"][a.run]["method"]
    if method != {"E09": "clm_short_256", "E10": "mlp"}[a.run]:
        raise RuntimeError("utility pair method changed")
    source_manifest = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    selected = [json.loads(line) for line in source_manifest.read_text().splitlines() if line.strip()]
    if len(selected) != 24300:
        raise RuntimeError("run5 training size changed")
    union = torch.load(ROOT / "outputs/pace-clm-mixtures/features/union.pt", map_location="cpu", weights_only=False)
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(union["metadata"])}
    keys = [(r["video_id"], int(r["frame_index"])) for r in selected]
    ids = [lookup[key] for key in keys]
    train_x, train_y = union["features"][ids].float(), union["targets"][ids].long()
    if train_y.tolist() != [r["gold_action_id"] for r in selected]:
        raise RuntimeError("utility training feature labels mismatch")
    utility_path = OUT / "cache/utility/utility_targets.pt"
    utility = torch.load(utility_path, map_location="cpu", weights_only=False)
    candidate_manifest = ROOT / "manifests/exp10/e09_utility_candidates.jsonl"
    candidates = [json.loads(line) for line in candidate_manifest.read_text().splitlines() if line.strip()]
    if utility["keys"] != [(r["video_id"], int(r["frame_index"])) for r in candidates] or len(candidates) != 4000:
        raise RuntimeError("utility candidate alignment mismatch")
    if {r["video_id"] for r in candidates} - {r["video_id"] for r in selected}:
        raise RuntimeError("utility candidate outside training videos")
    old = torch.load(OLD / "train.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "dev"]
    if len(dev_ids) != 2700 or {r["video_id"] for r in selected} & {old["metadata"][i]["video_id"] for i in dev_ids}:
        raise RuntimeError("utility train/dev video overlap")
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    variants = torch.load(OLD / "action_variants.pt", map_location="cpu", weights_only=False)
    actions = variants["short_labels"].float() if method == "clm_short_256" else old["action_features"].float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)
    utility_x = utility["features"].float().to(device)
    utility_y = utility["targets"].float().to(device)
    dev_x, dev_y = old["features"][dev_ids].float().to(device), old["targets"][dev_ids].long().to(device)
    actions = actions.to(device)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    base = build_model(method, int(old["hidden_size"]), 256 if method == "clm_short_256" else 0)
    model = UtilityHead(base, int(old["hidden_size"]), 256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator, device=device)
        batches = list(order.split(cfg["training"]["batch_size"]))
        utility_order = torch.randperm(len(utility_y), generator=generator, device=device)
        utility_batches = torch.tensor_split(utility_order, len(batches))
        for batch, utility_batch in zip(batches, utility_batches, strict=True):
            logits, _ = model(train_x[batch], actions if method == "clm_short_256" else None)
            _, predicted_utility = model(utility_x[utility_batch], actions if method == "clm_short_256" else None)
            loss = F.cross_entropy(logits, train_y[batch]) + 0.25 * F.mse_loss(predicted_utility, utility_y[utility_batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_logits, dev_utility = model(dev_x, actions if method == "clm_short_256" else None)
            dev_loss = float(F.cross_entropy(dev_logits, dev_y))
        print(json.dumps({"run": a.run, "epoch": epoch, "dev_cross_entropy": dev_loss}), flush=True)
        if dev_loss < best_loss - 1e-4:
            best_loss, best_epoch = dev_loss, epoch
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= cfg["training"]["early_stop_patience"]:
            break
    if best_state is None:
        raise RuntimeError("no trained utility checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_logits, dev_utility = model(dev_x, actions if method == "clm_short_256" else None)
        dev_logits, dev_utility = dev_logits.cpu(), dev_utility.cpu()
    options = []
    for alpha in cfg["interventions"]["answer_utility"]["alpha_grid"]:
        adjusted = dev_logits.clone()
        adjusted[:, 0] += float(alpha) * dev_utility
        margin, metrics = calibrate(adjusted, dev_y.cpu(), weights)
        options.append((metrics["weighted_macro_f1"], -float(alpha), float(alpha), margin, metrics))
    _, _, alpha, margin, calibrated = max(options)
    result = {"condition": a.run, "method": method, "seed": seed,
              "train_size": len(train_y), "train_distribution": dict(Counter(r["gold_action"] for r in selected)),
              "utility_candidate_count": len(utility_y), "dev_size": len(dev_y),
              "best_epoch": best_epoch, "dev_nll": best_loss, "response_margin": margin,
              "utility_alpha": alpha, "utility_hidden_width": 256, "dev_calibrated": calibrated,
              "parameters": sum(p.numel() for p in model.parameters()),
              "config_sha256": sha(cfg_path), "train_manifest_sha256": sha(source_manifest),
              "utility_targets_sha256": sha(utility_path)}
    path = OUT / a.run / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".pt.tmp")
    torch.save({"model": best_state, "run": result,
                "dev_logits": dev_logits, "dev_utility": dev_utility, "dev_targets": dev_y.cpu()}, temp)
    os.replace(temp, path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
