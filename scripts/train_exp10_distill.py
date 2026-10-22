#!/usr/bin/env python3
"""Train the matched E03/E04 frozen-head PACE decision-distillation pair."""

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
from train_and_eval_heads import build_model, logits_for

OLD = Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features")
OUT = ROOT / "outputs/pace-clm-exp10"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=("E03", "E04"), required=True)
    a = p.parse_args()
    cfg_path = ROOT / "manifests/exp10/config.v1.json"
    cfg = json.loads(cfg_path.read_text())
    method = cfg["runs"][a.run]["method"]
    train_cfg = cfg["training"]
    expected = {"E03": "clm_short_256", "E04": "mlp"}
    if method != expected[a.run]:
        raise RuntimeError("versioned pair configuration changed")
    source_manifest = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    selected = [json.loads(line) for line in source_manifest.read_text().splitlines() if line.strip()]
    teacher_path = OUT / "cache/teacher/teacher.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if teacher["manifest_sha256"] != sha(source_manifest):
        raise RuntimeError("teacher/source manifest mismatch")
    keys = [(r["video_id"], int(r["frame_index"])) for r in selected]
    if teacher["keys"] != keys or len(set(keys)) != 24300:
        raise RuntimeError("teacher position alignment mismatch")
    union = torch.load(ROOT / "outputs/pace-clm-mixtures/features/union.pt", map_location="cpu", weights_only=False)
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(union["metadata"])}
    ids = [lookup[key] for key in keys]
    train_x = union["features"][ids].float()
    train_y = union["targets"][ids].long()
    if not torch.equal(train_y, teacher["gold_action_ids"].long()):
        raise RuntimeError("teacher/feature labels differ")
    teacher_scores = teacher["log_scores"].float()
    old = torch.load(OLD / "train.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "dev"]
    if len(dev_ids) != 2700:
        raise RuntimeError("internal dev size mismatch")
    dev_x = old["features"][dev_ids].float()
    dev_y = old["targets"][dev_ids].long()
    if {r["video_id"] for r in selected} & {old["metadata"][i]["video_id"] for i in dev_ids}:
        raise RuntimeError("train/dev video overlap")
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    dev_weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    variants = torch.load(OLD / "action_variants.pt", map_location="cpu", weights_only=False)
    action_features = variants["short_labels"].float() if method == "clm_short_256" else old["action_features"].float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y, teacher_scores = train_x.to(device), train_y.to(device), teacher_scores.to(device)
    dev_x, dev_y, action_features = dev_x.to(device), dev_y.to(device), action_features.to(device)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = build_model(method, int(old["hidden_size"]), 256 if method == "clm_short_256" else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    T = float(cfg["interventions"]["pace_distillation"]["temperature"])
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator, device=device)
        for batch in order.split(train_cfg["batch_size"]):
            logits = logits_for(model, train_x[batch], action_features)
            gold_ce = F.cross_entropy(logits, train_y[batch])
            teacher_probs = F.softmax(teacher_scores[batch] / T, dim=-1)
            kl = F.kl_div(F.log_softmax(logits / T, dim=-1), teacher_probs,
                          reduction="batchmean") * T * T
            loss = 0.5 * gold_ce + 0.5 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = float(F.cross_entropy(logits_for(model, dev_x, action_features), dev_y))
        print(json.dumps({"run": a.run, "epoch": epoch, "dev_cross_entropy": dev_loss}), flush=True)
        if dev_loss < best_loss - 1e-4:
            best_loss, best_epoch = dev_loss, epoch
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= train_cfg["early_stop_patience"]:
            break
    if best_state is None:
        raise RuntimeError("no trained checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_logits = logits_for(model, dev_x, action_features).cpu()
    margin, calibrated = calibrate(dev_logits, dev_y.cpu(), dev_weights)
    result = {"condition": a.run, "method": method, "seed": seed,
              "train_size": len(train_y), "train_distribution": dict(Counter(r["gold_action"] for r in selected)),
              "dev_size": len(dev_y), "best_epoch": best_epoch, "dev_nll": best_loss,
              "response_margin": margin, "dev_calibrated": calibrated,
              "parameters": sum(p.numel() for p in model.parameters()),
              "config_sha256": sha(cfg_path), "train_manifest_sha256": sha(source_manifest),
              "teacher_sha256": sha(teacher_path)}
    path = OUT / a.run / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".pt.tmp")
    torch.save({"model": best_state, "run": result,
                "dev_logits": dev_logits, "dev_targets": dev_y.cpu()}, temp)
    os.replace(temp, path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
