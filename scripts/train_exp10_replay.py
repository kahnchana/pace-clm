#!/usr/bin/env python3
"""Train the matched E05/E06 generated-history replay pair."""

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
    p.add_argument("--run", choices=("E05", "E06"), required=True)
    a = p.parse_args()
    config_path = ROOT / "manifests/exp10/config.v1.json"
    config = json.loads(config_path.read_text())
    method = config["runs"][a.run]["method"]
    if method != {"E05": "clm_short_256", "E06": "mlp"}[a.run]:
        raise RuntimeError("versioned replay pair configuration changed")
    manifest_path = ROOT / "manifests/exp10/e05_train.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    cache_path = OUT / "cache/replay/e05.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["manifest_sha256"] != sha(manifest_path):
        raise RuntimeError("replay feature manifest mismatch")
    if cache["keys"] != [(r["video_id"], int(r["frame_index"])) for r in rows] or len(rows) != 24300:
        raise RuntimeError("replay position alignment mismatch")
    if cache["targets"].tolist() != [r["gold_action_id"] for r in rows]:
        raise RuntimeError("replay labels mismatch")
    old = torch.load(OLD / "train.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, r in enumerate(old["metadata"]) if r["selection_split"] == "dev"]
    if len(dev_ids) != 2700 or {r["video_id"] for r in rows} & {old["metadata"][i]["video_id"] for i in dev_ids}:
        raise RuntimeError("replay train/dev separation mismatch")
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    dev_weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    variants = torch.load(OLD / "action_variants.pt", map_location="cpu", weights_only=False)
    actions = variants["short_labels"].float() if method == "clm_short_256" else old["action_features"].float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x = cache["features"].float().to(device)
    train_y = cache["targets"].long().to(device)
    dev_x = old["features"][dev_ids].float().to(device)
    dev_y = old["targets"][dev_ids].long().to(device)
    actions = actions.to(device)
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = build_model(method, int(old["hidden_size"]), 256 if method == "clm_short_256" else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator, device=device)
        for batch in order.split(config["training"]["batch_size"]):
            loss = F.cross_entropy(logits_for(model, train_x[batch], actions), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = float(F.cross_entropy(logits_for(model, dev_x, actions), dev_y))
        print(json.dumps({"run": a.run, "epoch": epoch, "dev_cross_entropy": dev_loss}), flush=True)
        if dev_loss < best_loss - 1e-4:
            best_loss, best_epoch = dev_loss, epoch
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= config["training"]["early_stop_patience"]:
            break
    if best_state is None:
        raise RuntimeError("no trained replay checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_logits = logits_for(model, dev_x, actions).cpu()
    margin, calibrated = calibrate(dev_logits, dev_y.cpu(), dev_weights)
    result = {"condition": a.run, "method": method, "seed": seed,
              "train_size": len(train_y), "train_distribution": dict(Counter(r["gold_action"] for r in rows)),
              "dev_size": len(dev_y), "best_epoch": best_epoch, "dev_nll": best_loss,
              "response_margin": margin, "dev_calibrated": calibrated,
              "parameters": sum(p.numel() for p in model.parameters()),
              "config_sha256": sha(config_path), "train_manifest_sha256": sha(manifest_path),
              "replay_cache_sha256": sha(cache_path)}
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
