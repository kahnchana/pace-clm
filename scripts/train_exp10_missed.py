#!/usr/bin/env python3
"""Train the matched E01/E02 missed-window ranking pair."""

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=("E01", "E02"), required=True)
    a = p.parse_args()
    config_path = ROOT / "manifests/exp10/config.v1.json"
    config = json.loads(config_path.read_text())
    method = config["runs"][a.run]["method"]
    if method != {"E01": "clm_short_256", "E02": "mlp"}[a.run]:
        raise RuntimeError("versioned E01/E02 pair configuration changed")
    train_path = ROOT / "manifests/exp10/e01_train.jsonl"
    triples_path = ROOT / "manifests/exp10/e01_rank_triples.jsonl"
    train_rows = read_jsonl(train_path)
    triples = read_jsonl(triples_path)
    if len(train_rows) != 24300 or len(triples) != round(3791 * .2):
        raise RuntimeError("E01/E02 frozen position or triple count mismatch")
    features_path = OUT / "cache/e01/e01.pt"
    data = torch.load(features_path, map_location="cpu", weights_only=False)
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(data["metadata"])}
    if len(lookup) != len(data["metadata"]):
        raise RuntimeError("duplicate E01 feature")
    train_ids = [lookup[(r["video_id"], int(r["frame_index"]))] for r in train_rows]
    train_x = data["features"][train_ids].float()
    train_y = data["targets"][train_ids].long()
    if train_y.tolist() != [r["gold_action_id"] for r in train_rows]:
        raise RuntimeError("E01 feature/label mismatch")
    triples_ids = torch.tensor([[lookup[(t["video_id"], int(t[key]))] for key in
                                 ("positive", "before", "after")] for t in triples], dtype=torch.long)
    old = torch.load(OLD / "train.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, r in enumerate(old["metadata"]) if r["selection_split"] == "dev"]
    if len(dev_ids) != 2700:
        raise RuntimeError("dev position count mismatch")
    dev_x = old["features"][dev_ids].float()
    dev_y = old["targets"][dev_ids].long()
    if {r["video_id"] for r in train_rows} & {old["metadata"][i]["video_id"] for i in dev_ids}:
        raise RuntimeError("train/dev video overlap")
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    dev_weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    variants = torch.load(OLD / "action_variants.pt", map_location="cpu", weights_only=False)
    actions = variants["short_labels"].float() if method == "clm_short_256" else old["action_features"].float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)
    all_x = data["features"].float().to(device)
    triples_ids = triples_ids.to(device)
    dev_x, dev_y, actions = dev_x.to(device), dev_y.to(device), actions.to(device)
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
    margin = float(config["interventions"]["missed_window"]["ranking_margin_logit"])
    rank_weight = float(config["interventions"]["missed_window"]["ranking_loss_weight"])
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator, device=device)
        rank_order = torch.randperm(len(triples), generator=generator, device=device)
        batches = order.split(config["training"]["batch_size"])
        for number, batch in enumerate(batches):
            logits = logits_for(model, train_x[batch], actions)
            ce = F.cross_entropy(logits, train_y[batch])
            lower = number * len(triples) // len(batches)
            upper = (number + 1) * len(triples) // len(batches)
            selected_triples = triples_ids[rank_order[lower:upper]]
            ranked_logits = logits_for(model, all_x[selected_triples.flatten()], actions)
            response = ranked_logits[:, 0].reshape(-1, 3)
            ranking = (F.relu(margin - (response[:, 0] - response[:, 1])) +
                       F.relu(margin - (response[:, 0] - response[:, 2]))).mean() / 2
            loss = ce + rank_weight * ranking
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
        raise RuntimeError("no trained E01/E02 checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_logits = logits_for(model, dev_x, actions).cpu()
    response_margin, calibrated = calibrate(dev_logits, dev_y.cpu(), dev_weights)
    result = {"condition": a.run, "method": method, "seed": seed,
              "train_size": len(train_y), "train_distribution": dict(Counter(r["gold_action"] for r in train_rows)),
              "dev_size": len(dev_y), "best_epoch": best_epoch, "dev_nll": best_loss,
              "response_margin": response_margin, "dev_calibrated": calibrated,
              "parameters": sum(p.numel() for p in model.parameters()),
              "config_sha256": sha(config_path), "train_manifest_sha256": sha(train_path),
              "ranking_triples_sha256": sha(triples_path), "feature_cache_sha256": sha(features_path)}
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
