#!/usr/bin/env python3
"""Replay the PACE-exposure run and retain its best early dev-NLL checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_10pct_curve import calibrate
from train_and_eval_heads import build_model, logits_for

PRESENTATIONS = 263760
UPDATES = 3260


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-feature-dir", type=Path, required=True)
    p.add_argument("--mixture-feature-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-step", type=int, default=1000)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old = torch.load(a.old_feature_dir / "train.pt", map_location="cpu", weights_only=False)
    union = torch.load(a.mixture_feature_dir / "union.pt", map_location="cpu", weights_only=False)
    variants = torch.load(a.old_feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(union["metadata"])}
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line]
    ids = [lookup[(r["video_id"], int(r["frame_index"]))] for r in rows]
    train_x = union["features"][ids].float().to(device)
    train_y = union["targets"][ids].long().to(device)
    dev_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "dev"]
    dev_x = old["features"][dev_ids].float().to(device)
    dev_y = old["targets"][dev_ids].long().to(device)
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    for method in ("mlp", "clm_short_256"):
        vectors = (old["action_features"] if method == "mlp" else variants["short_labels"]).float().to(device)
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            model = build_model(method, int(old["hidden_size"]), 256 if method == "clm_short_256" else 0).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
            generator = torch.Generator(device=device).manual_seed(seed)
            full = [torch.randperm(len(train_y), generator=generator, device=device) for _ in range(10)]
            tail_count = PRESENTATIONS - 10 * len(train_y)
            partial = []
            for action in range(3):
                action_ids = torch.where(train_y == action)[0]
                count = round(tail_count * len(action_ids) / len(train_y))
                partial.append(action_ids[torch.randperm(len(action_ids), generator=generator, device=device)[:count]])
            tail = torch.cat(partial)
            if len(tail) != tail_count:
                difference = tail_count - len(tail)
                if difference > 0:
                    tail = torch.cat([tail, torch.randperm(len(train_y), generator=generator, device=device)[:difference]])
                else:
                    tail = tail[:tail_count]
            tail = tail[torch.randperm(len(tail), generator=generator, device=device)]
            presentations = torch.cat([*full, tail])
            sizes = [81] * (PRESENTATIONS - 80 * UPDATES) + [80] * (UPDATES - (PRESENTATIONS - 80 * UPDATES))
            step_sizes = torch.tensor(sizes, device=device)[torch.randperm(UPDATES, generator=generator, device=device)].tolist()
            best_nll = float("inf")
            best_step = 0
            best_state = None
            cursor = 0
            history = []
            for step, size in enumerate(step_sizes[:a.max_step], 1):
                batch = presentations[cursor:cursor + size]
                cursor += size
                model.train()
                loss = F.cross_entropy(logits_for(model, train_x[batch], vectors), train_y[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        nll = float(F.cross_entropy(logits_for(model, dev_x, vectors), dev_y))
                    history.append({"step": step, "nll": nll})
                    if nll < best_nll:
                        best_nll = nll
                        best_step = step
                        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            if best_state is None:
                raise RuntimeError("no dev checkpoint was evaluated")
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                dev_logits = logits_for(model, dev_x, vectors).cpu()
            margin, calibrated = calibrate(dev_logits, dev_y.cpu(), weights)
            result = {"condition": "run5_early", "method": method, "seed": seed,
                      "best_step": best_step, "best_dev_nll": best_nll,
                      "response_margin": margin, "dev_calibrated": calibrated,
                      "dev_history": history}
            torch.save({"model": best_state, "run": result},
                       a.output_dir / f"run5_early.{method}.seed{seed}.pt")
            print(json.dumps({k: result[k] for k in ("method", "seed", "best_step", "best_dev_nll")}), flush=True)


if __name__ == "__main__":
    main()
