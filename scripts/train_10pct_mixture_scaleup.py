#!/usr/bin/env python3
"""Train selected-mixture and balanced heads for 263,760 presentations/3,260 updates."""

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
PRESENTATIONS = 263760
UPDATES = 3260


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selected", choices=("control", "run1", "run2", "run3", "run4", "run5"), required=True)
    p.add_argument("--old-feature-dir", type=Path, required=True)
    p.add_argument("--mixture-feature-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if not a.worker and torch.cuda.device_count() > 1:
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={min(8,torch.cuda.device_count())}",
                        str(Path(__file__).resolve()), "--selected", a.selected, "--old-feature-dir", str(a.old_feature_dir),
                        "--mixture-feature-dir", str(a.mixture_feature_dir), "--manifest-dir", str(a.manifest_dir),
                        "--output-dir", str(a.output_dir), "--worker"], check=True)
        return
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    old = torch.load(a.old_feature_dir / "train.pt", map_location="cpu", weights_only=False)
    variants = torch.load(a.old_feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    dev_ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "dev"]
    dev_x = old["features"][dev_ids].float().to(device)
    dev_y = old["targets"][dev_ids].long().to(device)
    summary = json.loads((ROOT / "manifests/pace_train_10pct_summary.json").read_text())
    dev_weights = torch.tensor([
        summary["candidate_distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"] /
        summary["distribution"][f"dev/{old['metadata'][i]['gold_action']}/{old['metadata'][i]['selection_stratum']}"]
        for i in dev_ids], dtype=torch.float32)
    if a.selected in ("control", "run1", "run2"):
        ids = [i for i, row in enumerate(old["metadata"]) if row["selection_split"] == "train"]
        train_x = old["features"][ids].float().to(device)
        train_y = old["targets"][ids].long().to(device)
        train_meta = [old["metadata"][i] for i in ids]
    else:
        union = torch.load(a.mixture_feature_dir / "union.pt", map_location="cpu", weights_only=False)
        lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(union["metadata"])}
        train_meta = [json.loads(line) for line in (a.manifest_dir / f"{a.selected}.jsonl").read_text().splitlines()]
        ids = [lookup[(r["video_id"], int(r["frame_index"]))] for r in train_meta]
        train_x = union["features"][ids].float().to(device)
        train_y = union["targets"][ids].long().to(device)
    if len(train_y) != 24300:
        raise RuntimeError("selected training positions drifted")
    short = variants["short_labels"].float().to(device)
    descriptive = old["action_features"].float().to(device)
    condition_weights = torch.tensor(PRIOR.get(a.selected, (1/3, 1/3, 1/3)), device=device) * 3
    run_no = 0
    for regime in ("selected", "balanced"):
        weights = condition_weights if regime == "selected" else torch.ones(3, device=device)
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
                # Ten full passes plus a deterministic, state-stratified partial pass.
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
                if len(presentations) != PRESENTATIONS:
                    raise RuntimeError("presentation budget mismatch")
                sizes = [81] * (PRESENTATIONS - 80 * UPDATES) + [80] * (UPDATES - (PRESENTATIONS - 80 * UPDATES))
                step_sizes = torch.tensor(sizes, device=device)[torch.randperm(UPDATES, generator=generator, device=device)].tolist()
                cursor = 0
                dev_history = []
                for step, size in enumerate(step_sizes, 1):
                    batch = presentations[cursor:cursor + size]
                    cursor += size
                    model.train()
                    loss = (F.cross_entropy(logits_for(model, train_x[batch], vectors), train_y[batch], reduction="none") * weights[train_y[batch]]).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if step % 100 == 0 or step == UPDATES:
                        model.eval()
                        with torch.no_grad():
                            dev_history.append({"step": step, "nll": float(F.cross_entropy(logits_for(model, dev_x, vectors), dev_y))})
                if cursor != PRESENTATIONS:
                    raise RuntimeError("optimizer budget mismatch")
                model.eval()
                with torch.no_grad():
                    dev_logits = logits_for(model, dev_x, vectors).cpu()
                margin, calibrated = calibrate(dev_logits, dev_y.cpu(), dev_weights)
                sample_weights = weights[train_y]
                result = {"selected_mixture": a.selected, "regime": regime, "method": method, "seed": seed,
                          "unique_positions": len(train_y), "unique_videos": len({r["video_id"] for r in train_meta}),
                          "state_distribution": dict(Counter(r["gold_action"] for r in train_meta)),
                          "presentations": PRESENTATIONS, "optimizer_updates": UPDATES,
                          "position_repeat_min": 10, "position_repeat_max": 11,
                          "effective_sample_size_per_pass": float(sample_weights.sum().square() / sample_weights.square().sum()),
                          "class_weights": weights.cpu().tolist(), "dev_history": dev_history,
                          "response_margin": margin, "dev_calibrated": calibrated}
                checkpoint = a.output_dir / f"{regime}.{method}.seed{seed}.pt"
                torch.save({"model": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                            "run": result, "dev_logits": dev_logits, "dev_targets": dev_y.cpu()}, checkpoint)
                print(json.dumps({"regime": regime, "method": method, "seed": seed,
                                  "updates": UPDATES, "presentations": PRESENTATIONS, "dev_nll": dev_history[-1]["nll"]}), flush=True)


if __name__ == "__main__":
    main()
