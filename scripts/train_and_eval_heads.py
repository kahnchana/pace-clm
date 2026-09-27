#!/usr/bin/env python3
"""Train CLM/classifier baselines and evaluate them on frozen PACE-200 features."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pace_clm.heads import ContrastiveActionModel, ProjectionHead
from pace_clm.labels import Action


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int = 2560) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, len(Action)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class SharedContrastiveModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.head = ProjectionHead(input_dim, hidden_dim, output_dim, depth=3)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * self.head(states) @ self.head(actions).T


def logits_for(model, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if isinstance(model, (ContrastiveActionModel, SharedContrastiveModel)):
        return model(states, actions)
    return model(states)


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probabilities = logits.softmax(-1)
    predictions = logits.argmax(-1)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for gold, pred in zip(targets.cpu(), predictions.cpu(), strict=True):
        confusion[int(gold), int(pred)] += 1
    per_class = {}
    f1_values = []
    for action in Action:
        index = int(action)
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[action.name] = {"precision": precision, "recall": recall, "f1": f1}
    confidence, chosen = probabilities.max(-1)
    correct = chosen.eq(targets)
    ece = 0.0
    for low in torch.linspace(0, 0.9, 10, device=logits.device):
        mask = confidence.ge(low) & confidence.lt(low + 0.1)
        if mask.any():
            ece += float(
                mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    return {
        "accuracy": float(predictions.eq(targets).float().mean()),
        "macro_f1": sum(f1_values) / len(f1_values),
        "nll": float(F.cross_entropy(logits, targets)),
        "ece": ece,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def behavioral_metrics(predictions: torch.Tensor, metadata: list[dict]) -> dict:
    by_video: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for pred, row in zip(predictions.tolist(), metadata, strict=True):
        by_video[row["video_id"]].append(
            (int(row["frame_index"]), int(row["gold_action_id"]), int(pred))
        )
    fires = repeats = gold_responses = found = 0
    latencies = []
    out_of_window = 0
    for rows in by_video.values():
        rows.sort()
        pred_fire_times = [time for time, _, pred in rows if pred == int(Action.Response)]
        fires += len(pred_fire_times)
        repeats += sum(
            1 for _, gold, pred in rows if pred == int(Action.Response) and gold == int(Action.Hold)
        )
        gold_times = [time for time, gold, _ in rows if gold == int(Action.Response)]
        gold_responses += len(gold_times)
        used = set()
        for gold_time in gold_times:
            matches = [
                time
                for time in pred_fire_times
                if time not in used and gold_time <= time < gold_time + 4
            ]
            if matches:
                chosen = min(matches)
                used.add(chosen)
                found += 1
                latencies.append(chosen - gold_time)
        out_of_window += sum(time not in used for time in pred_fire_times)
    precision = found / fires if fires else 0.0
    recall = found / gold_responses if gold_responses else 0.0
    timing_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "fires": fires,
        "fires_per_video": fires / len(by_video),
        "repeat_fires_on_hold": repeats,
        "repeat_rate": repeats / fires if fires else 0.0,
        "window_matched_fires": found,
        "out_of_window_fires": out_of_window,
        "firing_precision_4s": precision,
        "response_recall_4s": recall,
        "timing_f1_4s": timing_f1,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def train_model(
    model: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[float]:
    torch.manual_seed(seed)
    generator = torch.Generator(device=states.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(states), generator=generator, device=states.device)
        for indices in order.split(128):
            logits = logits_for(model, states[indices], actions)
            loss = F.cross_entropy(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def tiny_overfit(
    states: torch.Tensor, targets: torch.Tensor, actions: torch.Tensor, *, input_dim: int
) -> dict:
    chosen = []
    for action in Action:
        chosen.extend(torch.where(targets == int(action))[0][:8].tolist())
    indices = torch.tensor(chosen, device=states.device)
    model = ContrastiveActionModel(input_dim).to(states.device)
    losses = train_model(
        model,
        states[indices],
        targets[indices],
        actions,
        seed=42,
        epochs=400,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    accuracy = float(
        logits_for(model, states[indices], actions).argmax(-1).eq(targets[indices]).float().mean()
    )
    if accuracy < 0.99:
        raise RuntimeError(f"tiny-set overfit failed: accuracy={accuracy:.3f}")
    return {"examples": len(indices), "accuracy": accuracy, "final_loss": losses[-1]}


def build_model(name: str, input_dim: int, output_dim: int) -> nn.Module:
    if name == "linear":
        return nn.Linear(input_dim, 3)
    if name == "mlp":
        return MLPClassifier(input_dim)
    if name.startswith("clm_shared"):
        return SharedContrastiveModel(input_dim, 1536, output_dim)
    return ContrastiveActionModel(input_dim, hidden_dim=1536, output_dim=output_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/features/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results/pace200_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = torch.load(args.feature_dir / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(args.feature_dir / "pace200.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = train["features"].float().to(device)
    train_y = train["targets"].long().to(device)
    val_x = val["features"].float().to(device)
    val_y = val["targets"].long().to(device)
    actions = train["action_features"].float().to(device)
    input_dim = int(train["hidden_size"])
    gate = tiny_overfit(train_x, train_y, actions, input_dim=input_dim)

    action_variants_path = args.feature_dir / "action_variants.pt"
    action_variants = (
        torch.load(action_variants_path, map_location="cpu", weights_only=False)
        if action_variants_path.is_file()
        else {}
    )
    specifications = [
        ("raw_cosine", 0, "descriptive"),
        ("linear", 0, "descriptive"),
        ("mlp", 0, "descriptive"),
        ("clm_128", 128, "descriptive"),
        ("clm_256", 256, "descriptive"),
        ("clm_512", 512, "descriptive"),
        ("clm_shared_512", 512, "descriptive"),
    ]
    if "short_labels" in action_variants:
        specifications.append(("clm_short_512", 512, "short_labels"))
    seeds = (42, 43, 44)
    results = []
    for name, output_dim, action_variant in specifications:
        run_actions = (
            actions
            if action_variant == "descriptive"
            else action_variants[action_variant].float().to(device)
        )
        run_seeds = (42,) if name == "raw_cosine" else seeds
        for seed in run_seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if name == "raw_cosine":
                logits = F.normalize(val_x, dim=-1) @ F.normalize(run_actions, dim=-1).T
                model = None
                losses = []
            else:
                model = build_model(name, input_dim, output_dim).to(device)
                losses = train_model(
                    model,
                    train_x,
                    train_y,
                    run_actions,
                    seed=seed,
                    epochs=80,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                )
                with torch.no_grad():
                    logits = logits_for(model, val_x, run_actions)
            metrics = classification_metrics(logits, val_y)
            predictions = logits.argmax(-1)
            behavior = behavioral_metrics(predictions, val["metadata"])
            run = {
                "method": name,
                "action_text": action_variant,
                "seed": seed,
                "trainable_parameters": sum(p.numel() for p in model.parameters()) if model else 0,
                "classification": metrics,
                "behavior": behavior,
                "final_train_loss": losses[-1] if losses else None,
                "prediction_distribution": dict(
                    Counter(Action(value).name for value in predictions.cpu().tolist())
                ),
            }
            results.append(run)
            checkpoint = args.output_dir / f"{name}.seed{seed}.pt"
            torch.save(
                {
                    "model": model.state_dict() if model else None,
                    "logits": logits.cpu(),
                    "probabilities": logits.softmax(-1).cpu(),
                    "predictions": predictions.cpu(),
                    "targets": val_y.cpu(),
                    "metadata": val["metadata"],
                    "run": run,
                },
                checkpoint,
            )
            print(json.dumps(run, sort_keys=True), flush=True)
    payload = {
        "tiny_overfit": gate,
        "train_distribution": dict(Counter(Action(x).name for x in train_y.cpu().tolist())),
        "validation_distribution": dict(Counter(Action(x).name for x in val_y.cpu().tolist())),
        "runs": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
