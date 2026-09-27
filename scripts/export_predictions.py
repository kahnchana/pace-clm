#!/usr/bin/env python3
"""Export per-decision logits/probabilities from binary run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pace_clm.labels import Action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/results/pace200_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.result_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in sorted(args.result_dir.glob(f"*.seed{args.seed}.pt")):
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rows = []
        for index, metadata in enumerate(artifact["metadata"]):
            gold = int(artifact["targets"][index])
            predicted = int(artifact["predictions"][index])
            rows.append(
                {
                    **metadata,
                    "gold_action": Action(gold).name,
                    "gold_action_id": gold,
                    "predicted_action": Action(predicted).name,
                    "predicted_action_id": predicted,
                    "logits": [float(value) for value in artifact["logits"][index]],
                    "probabilities": [float(value) for value in artifact["probabilities"][index]],
                }
            )
        output = output_dir / f"{checkpoint.name.removesuffix('.pt')}.jsonl"
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
