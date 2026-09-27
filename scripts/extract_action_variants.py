#!/usr/bin/env python3
"""Extract fixed descriptive and short-label action embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from stream_gemma.inference.qwen_stream import QwenRuntime

from pace_clm.backbone import FrozenFeatureExtractor
from pace_clm.config import load_mvp_config
from pace_clm.experiment import direct_inputs
from pace_clm.labels import Action


def main() -> None:
    config = load_mvp_config(ROOT / "configs/qwen_mvp.yaml")
    runtime = QwenRuntime.load(
        checkpoint=str((ROOT / config["model"]["adapter"]).resolve()),
        model_name=config["model"]["name"],
        model_revision=config["model"]["revision"],
        frame_num_tokens=int(config["streaming"]["frame_num_tokens"]),
        generation_prefix_mode="direct",
    )
    extractor = FrozenFeatureExtractor(
        runtime.model, vision_token_ids=[runtime.processor.image_token_id]
    )
    variants = {}
    for version, texts in {
        "short_labels": {action.name: action.name for action in Action},
    }.items():
        features = []
        for action in Action:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": texts[action.name]}],
                }
            ]
            features.append(extractor.extract(direct_inputs(runtime, messages)).float().cpu())
        variants[version] = torch.cat(features)
    output = ROOT / "outputs/features/v1/action_variants.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(variants, output)
    print({name: list(value.shape) for name, value in variants.items()})


if __name__ == "__main__":
    main()
