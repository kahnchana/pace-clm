#!/usr/bin/env python3
"""One-GPU real PACE-Qwen hidden-state extraction contract smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STREAM_GEMMA / "src"))

from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.episode import format_frame_time
from stream_gemma.inference.qwen_adapters import (
    remove_empty_think_generation_suffix,
)
from stream_gemma.inference.qwen_stream import QwenRuntime

from pace_clm.backbone import FrozenFeatureExtractor, assert_frozen
from pace_clm.config import load_mvp_config
from pace_clm.experiment import state_timeline


def decode_first(path: str) -> Image.Image:
    capture = cv2.VideoCapture(path)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode first frame: {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def direct_inputs(runtime: QwenRuntime, messages: list[dict], vision=None) -> dict:
    if vision is None:
        inputs = runtime.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    else:
        inputs = runtime.assemble_turn(messages, vision)
    inputs = remove_empty_think_generation_suffix(inputs, runtime.tokenizer)
    return {
        key: value.to(runtime.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_mvp.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=STREAM_GEMMA
        / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl",
    )
    parser.add_argument("--training-video", action="store_true",
                        help="Use the first frozen run5 training video for the smoke test")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/smoke/real_model.json")
    args = parser.parse_args()
    config = load_mvp_config(args.config)
    if args.training_video:
        selected = json.loads((ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl")
                              .read_text().splitlines()[0])
        video_id = selected["video_id"]
        metadata = json.loads((STREAM_GEMMA / "datasets/markii/metadata.json").read_text())
        source = metadata[video_id]
        raw_record = json.loads((STREAM_GEMMA / "datasets/markii/train.jsonl")
                                .read_text().splitlines()[int(selected["raw_index"])])
        timeline = state_timeline(raw_record, metadata)
        row = {"video_id": video_id,
               "resolved_video_path": source.get("path") or str(
                   STREAM_GEMMA / "datasets/markii/videos" / f"{video_id}.mp4"),
               "inference_source_indices": [0], "question": timeline["question"]}
    else:
        row = json.loads(args.manifest.read_text().splitlines()[0])

    runtime = QwenRuntime.load(
        checkpoint=str((ROOT / config["model"]["adapter"]).resolve()),
        model_name=config["model"]["name"],
        model_revision=config["model"]["revision"],
        frame_num_tokens=int(config["streaming"]["frame_num_tokens"]),
        generation_prefix_mode=config["streaming"]["qwen_prefix_mode"],
    )
    extractor = FrozenFeatureExtractor(
        runtime.model, vision_token_ids=[runtime.processor.image_token_id]
    )
    assert_frozen(runtime.model)

    image = runtime._resize([decode_first(row["resolved_video_path"])])[0]
    frame = runtime.encode_frame(image)
    runtime.encode_frames_features([frame], chunk=1)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": row["question"]}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{format_frame_time(0)} "},
                {"type": "image", "image": image},
            ],
        },
    ]
    state_inputs = direct_inputs(runtime, messages, [frame])
    with runtime._cached_image_features([frame]):
        state_a = extractor.extract(state_inputs)
        state_b = extractor.extract(state_inputs)

    action_features = []
    action_config = json.loads(
        json.dumps(__import__("yaml").safe_load((ROOT / "configs/actions_v1.yaml").read_text()))
    )
    for description in action_config["actions"].values():
        action_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": str(description)}],
            }
        ]
        action_features.append(extractor.extract(direct_inputs(runtime, action_messages)))

    action_matrix = torch.cat(action_features, dim=0)
    trainable = [name for name, p in runtime.model.named_parameters() if p.requires_grad]
    result = {
        "hidden_size": extractor.hidden_size,
        "state_shape": list(state_a.shape),
        "action_shape": list(action_matrix.shape),
        "cached_uncached_max_abs": float((state_a - state_b).abs().max().cpu()),
        "backbone_trainable_parameters": trainable,
        "state_finite": bool(torch.isfinite(state_a).all()),
        "actions_finite": bool(torch.isfinite(action_matrix).all()),
        "lm_head_called": False,
        "video_id": row["video_id"],
        "causal_source_indices": [row["inference_source_indices"][0]],
    }
    if result["state_shape"] != [1, extractor.hidden_size]:
        raise RuntimeError(result)
    if result["action_shape"] != [3, extractor.hidden_size]:
        raise RuntimeError(result)
    if trainable or result["cached_uncached_max_abs"] != 0.0:
        raise RuntimeError(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
