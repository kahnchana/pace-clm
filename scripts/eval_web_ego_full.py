#!/usr/bin/env python3
"""Full, resumable WEB/EGO rollout of the frozen run5 MLP action head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG / "src"), str(SG)]

from pace_clm.backbone import FrozenFeatureExtractor
from pace_clm.experiment import direct_inputs
from pace_clm.labels import Action
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import QwenRuntime
from zs_infer.adapters import web

from generate_selected_responses import forced_response
from train_and_eval_heads import build_model


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hash_.update(block)
    return hash_.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("web", "ego"))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0, help="Only for a protocol smoke run; 0 means full split")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_promoted_contract(SG)
    source = SG / f"datasets/proactive-videoqa-{args.benchmark}/anno.json"
    video_root = SG / f"datasets/proactive-videoqa-{args.benchmark}/videos"
    source_fps = 2 if args.benchmark == "web" else 6
    examples, annotations = web.load_examples(source, video_root)
    expected = 500 if args.benchmark == "web" else 326
    if len(examples) != expected or len(annotations) != expected:
        raise RuntimeError(f"incomplete {args.benchmark} split: {len(examples)} resolved / {len(annotations)} annotated, expected {expected}")
    if args.max_examples:
        examples = examples[: args.max_examples]
    if rank == 0:
        manifest = {
            "benchmark": args.benchmark,
            "examples": len(examples),
            "annotation_sha256": digest(source),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest(args.checkpoint),
            "model": contract,
            "target_fps": 1,
            "source_fps": source_fps,
            "window_frames": 32,
            "memory_mode": "last-response",
            "memory_instruction_version": "v2",
            "memory_select": "aged-out",
            "prefix_mode": "direct",
            "response_max_new_tokens": args.max_new_tokens,
            "question_ids": [example["question_id"] for example in examples],
        }
        path = args.output_dir / "manifest.json"
        if path.is_file() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("output directory contains a different frozen manifest")
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    shard = args.output_dir / f"predictions.rank{rank:02d}.jsonl"
    completed = set()
    if shard.is_file():
        for line in shard.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["question_id"])
    assigned = [example for index, example in enumerate(examples) if index % world == rank and example["question_id"] not in completed]
    print(f"[rank {rank}] assigned={len(assigned)} completed={len(completed)}", flush=True)
    if not assigned:
        return

    runtime = QwenRuntime.load(
        checkpoint=contract["adapter"],
        model_name=contract["model_name"],
        model_revision=contract["model_revision"],
        frame_num_tokens=144,
        generation_prefix_mode="direct",
    )
    extractor = FrozenFeatureExtractor(runtime.model, vision_token_ids=(runtime.processor.image_token_id,))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["run"]["condition"] != "run5" or checkpoint["run"]["method"] != "mlp" or checkpoint["run"]["seed"] != 42:
        raise RuntimeError("unexpected action checkpoint")
    head = build_model("mlp", extractor.hidden_size, 0).to(runtime.device).eval()
    head.load_state_dict(checkpoint["model"])
    margin = float(checkpoint["run"]["response_margin"])
    counts: Counter[str] = Counter()

    def choose_turn(messages, *, max_new_tokens, vision=None):
        if vision is None:
            raise RuntimeError("vision cache is required for full rollout")
        inputs = direct_inputs(runtime, messages, vision)
        with runtime._cached_image_features(vision):
            pooled = extractor.extract(inputs)
        with torch.inference_mode():
            logits = head(pooled.float())
            logits[:, int(Action.Response)] -= margin
            action = Action(int(logits.argmax(-1).item()))
        counts[action.name] += 1
        if action == Action.Response:
            content = forced_response(runtime, messages, vision, min(max_new_tokens, args.max_new_tokens))
            if not content:
                counts["empty_response"] += 1
            return f"{STATE_TAGS['Response']} {content}".strip()
        return STATE_TAGS["Silence" if action == Action.Silence else "Standby"]

    runtime.generate_turn = choose_turn
    started = time.monotonic()
    with shard.open("a", encoding="utf-8") as handle:
        for index, example in enumerate(assigned, 1):
            prepared = web.prepare_frames(example, source_fps=source_fps, frame_cap=None)
            record = web.generate(
                runtime,
                example,
                mode="v24",
                source_fps=source_fps,
                frame_cap=None,
                v24_system_prompt=SYSTEM_PROMPT,
                v24_memory_mode="last-response",
                v24_memory_instruction_version="v2",
                v24_memory_select="aged-out",
                prepared=prepared,
            )
            record["pace_clm_actions"] = dict(counts)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(f"[rank {rank}] {index}/{len(assigned)} {example['question_id']} frames={len(prepared['images'])} fires={len(record['model_response_list'])} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
