#!/usr/bin/env python3
"""Full, resumable WEB/EGO rollout of a frozen exp10 head or PACE token policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
OLD = Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG / "src"), str(SG)]

from generate_selected_responses import forced_response
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import QwenRuntime
from train_and_eval_heads import build_model, logits_for
from zs_infer.adapters import web

from pace_clm.backbone import FrozenFeatureExtractor
from pace_clm.experiment import direct_inputs
from pace_clm.labels import Action
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from pace_clm.temporal import CausalTemporalAdapter, TemporalHead
from pace_clm.utility import UtilityHead


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
    parser.add_argument("--token-baseline", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0, help="Only for a protocol smoke run; 0 means full split")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.token_baseline and args.checkpoint != ROOT / "outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt":
        parser.error("token baseline cannot also specify a checkpoint")
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
            "checkpoint": None if args.token_baseline else str(args.checkpoint.resolve()),
            "checkpoint_sha256": None if args.token_baseline else digest(args.checkpoint),
            "policy": "pace_token" if args.token_baseline else "frozen_head",
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
    if not args.token_baseline:
        extractor = FrozenFeatureExtractor(runtime.model, vision_token_ids=(runtime.processor.image_token_id,))
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        method = checkpoint["run"]["method"]
        if method not in ("mlp", "clm_short_256") or checkpoint["run"]["seed"] != 42:
            raise RuntimeError("unexpected action checkpoint")
        temporal = bool(checkpoint["run"].get("temporal_hidden_width"))
        utility = bool(checkpoint["run"].get("utility_hidden_width"))
        base = build_model(method, extractor.hidden_size, 256 if method == "clm_short_256" else 0)
        head = (TemporalHead(CausalTemporalAdapter(extractor.hidden_size, 256), base)
                if temporal else UtilityHead(base, extractor.hidden_size, 256)
                if utility else base)
        head.load_state_dict(checkpoint["model"])
        head = head.to(runtime.device).eval()
        short = (torch.load(OLD / "action_variants.pt", map_location="cpu", weights_only=False)["short_labels"]
                 .float().to(runtime.device) if method == "clm_short_256" else None)
        margin = float(checkpoint["run"]["response_margin"])
        counts: Counter[str] = Counter()
        temporal_states: deque[torch.Tensor] = deque(maxlen=4)

        def choose_turn(messages, *, max_new_tokens, vision=None):
            if vision is None:
                raise RuntimeError("vision cache is required for full rollout")
            inputs = direct_inputs(runtime, messages, vision)
            with runtime._cached_image_features(vision):
                pooled = extractor.extract(inputs)
            with torch.inference_mode():
                if temporal:
                    temporal_states.append(pooled.float().squeeze(0))
                    states = [temporal_states[0]] * (4 - len(temporal_states)) + list(temporal_states)
                    logits = head(torch.stack(states).unsqueeze(0), short)
                elif utility:
                    logits, predicted_utility = head(pooled.float(), short)
                    logits[:, int(Action.Response)] += float(checkpoint["run"]["utility_alpha"]) * predicted_utility
                else:
                    logits = logits_for(head, pooled.float(), short)
                logits[:, int(Action.Response)] -= margin
                action = Action(int(logits.argmax(-1).item()))
            counts[action.name] += 1
            if action == Action.Response:
                content = forced_response(runtime, messages, vision, min(max_new_tokens, args.max_new_tokens))
                if not content or not content.strip():
                    raise RuntimeError("empty Response text")
                return f"{STATE_TAGS['Response']} {content}".strip()
            return STATE_TAGS["Silence" if action == Action.Silence else "Standby"]

        runtime.generate_turn = choose_turn
    started = time.monotonic()
    with shard.open("a", encoding="utf-8") as handle:
        for index, example in enumerate(assigned, 1):
            video_started = time.monotonic()
            if not args.token_baseline and temporal:
                temporal_states.clear()
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
            record["exp10_video_wall_seconds"] = time.monotonic() - video_started
            record["exp10_frame_count"] = len(prepared["images"])
            if not args.token_baseline:
                record["pace_clm_actions"] = dict(counts)
            if int(record.get("v24_parse", {}).get("malformed", 0)):
                raise RuntimeError("malformed action emission")
            if any(not str(fire.get("content", "")).strip() for fire in record["model_response_list"]):
                raise RuntimeError("empty Response text")
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(f"[rank {rank}] {index}/{len(assigned)} {example['question_id']} frames={len(prepared['images'])} fires={len(record['model_response_list'])} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
