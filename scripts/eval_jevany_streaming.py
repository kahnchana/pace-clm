#!/usr/bin/env python3
"""Resumable self-generated PACE rollout with a saved JevAny-style controller."""

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

from generate_selected_responses import forced_response
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import QwenRuntime
from zs_infer.adapters import web

from pace_clm.experiment import decode_indices
from pace_clm.jevany_streaming import (
    ACTIONS, PointerHead, controller_logits, decision_inputs, option_text,
    response_blind_messages,
)
from pace_clm.stream_gemma_adapter import validate_promoted_contract


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def with_options(messages: list[dict]) -> list[dict]:
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("current causal frame must be the final user turn")
    augmented = list(messages)
    last = dict(messages[-1])
    last["content"] = list(last["content"]) + [{"type": "text", "text": option_text()}]
    augmented[-1] = last
    return augmented


def base_answer(runtime, messages: list[dict], vision: list, max_new_tokens: int) -> str:
    """Ask untuned Qwen for answer text without PACE's learned Response prefix."""
    answer_messages = [dict(message) for message in messages]
    answer_messages[0]["content"] = [{"type": "text", "text":
        "Answer the user's question using only the video frames observed so far. "
        "Give one concise, factual answer in plain text. Do not use streaming control tags."}]
    answer_messages[-1]["content"] = list(messages[-1]["content"]) + [{
        "type": "text", "text": "\nAnswer the question now in one or two sentences."
    }]
    # Keep Qwen's native empty-think suffix for the unadapted base model.
    inputs = {key: value.to(runtime.device) if hasattr(value, "to") else value
              for key, value in runtime.assemble_turn(answer_messages, vision).items()}
    with runtime._cached_image_features(vision), torch.inference_mode():
        output = runtime.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
    return runtime.processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True,
                        help="Saved step directory with controller_adapter/ and pointer_head.pt")
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--benchmark", choices=("web", "ego"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-margin", type=float, default=0.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-model", choices=("promoted", "base"), default="promoted",
                        help="Adapter used for Response text; base disables all LoRA adapters")
    args = parser.parse_args()
    if bool(args.validation_manifest) == bool(args.benchmark):
        parser.error("specify exactly one of --validation-manifest or --benchmark")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    contract = validate_promoted_contract(SG)
    controller_adapter = args.controller / "controller_adapter/adapter_model.safetensors"
    pointer_path = args.controller / "pointer_head.pt"
    if not controller_adapter.is_file() or not pointer_path.is_file():
        raise FileNotFoundError("controller checkpoint is incomplete")
    summary_path = args.controller.parent / "summary.rank0.json"
    if summary_path.is_file():
        controller_mode = json.loads(summary_path.read_text())["mode"]
    else:
        controller_mode = torch.load(
            args.controller / "training_state.pt", map_location="cpu", weights_only=False
        )["mode"]
    response_blind = controller_mode == "J2"
    if args.validation_manifest:
        rows = read_jsonl(args.validation_manifest)
        split = "validation"
        source_hash = digest(args.validation_manifest)
        source_fps = 0
    else:
        source = SG / f"datasets/proactive-videoqa-{args.benchmark}/anno.json"
        video_root = SG / f"datasets/proactive-videoqa-{args.benchmark}/videos"
        rows, annotations = web.load_examples(source, video_root)
        expected = 500 if args.benchmark == "web" else 326
        if len(rows) != expected or len(annotations) != expected:
            raise ValueError(f"incomplete benchmark {args.benchmark}: {len(rows)}")
        split = args.benchmark
        source_hash = digest(source)
        source_fps = 2 if split == "web" else 6
    if args.max_videos:
        rows = rows[:args.max_videos]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "split": split, "rows": len(rows), "source_sha256": source_hash,
        "controller_adapter_sha256": digest(controller_adapter),
        "pointer_head_sha256": digest(pointer_path),
        "answer_model": args.answer_model,
        "answer_prompt": "base-native-v2" if args.answer_model == "base" else "pace-forced-response",
        "original_generator_adapter_sha256": (digest(Path(contract["adapter"]) / "adapter_model.safetensors")
                                              if args.answer_model == "promoted" else None),
        "response_margin": args.response_margin, "max_new_tokens": args.max_new_tokens,
        "window_frames": 32, "memory_mode": "last-response", "memory_instruction": "v2",
        "memory_select": "aged-out", "prefix": "direct", "history": "self-generated",
        "question_ids": [row["question_id"] for row in rows],
        "controller_history": "response-blind" if response_blind else "j1-original",
    }
    protocol_path = args.output_dir / "manifest.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("existing rollout protocol differs")
    if rank == 0:
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    shard = args.output_dir / f"predictions.rank{rank:02d}.jsonl"
    completed = {row["question_id"] for row in read_jsonl(shard)} if shard.is_file() else set()
    assigned = [row for index, row in enumerate(rows) if index % world == rank
                and row["question_id"] not in completed]
    if not assigned:
        return
    runtime = QwenRuntime.load(
        checkpoint=str(args.controller / "controller_adapter"),
        model_name=contract["model_name"], model_revision=contract["model_revision"],
        frame_num_tokens=144, generation_prefix_mode="direct",
    )
    if args.answer_model == "promoted":
        runtime.model.load_adapter(contract["adapter"], adapter_name="generator", is_trainable=False)
    runtime.model.set_adapter("default")
    runtime.model.eval()
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(pointer_path, map_location="cpu", weights_only=True))
    head.eval()
    counts: Counter[str] = Counter()
    decision_seconds = 0.0
    generation_seconds = 0.0

    def choose_turn(messages, *, max_new_tokens, vision=None):
        nonlocal decision_seconds, generation_seconds
        if vision is None:
            raise RuntimeError("vision cache is required")
        runtime.model.set_adapter("default")
        started = time.monotonic()
        inputs, options, decision = decision_inputs(
            runtime, with_options(response_blind_messages(messages) if response_blind else messages), vision
        )
        with torch.no_grad():
            logits = controller_logits(runtime, head, inputs, vision, options, decision,
                                       train_decoder=False)
            logits[ACTIONS.index("Response")] -= args.response_margin
            action = ACTIONS[int(logits.argmax().item())]
        decision_seconds += time.monotonic() - started
        counts[action] += 1
        if action == "Response":
            started = time.monotonic()
            if args.answer_model == "base":
                with runtime.model.disable_adapter():
                    content = base_answer(runtime, messages, vision,
                                          min(max_new_tokens, args.max_new_tokens))
            else:
                runtime.model.set_adapter("generator")
                try:
                    content = forced_response(runtime, messages, vision,
                                              min(max_new_tokens, args.max_new_tokens))
                finally:
                    runtime.model.set_adapter("default")
            generation_seconds += time.monotonic() - started
            if not content or not content.strip():
                raise ValueError(f"empty response from {args.answer_model} answer model")
            return f"{STATE_TAGS['Response']} {content}".strip()
        return STATE_TAGS["Silence" if action == "Silence" else "Standby"]

    runtime.generate_turn = choose_turn
    started = time.monotonic()
    with shard.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(assigned, 1):
            prior_counts = counts.copy()
            prior_decision_seconds = decision_seconds
            prior_generation_seconds = generation_seconds
            if args.validation_manifest:
                indices = [int(value) for value in row["inference_source_indices"]]
                times = [float(value) for value in row["inference_timestamps"]]
                images = runtime._resize(decode_indices(row["resolved_video_path"], indices))
                prepared = {"images": images, "frame_times": times}
                example = {"question_id": row["question_id"], "question": row["question"],
                           "path": row["resolved_video_path"]}
            else:
                example = row
                prepared = web.prepare_frames(row, source_fps=source_fps, frame_cap=None)
            result = web.generate(
                runtime, example, mode="v24", source_fps=source_fps, frame_cap=None,
                v24_system_prompt=SYSTEM_PROMPT, v24_memory_mode="last-response",
                v24_memory_instruction_version="v2", v24_memory_select="aged-out",
                prepared=prepared,
            )
            if args.validation_manifest:
                result["video_id"] = row["video_id"]
            result["jevany_actions"] = {
                action: counts[action] - prior_counts[action] for action in ACTIONS
            }
            result["jevany_decision_seconds"] = decision_seconds - prior_decision_seconds
            result["jevany_generation_seconds"] = generation_seconds - prior_generation_seconds
            if int(result.get("v24_parse", {}).get("malformed", 0)):
                raise ValueError("malformed response emission")
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(f"[rank {rank}] {index}/{len(assigned)} {row['question_id']} "
                  f"fires={len(result['model_response_list'])} elapsed={time.monotonic()-started:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
