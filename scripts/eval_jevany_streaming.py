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
from stream_gemma.inference.closed_api import VertexGeminiRuntime
from zs_infer.adapters import web

from pace_clm.experiment import decode_indices
from pace_clm.jevany_streaming import (
    ACTIONS, PointerHead, controller_logits, decision_inputs, option_text,
    response_blind_messages,
)
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from pace_clm.qwen_controller_runtime import MODEL_REVISIONS, load_dense_controller


GEMINI_BASE_PROMPT = (
    "Answer the standing video request using only frames observed through the "
    "current timestamp. Give one concise factual sentence in plain text. "
    "Do not emit streaming control tags."
)
GEMINI_PROMPT_VARIANTS = {
    "baseline": "",
    "event_boundary": (
        "Treat an uninterrupted action as one event. A new frame, camera angle, "
        "small movement, or incremental progress is not a new action. If the "
        "current action has already been described and is merely continuing, "
        "repeat the earlier sentence exactly. Describe a new action only when "
        "it begins or resumes after stopping."
    ),
    "no_premature_absence": (
        "Report only an action positively visible in the observed frames. "
        "Do not say that an action has not happened yet, predict a later action, "
        "or describe preparation for an action as the action itself. If the "
        "requested action is not visible, answer exactly: 'The requested "
        "action is not visible in the observed frames.' Keep that sentence "
        "identical on later calls until evidence appears."
    ),
    "stable_action_words": (
        "Name the requested action in one short actor-verb-object clause. For "
        "the same action, keep the same verb and object words in every answer. "
        "Do not add tool, location, or progress details unless they materially "
        "change the answer to the user's question. Do not paraphrase an earlier "
        "answer just to describe the same action."
    ),
    "current_event_only": (
        "Answer only with the question-relevant action visible at the current "
        "timestamp. Do not recap earlier actions, combine past and current "
        "actions into a running summary, or mention a background process unless "
        "it is the requested new action. Use one plain sentence."
    ),
}


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


def gemini_answer(provider: VertexGeminiRuntime, messages: list[dict],
                  max_new_tokens: int, prompt_variant: str = "baseline") -> str:
    """Generate one answer from the same causal frame window as the controller."""
    addition = GEMINI_PROMPT_VARIANTS[prompt_variant]
    system_text = GEMINI_BASE_PROMPT + ("\n\n" + addition if addition else "")
    request = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for index, message in enumerate(messages[1:]):
        if message["role"] == "assistant":
            emitted = str(message["content"][0]["text"])
            if emitted.startswith(STATE_TAGS["Response"]):
                request.append({"role": "user", "content": [{"type": "text", "text":
                    f"Earlier answer: {emitted[len(STATE_TAGS['Response']):].strip()}"}]})
        else:
            request.append(message)
    request.append({"role": "user", "content": [{"type": "text", "text":
        "Answer the request for the current frame now."}]})
    return provider.generate_messages(request, max_new_tokens=max_new_tokens).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True,
                        help="Saved step directory with controller_adapter/ and pointer_head.pt")
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--benchmark", choices=("web", "ego"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-margin", type=float, default=0.0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--selection-ids", type=Path,
                        help="JSON array of benchmark question IDs to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-model", choices=("promoted", "base", "gemini-3.8-flash"),
                        default="promoted", help="Generator used for Response text")
    parser.add_argument("--gemini-prompt-variant", choices=tuple(GEMINI_PROMPT_VARIANTS),
                        default="baseline")
    parser.add_argument("--record-action-logits", action="store_true",
                        help="Store per-decision raw action logits in each rollout row")
    args = parser.parse_args()
    from pace_clm.evaluation_policy import require_raw_action_scores
    require_raw_action_scores(args.response_margin)
    if args.gemini_prompt_variant != "baseline" and args.answer_model != "gemini-3.8-flash":
        parser.error("Gemini prompt variants require --answer-model gemini-3.8-flash")
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
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    controller_mode = summary.get("mode")
    if controller_mode is None:
        controller_mode = torch.load(
            args.controller / "training_state.pt", map_location="cpu", weights_only=False
        )["mode"]
    identity_path = args.controller / "model_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    separate_generator = identity.get("model_name") in MODEL_REVISIONS
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
    if args.selection_ids:
        selected_ids = json.loads(args.selection_ids.read_text())
        if not isinstance(selected_ids, list) or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selection IDs must be a unique JSON array")
        by_id = {row["question_id"]: row for row in rows}
        if set(selected_ids) - by_id.keys():
            raise ValueError("selection contains IDs absent from benchmark")
        rows = [by_id[question_id] for question_id in selected_ids]
    if args.max_videos:
        rows = rows[:args.max_videos]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "split": split, "rows": len(rows), "source_sha256": source_hash,
        "controller_adapter_sha256": digest(controller_adapter),
        "pointer_head_sha256": digest(pointer_path),
        "answer_model": args.answer_model,
        "answer_location": "global" if args.answer_model == "gemini-3.8-flash" else None,
        "answer_prompt": ("base-native-v2" if args.answer_model == "base" else
                          "gemini-causal-window-v1" if args.answer_model == "gemini-3.8-flash"
                          else "pace-forced-response"),
        "original_generator_adapter_sha256": (digest(Path(contract["adapter"]) / "adapter_model.safetensors")
                                              if args.answer_model == "promoted" else None),
        "response_margin": args.response_margin, "max_new_tokens": args.max_new_tokens,
        "window_frames": 32, "memory_mode": "last-response", "memory_instruction": "v2",
        "memory_select": "aged-out", "prefix": "direct", "history": "self-generated",
        "question_ids": [row["question_id"] for row in rows],
        "controller_history": "response-blind" if response_blind else "j1-original",
        "controller_model": identity or {"model_name": contract["model_name"],
                                          "model_revision": contract["model_revision"]},
    }
    if args.answer_model == "gemini-3.8-flash":
        protocol["empty_answer_policy"] = "retry-with-256-visible-tokens-then-explicit-unavailable-v1"
        if args.gemini_prompt_variant != "baseline":
            protocol["gemini_prompt_variant"] = args.gemini_prompt_variant
            protocol["gemini_prompt_addition"] = GEMINI_PROMPT_VARIANTS[args.gemini_prompt_variant]
            protocol["answer_prompt"] = "gemini-causal-window-v1-" + args.gemini_prompt_variant
    if args.record_action_logits:
        protocol["record_action_logits"] = True
    protocol_path = args.output_dir / "manifest.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("existing rollout protocol differs")
    if rank == 0:
        temporary_protocol_path = protocol_path.with_suffix(".json.tmp")
        temporary_protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        temporary_protocol_path.replace(protocol_path)
    shard = args.output_dir / f"predictions.rank{rank:02d}.jsonl"
    completed = {row["question_id"] for row in read_jsonl(shard)} if shard.is_file() else set()
    assigned = [row for index, row in enumerate(rows) if index % world == rank
                and row["question_id"] not in completed]
    if not assigned:
        return
    if separate_generator:
        runtime, _ = load_dense_controller(
            model_name=identity["model_name"], model_revision=identity["model_revision"],
            checkpoint=str(args.controller / "controller_adapter"),
        )
    else:
        runtime = QwenRuntime.load(
            checkpoint=str(args.controller / "controller_adapter"),
            model_name=contract["model_name"], model_revision=contract["model_revision"],
            frame_num_tokens=144, generation_prefix_mode="direct",
        )
    generator = (QwenRuntime.load(
        checkpoint=contract["adapter"], model_name=contract["model_name"],
        model_revision=contract["model_revision"], frame_num_tokens=144,
        generation_prefix_mode="direct",
    ) if separate_generator and args.answer_model == "promoted" else runtime)
    if args.answer_model == "promoted" and not separate_generator:
        runtime.model.load_adapter(contract["adapter"], adapter_name="generator", is_trainable=False)
    provider = (VertexGeminiRuntime.load(model_name="gemini-3.8-flash", location="global")
                if args.answer_model == "gemini-3.8-flash" else None)
    runtime.model.set_adapter("default")
    runtime.model.eval()
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(pointer_path, map_location="cpu", weights_only=True))
    head.eval()
    counts: Counter[str] = Counter()
    decision_seconds = 0.0
    generation_seconds = 0.0
    action_trace: list[dict] = []
    provider_trace: list[dict] = []
    generator_frames: dict[int, tuple[object, object]] = {}

    def choose_turn(messages, *, max_new_tokens, vision=None):
        nonlocal decision_seconds, generation_seconds
        if vision is None:
            raise RuntimeError("vision cache is required")
        if not separate_generator:
            runtime.model.set_adapter("default")
        started = time.monotonic()
        inputs, options, decision = decision_inputs(
            runtime, with_options(response_blind_messages(messages) if response_blind else messages), vision
        )
        with torch.no_grad():
            logits = controller_logits(runtime, head, inputs, vision, options, decision,
                                       train_decoder=False)
            if args.record_action_logits:
                action_trace.append({"logits": [float(x) for x in logits.detach().cpu().tolist()],
                                     "action": ACTIONS[int(logits.argmax().item())],
                                     "frame_index": len(action_trace),
                                     "frame_time_text": messages[-1]['content'][0].get('text', ''),
                                     "history_turns": sum(m['role'] == 'assistant' for m in messages),
                                     "memory_question": messages[1]['content'][0].get('text', '')})
            action = ACTIONS[int(logits.argmax().item())]
        decision_seconds += time.monotonic() - started
        counts[action] += 1
        if action == "Response":
            started = time.monotonic()
            if args.answer_model == "base":
                with runtime.model.disable_adapter():
                    content = base_answer(runtime, messages, vision,
                                          min(max_new_tokens, args.max_new_tokens))
            elif provider is not None:
                content = gemini_answer(provider, messages,
                                        min(max_new_tokens, args.max_new_tokens),
                                        args.gemini_prompt_variant)
                call = provider.last_call
                provider_trace.append({
                    "status": call.get("status"),
                    "attempts": call.get("attempts"),
                    "errors": call.get("errors"),
                    "elapsed_seconds": call.get("elapsed_seconds"),
                    "model_version": (call.get("response") or {}).get("modelVersion"),
                    "usage_metadata": (call.get("response") or {}).get("usageMetadata"),
                })
                if not content:
                    if call.get("status") != 200:
                        raise RuntimeError(f"Gemini generation failed: {call.get('errors')}")
                    content = gemini_answer(provider, messages, 256,
                                            args.gemini_prompt_variant)
                    call = provider.last_call
                    provider_trace.append({
                        "status": call.get("status"),
                        "attempts": call.get("attempts"),
                        "errors": call.get("errors"),
                        "elapsed_seconds": call.get("elapsed_seconds"),
                        "model_version": (call.get("response") or {}).get("modelVersion"),
                        "usage_metadata": (call.get("response") or {}).get("usageMetadata"),
                        "empty_answer_retry": True,
                    })
                    if not content:
                        if call.get("status") != 200:
                            raise RuntimeError(f"Gemini retry failed: {call.get('errors')}")
                        content = "I cannot determine the answer from the visible frames."
                        provider_trace[-1]["explicit_unavailable_fallback"] = True
            else:
                if separate_generator:
                    images = [part["image"] for message in messages
                              for part in message["content"] if part.get("type") == "image"]
                    missing_images = [image for image in images if id(image) not in generator_frames]
                    if missing_images:
                        prepared_frames = [generator.encode_frame(image)
                                           for image in generator._resize(missing_images)]
                        generator.encode_frames_features(prepared_frames)
                        generator_frames.update((id(image), (image, frame))
                                                for image, frame in zip(missing_images, prepared_frames, strict=True))
                    generator_vision = [generator_frames[id(image)][1] for image in images]
                    content = forced_response(generator, messages, generator_vision,
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
            action_trace.clear()
            provider_trace.clear()
            generator_frames.clear()
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
            result["jevany_peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(runtime.device) / 1024**3
            if provider is not None:
                result["gemini_calls"] = list(provider_trace)
            if args.record_action_logits:
                result["jevany_action_trace"] = list(action_trace)
            if int(result.get("v24_parse", {}).get("malformed", 0)):
                raise ValueError("malformed response emission")
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(f"[rank {rank}] {index}/{len(assigned)} {row['question_id']} "
                  f"fires={len(result['model_response_list'])} elapsed={time.monotonic()-started:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
