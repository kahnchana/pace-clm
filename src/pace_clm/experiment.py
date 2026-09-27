"""PACE-200 experiment data and frozen-feature extraction helpers."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image

from .backbone import FrozenFeatureExtractor
from .labels import Action, to_action


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def annotation_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def state_timeline(
    record: dict[str, Any], metadata: dict[str, Any], *, keep_trailing: bool = True
) -> dict[str, Any]:
    """Build the exact Stream-Gemma event-interior state timeline."""
    from stream_gemma.data import _conversation_and_range, _parse_record
    from stream_gemma.episode import flatten_stream_conversation, paint_v24_states

    relative, question, text_stream = _parse_record(record)
    video_id = str(Path(relative).with_suffix(""))
    meta = metadata[video_id]
    built = _conversation_and_range(
        question=question,
        text_stream=text_stream,
        metadata=meta,
        source_fps=1,
        keep_trailing_silence=keep_trailing,
    )
    if built is None:
        raise ValueError(f"could not construct timeline for {video_id}")
    conversation, source_range = built
    emits = flatten_stream_conversation(conversation)
    painted = paint_v24_states(emits, standby_policy="event_interior")
    return {
        "video_id": video_id,
        "question": question,
        "video_path": meta.get("path") or "",
        "source_start": source_range.start,
        "source_stop": source_range.stop,
        "emits": emits,
        "states": painted.states,
    }


def absolute_timeline(timeline: dict[str, Any], frame_count: int) -> tuple[list[str], list[str]]:
    emits = ["silent"] * frame_count
    states = ["Silence"] * frame_count
    start = int(timeline["source_start"])
    for local, (emit, state) in enumerate(zip(timeline["emits"], timeline["states"], strict=True)):
        absolute = start + local
        if 0 <= absolute < frame_count:
            emits[absolute] = emit
            states[absolute] = state
    return emits, states


def build_balanced_train_manifest(
    *,
    annotation_path: str | Path,
    metadata_path: str | Path,
    validation_video_ids: set[str],
    output_path: str | Path,
    per_class: int = 256,
    record_stride: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    records = annotation_records(annotation_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("Response", "Silence", "Hold")
    }
    natural = Counter()
    videos = set()
    failures = 0
    for raw_index in range(0, len(records), record_stride):
        try:
            timeline = state_timeline(records[raw_index], metadata)
        except (KeyError, TypeError, ValueError, StopIteration):
            failures += 1
            continue
        video_id = timeline["video_id"]
        if video_id in validation_video_ids:
            raise RuntimeError(f"training/validation video leakage: {video_id}")
        videos.add(video_id)
        for local_frame, stream_state in enumerate(timeline["states"]):
            action = to_action(stream_state).name
            natural[action] += 1
            candidates[action].append(
                {
                    "split": "train",
                    "raw_index": raw_index,
                    "video_id": video_id,
                    "frame_index": int(timeline["source_start"]) + local_frame,
                    "gold_action": action,
                    "gold_action_id": int(Action[action]),
                }
            )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for action in ("Response", "Silence", "Hold"):
        pool = candidates[action]
        if len(pool) < per_class:
            raise RuntimeError(f"only {len(pool)} {action} candidates; need {per_class}")
        selected.extend(rng.sample(pool, per_class))
    selected.sort(key=lambda row: (row["video_id"], row["frame_index"], row["gold_action_id"]))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    return {
        "seed": seed,
        "record_stride": record_stride,
        "records_scanned": math.ceil(len(records) / record_stride),
        "timeline_failures": failures,
        "videos_scanned": len(videos),
        "natural_distribution": dict(natural),
        "candidate_distribution": {key: len(value) for key, value in candidates.items()},
        "selected_distribution": dict(Counter(row["gold_action"] for row in selected)),
        "examples": len(selected),
        "unique_videos": len({row["video_id"] for row in selected}),
    }


def decode_indices(path: str, indices: list[int]) -> list[Image.Image]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    images = []
    try:
        previous = -2
        for index in indices:
            if index != previous + 1 and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"could not seek {path} to {index}")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode {path} at {index}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            previous = index
    finally:
        capture.release()
    return images


def direct_inputs(runtime, messages: list[dict], vision=None) -> dict[str, Any]:
    from stream_gemma.inference.qwen_adapters import remove_empty_think_generation_suffix

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


def gold_messages(
    *,
    question: str,
    images: list[Image.Image],
    emits: list[str],
    states: list[str],
    index: int,
    max_rounds: int = 32,
) -> tuple[list[dict], int]:
    """Exact direct-prefix messages with a common gold causal history."""
    from stream_gemma.config import SYSTEM_PROMPT
    from stream_gemma.episode import STATE_TAGS, format_frame_time
    from stream_gemma.v24_memory import (
        MemoryRecord,
        question_for_mode,
        select_last_response_memory,
        system_prompt_for_mode,
    )

    start = max(0, index - max_rounds + 1)
    history = [
        MemoryRecord(frame_index=i, time_seconds=float(i), content=emits[i])
        for i in range(index)
        if states[i] == "Response" and emits[i] != "silent"
    ]
    record = select_last_response_memory(history, start, select="aged-out", before_frame=index)
    system = system_prompt_for_mode(SYSTEM_PROMPT, "last-response", "v2")
    turn_question = question_for_mode(question, "last-response", record)
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": turn_question}]},
    ]
    for prior in range(start, index):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{format_frame_time(prior)} "},
                    {"type": "image", "image": images[prior]},
                ],
            }
        )
        emit = STATE_TAGS[states[prior]]
        if states[prior] == "Response" and emits[prior] != "silent":
            emit = f"{emit} {emits[prior]}"
        messages.append({"role": "assistant", "content": [{"type": "text", "text": emit}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{format_frame_time(index)} "},
                {"type": "image", "image": images[index]},
            ],
        }
    )
    return messages, start


def extract_one(runtime, extractor: FrozenFeatureExtractor, messages, vision) -> torch.Tensor:
    inputs = direct_inputs(runtime, messages, vision)
    with runtime._cached_image_features(vision):
        return extractor.extract(inputs).float().cpu()
