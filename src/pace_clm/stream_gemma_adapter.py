"""Narrow, attributed adapter to the Stream-Gemma source-of-truth contracts.

The message topology below follows `stream_v25_qwen` in
`stream_gemma/inference/qwen_stream.py`. Benchmark loading and scoring are not
copied: callers must invoke Stream-Gemma for those operations.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotedQwenContract:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    model_revision: str = "59d61f3ce65a6d9863b86d2e96597125219dc754"
    adapter_relative_path: str = "outputs/lora-qwen-w1-long-neg"
    target_fps: int = 1
    window_frames: int = 32
    frame_num_tokens: int = 144
    memory_mode: str = "last-response"
    memory_instruction_version: str = "v2"
    memory_k: int = 1
    qwen_prefix_mode: str = "direct"


def connect_stream_gemma(root: str | Path) -> None:
    root = Path(root).resolve()
    source = root / "src"
    if not (source / "stream_gemma" / "__init__.py").is_file():
        raise FileNotFoundError(f"not a Stream-Gemma source tree: {root}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def validate_promoted_contract(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    connect_stream_gemma(root)
    config = importlib.import_module("stream_gemma.config")
    expected = PromotedQwenContract()
    observed = {
        "model_name": config.QWEN_MODEL_NAME,
        "model_revision": config.QWEN_MODEL_REVISION,
        "adapter": str(root / expected.adapter_relative_path),
    }
    if observed["model_name"] != expected.model_name:
        raise RuntimeError(f"unexpected Qwen model: {observed['model_name']}")
    if observed["model_revision"] != expected.model_revision:
        raise RuntimeError(f"unexpected Qwen revision: {observed['model_revision']}")
    adapter = Path(observed["adapter"])
    for name in ("adapter_config.json", "adapter_model.safetensors", "run_manifest.json"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    return observed


def build_causal_messages(
    *,
    stream_gemma_root: str | Path,
    question: str,
    system_prompt: str,
    images: list[Any],
    emitted: list[str],
    current_index: int,
    frame_times: list[float],
    memory_record: Any = None,
    subtitles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build one strictly causal turn with the promoted 32-frame topology."""
    connect_stream_gemma(stream_gemma_root)
    episode = importlib.import_module("stream_gemma.episode")
    memory = importlib.import_module("stream_gemma.v24_memory")
    contract = PromotedQwenContract()
    if not 0 <= current_index < len(images):
        raise IndexError("current_index outside images")
    if len(frame_times) != len(images):
        raise ValueError("frame_times must align with images")
    if len(emitted) < current_index:
        raise ValueError("one prior emission is required for every prior frame")
    if subtitles is not None and len(subtitles) != len(images):
        raise ValueError("subtitles must align with images")

    window_start = max(0, current_index - contract.window_frames + 1)
    turn_system = memory.system_prompt_for_mode(
        system_prompt, contract.memory_mode, contract.memory_instruction_version
    )
    turn_question = memory.question_for_mode(question, contract.memory_mode, memory_record)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": turn_system}]},
        {"role": "user", "content": [{"type": "text", "text": turn_question}]},
    ]
    for prior in range(window_start, current_index):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{episode.format_frame_time(frame_times[prior])} ",
                        },
                        {"type": "image", "image": images[prior]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": emitted[prior]}],
                },
            ]
        )
    current_text = f"{episode.format_frame_time(frame_times[current_index])} "
    if subtitles and subtitles[current_index]:
        current_text += f"{subtitles[current_index]}\n"
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": current_text},
                {"type": "image", "image": images[current_index]},
            ],
        }
    )
    return messages
