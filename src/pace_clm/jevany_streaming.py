"""Candidate-conditioned PACE streaming controller.

Pointer readout follows JevAny (Apache-2.0, SimpleJev/JevAny commit
b273aea682e5b748a2605808e52ccafb7a378924); PACE's native frame and
history topology stays in the Stream-Gemma runtime.
"""

from __future__ import annotations

from pathlib import Path
import re

import torch
import yaml

from .backbone import language_backbone

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("Silence", "Hold", "Response")
DESCRIPTIONS = yaml.safe_load((ROOT / "configs/actions_v1.yaml").read_text())["actions"]
CONTROLLER_SYSTEM = (
    "Choose one streaming action for each timestamped video frame. Use only the "
    "standing request, causal frames, and your prior action records. "
    "A prior Response record tells you when you answered, but does not contain "
    "the answer text. Choose Response when the current frames give sufficient "
    "new evidence to answer the request. Choose Hold when relevant evidence "
    "continues an event already reported. Choose Silence when no response is "
    "warranted. Do not infer answer wording from action records."
)
_ACTION_TAG = {"</Silence>": "Silence", "</Standby>": "Hold", "</Response>": "Response"}
_TIME = re.compile(r"^\d{2,}:\d{2}")
_MEMORY_LINE = re.compile(r"^- At (\d{2,}:\d{2}): ")


def response_blind_messages(messages: list[dict]) -> list[dict]:
    """Give the controller actions and times, never answer payloads."""
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("unexpected controller message prefix")
    converted = []
    for index, message in enumerate(messages):
        copy = dict(message)
        copy["content"] = [dict(part) for part in message["content"]]
        if index == 0:
            copy["content"] = [{"type": "text", "text": CONTROLLER_SYSTEM}]
        elif index == 1:
            original = copy["content"][0]["text"]
            question, separator, block = original.partition("\n\nPrior response memory:\n")
            if not separator:
                raise ValueError("missing prior response memory block")
            if block == "- none":
                action_block = "- none"
            else:
                lines = []
                for line in block.splitlines():
                    match = _MEMORY_LINE.match(line)
                    if not match:
                        raise ValueError("unrecognized prior response memory")
                    lines.append(f"- At {match.group(1)}: </Response>")
                action_block = "\n".join(lines)
            copy["content"][0]["text"] = question + "\n\nPrior response actions:\n" + action_block
        elif message["role"] == "assistant":
            if index < 1 or messages[index - 1]["role"] != "user":
                raise ValueError("assistant action lacks frame timestamp")
            prior_text = messages[index - 1]["content"][0]["text"]
            time_match = _TIME.match(prior_text)
            if not time_match:
                raise ValueError("missing prior frame timestamp")
            emission = copy["content"][0]["text"].strip()
            tag = next((tag for tag in _ACTION_TAG if emission.startswith(tag)), None)
            if tag is None:
                raise ValueError("unrecognized prior action")
            copy["content"] = [{"type": "text", "text": f"{tag} at {time_match.group()}"}]
        converted.append(copy)
    return converted


def decision_messages(row: dict, images: list, *, response_blind: bool = False) -> list[dict]:
    """Keep the PACE system, memory, time and assistant-history contract."""
    from stream_gemma.config import SYSTEM_PROMPT
    from stream_gemma.v24_memory import (
        MemoryRecord, question_for_mode, system_prompt_for_mode,
    )

    memory = MemoryRecord(**row["memory"]) if row.get("memory") else None
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_for_mode(
            SYSTEM_PROMPT, "last-response", "v2")}]},
        {"role": "user", "content": [{"type": "text", "text": question_for_mode(
            row["question"], "last-response", memory)}]},
    ]
    turns = row["turns"]
    if len(turns) != len(images) or not turns:
        raise ValueError("turn/image alignment failed")
    for index, (turn, image) in enumerate(zip(turns, images, strict=True)):
        content = [
            {"type": "text", "text": f'{turn["ts"]} '},
            {"type": "image", "image": image},
        ]
        if index == len(turns) - 1:
            content.append({"type": "text", "text": option_text()})
        messages.append({"role": "user", "content": content})
        if index < len(turns) - 1:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": turn["emit"]},
            ]})
    return response_blind_messages(messages) if response_blind else messages


def option_text() -> str:
    lines = ["\nSelect the streaming action for the current frame:"]
    for action in ACTIONS:
        lines.append(f"[Option {action}] {DESCRIPTIONS[action]} [/Option {action}]")
    lines.append("[Decision]")
    return "\n".join(lines)


def _render_with_expanded_images(runtime, messages: list[dict], vision: list) -> str:
    rendered = runtime.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    parts = rendered.split(runtime.processor.image_token)
    if len(parts) - 1 != len(vision):
        raise ValueError("image placeholder count differs from causal frames")
    return "".join(
        [parts[0], *(fragment for frame, tail in zip(vision, parts[1:], strict=True)
                     for fragment in (runtime.processor.image_token * frame.num_tokens, tail))]
    )


def decision_inputs(runtime, messages: list[dict], vision: list, *,
                    device=None) -> tuple[dict, list[int], int]:
    from stream_gemma.inference.qwen_adapters import remove_empty_think_generation_suffix

    rendered = _render_with_expanded_images(runtime, messages, vision)
    encoded = runtime.tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = encoded["offset_mapping"][0].tolist()
    candidate_positions = []
    for action in ACTIONS:
        needle = f"[/Option {action}]"
        end = rendered.rfind(needle) + len(needle)
        if end < len(needle):
            raise ValueError(f"missing {needle}")
        candidate_positions.append(max(i for i, (start, stop) in enumerate(offsets) if start < end <= stop))
    marker = "[Decision]"
    end = rendered.rfind(marker) + len(marker)
    if end < len(marker):
        raise ValueError("missing decision marker")
    decision_position = max(i for i, (start, stop) in enumerate(offsets) if start < end <= stop)
    inputs = remove_empty_think_generation_suffix(runtime.assemble_turn(messages, vision), runtime.tokenizer)
    if inputs["input_ids"].shape[1] > 0 and not torch.equal(
        inputs["input_ids"][0, :min(decision_position + 1, inputs["input_ids"].shape[1])],
        encoded["input_ids"][0, :min(decision_position + 1, inputs["input_ids"].shape[1])],
    ):
        raise ValueError("offset tokenizer disagrees with PACE processor")
    if decision_position >= inputs["input_ids"].shape[1]:
        raise ValueError("decision marker was removed with generation suffix")
    if any(pos >= decision_position for pos in candidate_positions):
        raise ValueError("candidate is not causal to decision marker")
    target = runtime.device if device is None else device
    return ({key: value.to(target) if hasattr(value, "to") else value
             for key, value in inputs.items()}, candidate_positions, decision_position)


class PointerHead(torch.nn.Module):
    """JevAny's q/k pointer readout, without modifying its head semantics."""

    def __init__(self, hidden_size: int, pointer_width: int = 256):
        super().__init__()
        self.q = torch.nn.Linear(hidden_size, pointer_width)
        self.k = torch.nn.Linear(hidden_size, pointer_width)
        self.scale = pointer_width ** -0.5

    def forward(self, decision: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        return self.k(options).matmul(self.q(decision)) * self.scale


def controller_logits(runtime, head: PointerHead, inputs: dict, vision: list,
                      options: list[int], decision: int, *, train_decoder: bool) -> torch.Tensor:
    backbone = language_backbone(runtime.model)
    with torch.set_grad_enabled(train_decoder):
        with runtime._cached_image_features(vision):
            hidden = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
    return head(hidden[0, decision].float(), hidden[0, options].float())
