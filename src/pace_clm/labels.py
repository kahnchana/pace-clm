"""Canonical paper-facing actions and the Stream-Gemma label boundary."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    Response = 0
    Silence = 1
    Hold = 2


STREAM_GEMMA_TO_ACTION = {
    "Response": Action.Response,
    "Silence": Action.Silence,
    # Stream-Gemma's Standby marks relevant/in-progress event frames on which
    # another answer is not warranted. PACE-CLM calls this paper-facing Hold.
    "Standby": Action.Hold,
}


def to_action(label: str) -> Action:
    try:
        return STREAM_GEMMA_TO_ACTION[label]
    except KeyError:
        raise ValueError(f"unknown Stream-Gemma state: {label!r}") from None


def action_names() -> tuple[str, ...]:
    return tuple(action.name for action in Action)
