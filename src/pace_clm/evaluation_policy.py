"""User-required inference contract: decisions come from learned scores only."""
from __future__ import annotations

import math


def require_raw_action_scores(response_margin: float) -> None:
    """Reject manually chosen decision offsets, including cached evaluations."""
    if not math.isfinite(response_margin) or response_margin != 0.0:
        raise ValueError(
            "Manual response margins are disabled by the user's 2026-11-08 "
            "instruction. Use the raw learned action scores (response margin 0); "
            "learn decision behavior through training."
        )
