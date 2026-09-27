"""Dataset boundary declarations.

PACE examples must be obtained through Stream-Gemma; this module intentionally
does not reimplement its parsers or state painter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .labels import Action


@dataclass(frozen=True)
class FeatureExample:
    video_id: str
    frame_index: int
    action: Action
