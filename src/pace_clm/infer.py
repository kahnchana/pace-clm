"""Inference selection shared by projection-head and baseline implementations."""

from __future__ import annotations

import torch

from .labels import Action


def select_actions(logits: torch.Tensor) -> list[Action]:
    if logits.ndim != 2 or logits.shape[1] != len(Action):
        raise ValueError("expected logits shaped [batch, 3]")
    return [Action(index) for index in logits.argmax(dim=-1).tolist()]
