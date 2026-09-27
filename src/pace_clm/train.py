"""Training primitives for the three-candidate state-to-action MVP."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def three_action_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("expected logits shaped [batch, 3]")
    return F.cross_entropy(logits, targets)
