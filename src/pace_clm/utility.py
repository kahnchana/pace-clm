"""Shared answer-utility prediction branch for E09/E10."""

from __future__ import annotations

import torch
from torch import nn


class UtilityHead(nn.Module):
    def __init__(self, base_head: nn.Module, input_dim: int, hidden_width: int = 256) -> None:
        super().__init__()
        self.base_head = base_head
        self.utility = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if actions is None:
            logits = self.base_head(states)
        else:
            logits = self.base_head(states, actions)
        return logits, self.utility(states).squeeze(-1)
