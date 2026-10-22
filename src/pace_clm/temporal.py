"""Shared causal GRU adapter for the matched temporal head pair."""

from __future__ import annotations

import torch
from torch import nn


class CausalTemporalAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_width, batch_first=True)
        self.output = nn.Linear(hidden_width, input_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[1] != 4:
            raise ValueError("causal temporal adapter expects four consecutive states")
        result, _ = self.gru(states)
        return self.output(result[:, -1, :])


class TemporalHead(nn.Module):
    def __init__(self, adapter: CausalTemporalAdapter, head: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head

    def forward(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.adapter(states)
        if actions is None:
            return self.head(pooled)
        return self.head(pooled, actions)
