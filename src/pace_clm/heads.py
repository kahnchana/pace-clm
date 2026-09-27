"""Independent CLM-style state/action projections and three-action logits."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1536,
        output_dim: int = 512,
        depth: int = 3,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        for index in range(depth):
            in_dim = input_dim if index == 0 else hidden_dim
            out_dim = output_dim if index == depth - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if index != depth - 1:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        return F.normalize(self.network(features), dim=-1)


class ContrastiveActionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1536,
        output_dim: int = 512,
        depth: int = 3,
        logit_scale_init: float = math.log(1 / 0.07),
    ) -> None:
        super().__init__()
        self.state_head = ProjectionHead(input_dim, hidden_dim, output_dim, depth)
        self.action_head = ProjectionHead(input_dim, hidden_dim, output_dim, depth)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * self.state_head(states) @ self.action_head(actions).T
