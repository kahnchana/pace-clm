"""Causally safe language-position pooling."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def final_language_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vision_token_ids: Iterable[int] = (),
    require_causal_endpoint: bool = True,
) -> torch.Tensor:
    """Return the final attended non-vision position for every batch row.

    With a decoder-only causal tower, requiring this to also be the last valid
    position proves that the pooled token can attend to every earlier input
    position. This handles both left and right padding.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have identical [batch, seq] shape")
    valid = attention_mask.to(dtype=torch.bool)
    language = valid.clone()
    for token_id in vision_token_ids:
        language &= input_ids.ne(int(token_id))
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    masked = positions.masked_fill(~language, -1)
    selected = masked.max(dim=1).values
    if torch.any(selected < 0):
        raise ValueError("every sample must contain an attended language token")
    if require_causal_endpoint:
        last_valid = positions.masked_fill(~valid, -1).max(dim=1).values
        if not torch.equal(selected, last_valid):
            raise ValueError(
                "final valid position is a vision placeholder; cannot prove full-input causal access"
            )
    return selected


def pool_final_language(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vision_token_ids: Iterable[int] = (),
) -> torch.Tensor:
    if hidden_states.ndim != 3 or hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden_states must be [batch, seq, hidden] aligned to input_ids")
    positions = final_language_positions(
        input_ids, attention_mask, vision_token_ids=vision_token_ids
    )
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions]
