"""Frozen PACE-Qwen feature extraction without applying the vocabulary head."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .pooling import pool_final_language


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    model.requires_grad_(False)
    model.eval()
    return model


def assert_frozen(model: torch.nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"backbone contains trainable parameters: {trainable[:5]}")
    if model.training:
        raise ValueError("backbone must be in eval mode")


def hidden_size_from_config(model: Any) -> int:
    config = model.config
    text_config = getattr(config, "text_config", None)
    size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
    if not size:
        raise ValueError("model configuration does not expose text hidden_size")
    return int(size)


def language_backbone(model: Any) -> Any:
    """Resolve the multimodal base module below PEFT and above the LM head."""
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        model = getter()
    module = getattr(model, "model", None)
    if module is None:
        raise ValueError("expected a conditional-generation model with a `.model` backbone")
    return module


class FrozenFeatureExtractor:
    def __init__(self, model: torch.nn.Module, *, vision_token_ids: Iterable[int] = ()) -> None:
        self.model = freeze_backbone(model)
        self.backbone = language_backbone(model)
        self.vision_token_ids = tuple(int(value) for value in vision_token_ids)
        self.hidden_size = hidden_size_from_config(model)

    def extract(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        assert_frozen(self.model)
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise ValueError("inputs require input_ids and attention_mask")
        forward_inputs = dict(inputs)
        forward_inputs.update(use_cache=False, return_dict=True)
        with torch.inference_mode():
            output = self.backbone(**forward_inputs)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output[0]
            pooled = pool_final_language(
                hidden,
                inputs["input_ids"],
                inputs["attention_mask"],
                vision_token_ids=self.vision_token_ids,
            )
        if pooled.shape[-1] != self.hidden_size:
            raise ValueError(
                f"pooled width {pooled.shape[-1]} != configured hidden_size {self.hidden_size}"
            )
        return pooled
