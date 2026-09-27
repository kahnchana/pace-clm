from types import SimpleNamespace

import torch
from torch import nn

from pace_clm.backbone import FrozenFeatureExtractor, assert_frozen


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))


def test_extractor_freezes_eval_and_skips_lm_head():
    model = FakeConditionalGeneration()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("LM head must not run")

    model.lm_head.forward = fail_if_called
    extractor = FrozenFeatureExtractor(model)
    pooled = extractor.extract(
        {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}
    )
    assert pooled.shape == (1, 4)
    assert not pooled.requires_grad
    assert not called
    assert_frozen(model)
