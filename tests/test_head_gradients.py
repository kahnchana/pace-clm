import torch
from torch.nn import functional as F

from pace_clm.heads import ContrastiveActionModel


def test_both_heads_and_scale_receive_gradients_with_cached_base_actions():
    model = ContrastiveActionModel(input_dim=8, hidden_dim=12, output_dim=4)
    states = torch.randn(6, 8)
    cached_base_actions = torch.randn(3, 8)
    logits = model(states, cached_base_actions)
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1, 2])).backward()
    assert model.logit_scale.grad is not None
    assert all(parameter.grad is not None for parameter in model.state_head.parameters())
    assert all(parameter.grad is not None for parameter in model.action_head.parameters())
    assert cached_base_actions.grad is None
