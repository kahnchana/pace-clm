import pytest
import torch

from pace_clm.pooling import final_language_positions, pool_final_language


def test_selects_final_valid_language_token_with_left_and_right_padding():
    ids = torch.tensor([[0, 5, 99, 7], [4, 99, 8, 0]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
    assert final_language_positions(ids, mask, vision_token_ids=[99]).tolist() == [3, 2]
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    assert torch.equal(
        pool_final_language(hidden, ids, mask, vision_token_ids=[99]), hidden[[0, 1], [3, 2]]
    )


def test_rejects_vision_placeholder_at_causal_endpoint():
    with pytest.raises(ValueError, match="vision placeholder"):
        final_language_positions(
            torch.tensor([[4, 5, 99]]), torch.ones(1, 3), vision_token_ids=[99]
        )
