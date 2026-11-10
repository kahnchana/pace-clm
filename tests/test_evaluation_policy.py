"""Prevent manual offsets from silently re-entering controller evaluation."""
import pytest

from pace_clm.evaluation_policy import require_raw_action_scores


@pytest.mark.parametrize('margin', [0.0, -0.0])
def test_raw_scores_allowed(margin):
    require_raw_action_scores(margin)


@pytest.mark.parametrize('margin', [1.0, -1.0, 0.001, float('nan'), float('inf')])
def test_offsets_and_invalid_values_rejected(margin):
    with pytest.raises(ValueError, match='raw learned action scores'):
        require_raw_action_scores(margin)
