"""Perfect-score adjustments must remain explicit and separate from measurements."""
import pytest
from retry_jevany_flash_web import aggregate_with_imputation


def measured():
    return dict(num_value_adding_fires=3,num_fires=6,num_gold_turns=4,
                pauc_recall_component=50.,num_videos=2)


def test_no_imputation_preserves_measured_harmonic_score():
    score=aggregate_with_imputation(measured(),0,0)
    assert score['pauc_f']==50.
    assert score['imputed_episodes']==0


def test_perfect_episode_adds_explicit_virtual_precision_and_recall():
    source=measured()
    score=aggregate_with_imputation(source,2,1)
    assert score['fire_precision']==62.5
    assert score['temporal_recall']==pytest.approx(100*4/6)
    assert score['pauc_f']==pytest.approx(100*2*(5/8)*(4/6)/((5/8)+(4/6)))
    assert score['headline_eligible'] is False
    assert score['imputed_episodes']==1
    assert source==measured()
