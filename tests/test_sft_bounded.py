"""Ensure bounded diagnostics cannot silently expand into new training seeds."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_qwen35_sft_bounded import BoundedCampaign, positive_signal


def test_budget_signal_requires_size_and_positive_paired_interval():
    assert positive_signal(dict(delta_final_minus_earlier=2., ci95=[.2, 3.]))
    assert not positive_signal(dict(delta_final_minus_earlier=2., ci95=[-.2, 3.]))
    assert not positive_signal(dict(delta_final_minus_earlier=.5, ci95=[.1, .9]))
    with pytest.raises(ValueError, match='invalid paired'):
        positive_signal(dict(delta_final_minus_earlier=float('nan'), ci95=[.1, .9]))


class FakeCampaign(BoundedCampaign):
    def __init__(self, output, promising):
        self.out = output
        (output / 'reports').mkdir()
        self.results_path = output / 'reports/results.json'
        self.results_path.write_text('{}')
        self.specs = {name: {} for name in ('anchor_b32', 'fresh_scale', 'replay_scale', 'rollout_control', 'rollout_corrected')}
        self.promising = promising
        self.trained = []
        self.evaluated = []

    def train(self, name, spec, seed=42):
        self.trained.append((name, seed))
        return self.out / name

    def evaluate(self, name, checkpoint, cohort):
        self.evaluated.append((name, cohort))
        self.results[name + '/' + cohort] = dict(name=name, cohort=cohort)

    def corrections(self, parent):
        pass

    def compare(self, control, candidate, cohort):
        if self.promising and candidate == 'fresh_scale':
            return dict(delta_final_minus_earlier=2., ci95=[.2, 3.])
        return dict(delta_final_minus_earlier=.1, ci95=[-1., 1.])

    def report(self):
        pass

    def status(self, *args, **kwargs):
        pass


def test_null_diagnostics_finish_without_confirmation_or_extra_seeds(tmp_path):
    campaign = FakeCampaign(tmp_path, promising=False)
    campaign.finish_comparisons()
    assert {name for name, _ in campaign.trained} == {
        'anchor_b32', 'fresh_scale', 'replay_scale', 'rollout_control', 'rollout_corrected'}
    assert {seed for _, seed in campaign.trained} == {42}
    assert not any(cohort == 'confirmation' for _, cohort in campaign.evaluated)
    result = json.loads((tmp_path / 'reports/bounded_decision.json').read_text())
    assert result['recipes_eligible_for_replication_review'] == []
    assert result['automatic_seed_replications'] is False


def test_promising_pair_confirms_both_arms_but_still_defers_replication(tmp_path):
    campaign = FakeCampaign(tmp_path, promising=True)
    campaign.finish_comparisons()
    confirmed = {name for name, cohort in campaign.evaluated if cohort == 'confirmation'}
    assert confirmed == {'historical_baseline', 'replay_scale', 'fresh_scale'}
    assert {seed for _, seed in campaign.trained} == {42}
    result = json.loads((tmp_path / 'reports/bounded_decision.json').read_text())
    assert result['recipes_eligible_for_replication_review'] == ['fresh_scale']
    assert result['automatic_seed_replications'] is False
