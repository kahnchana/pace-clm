from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_jevany_e2_rloo import coverage_decision, training_args
from audit_jevany_rl_reward import compare


def test_coverage_pause_requires_consecutive_checks_and_survives_resume():
    base = {'num_value_adding_fires': 100, 'num_fires': 200}
    def check(useful, fires=200):
        return {'num_value_adding_fires': useful, 'num_fires': fires}
    assert coverage_decision(base, [(8, check(89))]) is None
    assert coverage_decision(base, [(8, check(89)), (16, check(90)), (32, check(89))]) is None
    assert coverage_decision(base, [(8, check(89)), (16, check(80))]) == ('coverage_below_90_percent_twice', 16)
    assert coverage_decision(base, [(8, check(1, 19))]) == ('near_silence', 8)


def test_first_gate_is_counted_in_total_collection_budget(tmp_path):
    initial = training_args(tmp_path, 1)
    assert '--resume' not in initial
    (tmp_path/'latest.json').write_text('{}')
    continuation = training_args(tmp_path, 32)
    assert '--resume' in continuation
    assert continuation[continuation.index('--updates') + 1] == '32'
    assert continuation[continuation.index('--learners') + 1] == '7'
    assert continuation[continuation.index('--reward-version') + 1] == 'pauc-f-aligned-gpt-v1'


def test_rank_comparison_preserves_top_sets_and_ties():
    counts, old_top, new_top = compare([1, 1, 0, 0], [0, 0, 1, 1])
    assert old_top == [0, 1]
    assert new_top == [2, 3]
    assert counts == {'both_tied': 2, 'strict_inversion': 4}
