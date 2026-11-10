import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]/'scripts/run_jevany_rl_full_current_node.py'
spec = importlib.util.spec_from_file_location('rl_full_schedule', MODULE)
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


def test_evaluate_16_before_training_past_16():
    assert schedule.pending_actions(1, set()) == [('train',16),('web',16),('train',32),('web',32)]


def test_restart_does_not_skip_pending_web_checkpoint():
    assert schedule.pending_actions(20, set()) == [('web',16),('train',32),('web',32)]
    assert schedule.pending_actions(32, {16}) == [('web',32)]
    assert schedule.pending_actions(32, {16,32}) == []


def test_eval_view_preserves_training_checkpoint(tmp_path):
    checkpoint = tmp_path/'step-000016'
    checkpoint.mkdir()
    (checkpoint/'controller_adapter').mkdir()
    (checkpoint/'pointer_head.pt').write_bytes(b'head')
    (checkpoint/'model_identity.json').write_text('{}')
    (checkpoint/'training_state.pt').write_bytes(b'optimizer')
    view = schedule.checkpoint_view(tmp_path, checkpoint)
    assert view.parent.joinpath('summary.rank0.json').read_text().find('"J1"') >= 0
    assert view.joinpath('pointer_head.pt').resolve() == checkpoint/'pointer_head.pt'
    assert not view.joinpath('training_state.pt').exists()
    assert checkpoint.joinpath('training_state.pt').read_bytes() == b'optimizer'
    assert schedule.checkpoint_view(tmp_path, checkpoint) == view
