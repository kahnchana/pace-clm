import pytest

from pace_clm.labels import STREAM_GEMMA_TO_ACTION, Action, to_action


def test_all_three_stream_gemma_states_map_to_paper_actions():
    assert STREAM_GEMMA_TO_ACTION == {
        "Response": Action.Response,
        "Silence": Action.Silence,
        "Standby": Action.Hold,
    }


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError, match="unknown Stream-Gemma state"):
        to_action("Hold")
