"""Boundary checks for causal corrective labels and the raw-policy campaign."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from qwen35_sft_campaign import install_training_contexts, write
from prepare_qwen35_sft_corrections import action_target, visible_prior_answers
from run_qwen35_sft_campaign import Campaign


def test_corrective_target_requires_verified_coverage():
    event = {'start': 10, 'end': 15, 'content': 'The door opens.'}
    assert action_target(None, [], None) == 'Silence'
    assert action_target(event, [], None) == 'Response'
    assert action_target(event, ['A dog sleeps.'], False) == 'Response'
    assert action_target(event, ['The door opens.'], True) == 'Hold'
    with pytest.raises(ValueError, match='valid boolean'):
        action_target(event, ['The door opens.'], 'true')


def test_corrective_oracle_cannot_use_inaccessible_or_future_answers():
    answers = [dict(time=t, content=str(t)) for t in (1, 5, 30, 35, 40)]
    # At 40, the 32-frame window starts at 9: time 5 is the single aged-out memory.
    assert [a['time'] for a in visible_prior_answers(answers, 40, 9)] == [5, 30, 35]
    assert [a['time'] for a in visible_prior_answers(answers, 5, 0)] == [1]


def test_recorded_context_keeps_generated_history_and_exact_image_slots(monkeypatch):
    sentinel = object()
    fake_training = SimpleNamespace(decision_messages=lambda *args, **kwargs: sentinel)
    monkeypatch.setitem(sys.modules, 'train_jevany_streaming', fake_training)
    install_training_contexts()
    assert fake_training.decision_messages({}, []) is sentinel
    record = {'split': 'train', 'controller_messages': [
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '00:01 '}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '</Response> An incorrect answer.'}]},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '[Decision]'}]},
    ]}
    first, second = object(), object()
    messages = fake_training.decision_messages(record, [first, second])
    assert messages[0]['content'][0]['image'] is first
    assert messages[2]['content'][0]['image'] is second
    assert messages[1]['content'][0]['text'] == '</Response> An incorrect answer.'
    assert 'image' not in record['controller_messages'][0]['content'][0]
    with pytest.raises(ValueError, match='restricted'):
        fake_training.decision_messages(dict(record, split='development'), [first, second])
    with pytest.raises(ValueError, match='alignment'):
        fake_training.decision_messages(record, [first, second, object()])


def test_raw_policy_verification_rejects_score_adjusted_actions():
    prediction = {'jevany_actions': {'Silence': 1, 'Hold': 0, 'Response': 1},
                  'jevany_action_trace': [{'logits': [1., 1., 0.], 'action': 'Silence'},
                                         {'logits': [0., -1., .1], 'action': 'Response'}]}
    Campaign.verify_raw([prediction], 2)
    prediction['jevany_action_trace'][1]['action'] = 'Silence'
    with pytest.raises(ValueError, match='raw learned argmax'):
        Campaign.verify_raw([prediction], 2)


def test_frozen_artifacts_cannot_be_silently_replaced(tmp_path):
    target = tmp_path / 'plan.json'
    write(target, {'seed': 42}, frozen=True)
    write(target, {'seed': 42}, frozen=True)
    with pytest.raises(ValueError, match='frozen artifact differs'):
        write(target, {'seed': 43}, frozen=True)


def test_processor_metadata_freeze_survives_json_tuple_roundtrip(tmp_path):
    target = tmp_path / 'namespace.json'
    processor = {'image_mean': (.4, .5, .6), 'size': {'height': 384, 'width': 384}}
    write(target, processor, frozen=True)
    write(target, processor, frozen=True)
    write(target, dict(processor, image_mean=[.4, .5, .6]), frozen=True)
