import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from jevany_rl_reward import OFFICIAL, artifacts, request_key, score_traces
from zs_infer.pauc_f import score_pauc_f


def trace(fires=(), gold=True):
    return {'trajectory_id': 0,
            'spec': {'span_id': 'test', 'question': 'What changed?',
                     'rollout_start': 10, 'rollout_stop': 18,
                     'gold_points': [{'time_seconds': 11, 'content': 'It opened.'}] if gold else []},
            'steps': [{'row': {'frame_index': time}, 'action': 2 if time in fires else 0,
                       'answer': f'answer {time}' if time in fires else ''}
                      for time in range(10, 18)]}


def responses(ids, score='3'):
    return [{'custom_id': qid, 'response': {'body': {'choices': [
        {'message': {'content': score}}]}}} for qid in ids]


def test_official_wrapper_parity_accumulation_duplicates_and_absolute_time(tmp_path):
    item = trace((10, 11, 13, 15, 17))
    # End timestamp 15 is inclusive in the official scorer, all five fires count.
    ids = [f'test:0=0={time}' for time in (11, 13, 15)]
    score_traces([item], tmp_path, tmp_path/'unused.sqlite', supplied_outputs=responses(ids))
    requests = [json.loads(line) for line in (tmp_path/'requests.jsonl').read_text().splitlines()]
    assert len(requests) == 3
    assert 'answer 11 answer 13' in requests[1]['body']['messages'][1]['content']
    assert 'answer 10' not in requests[1]['body']['messages'][1]['content']
    assert item['reward_metrics']['num_fires'] == 5
    assert item['reward_metrics']['num_value_adding_fires'] == 1
    direct = score_pauc_f(tmp_path/'predictions.jsonl', tmp_path/'scores.json')
    assert item['reward'] == direct['pauc_f']/100


def test_silence_initial_pauc_convention_and_no_event_penalty(tmp_path):
    silent = trace()
    score_traces([silent], tmp_path/'silent', tmp_path/'unused', supplied_outputs=[])
    assert silent['reward'] == 0
    assert silent['reward_metrics']['pauc_recall_component'] == 25
    assert silent['reward_metrics']['num_value_adding_fires'] == 0
    none = trace((10, 14), gold=False)
    score_traces([none], tmp_path/'none', tmp_path/'unused', supplied_outputs=[])
    assert none['reward'] == -.5
    assert none['reward_metrics']['num_fires'] == 2


def test_boundary_censoring_and_incomplete_judging_rejected(tmp_path):
    item = trace((11,))
    with pytest.raises(ValueError, match='incomplete or duplicate'):
        score_traces([item], tmp_path, tmp_path/'unused', supplied_outputs=[])
    item['spec']['gold_points'][0]['time_seconds'] = 15
    with pytest.raises(ValueError, match='boundary-censored'):
        artifacts([item])


def test_cache_identity_includes_complete_request_and_model():
    body = {'model': 'gpt-4.1', 'messages': [{'role': 'user', 'content': 'a'}]}
    assert request_key(body) != request_key({**body, 'model': 'other'})
    assert request_key(body) != request_key({**body, 'messages': [{'role': 'user', 'content': 'b'}]})


def test_duplicate_text_is_deduplicated_only_in_official_judge_prompt(tmp_path):
    item = trace((11, 12))
    item['steps'][2]['answer'] = item['steps'][1]['answer']
    score_traces([item], tmp_path, tmp_path/'unused',
                 supplied_outputs=responses(['test:0=0=11']))
    assert item['reward_metrics']['num_fires'] == 2
    assert item['reward_metrics']['num_value_adding_fires'] == 1
