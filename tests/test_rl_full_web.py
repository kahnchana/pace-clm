import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_jevany_e2_full_web import seed_predictions, validate_judgments, sha, rows, write, write_rows


def fixture(tmp_path):
    source, output, checkpoint = tmp_path/'dev', tmp_path/'full', tmp_path/'checkpoint'
    source.mkdir()
    (checkpoint/'controller_adapter').mkdir(parents=True)
    (checkpoint/'controller_adapter/adapter_model.safetensors').write_bytes(b'adapter')
    (checkpoint/'pointer_head.pt').write_bytes(b'head')
    write(source/'manifest.json', {'response_margin': 0, 'history': 'self-generated',
          'answer_model': 'promoted', 'max_new_tokens': 96, 'rows': 2, 'question_ids': ['b','d'],
          'controller_adapter_sha256': sha(checkpoint/'controller_adapter/adapter_model.safetensors'),
          'pointer_head_sha256': sha(checkpoint/'pointer_head.pt')})
    write_rows(source/'predictions.jsonl', [{'question_id': q, 'model_response_list': []} for q in ('b','d')])
    return source, output, checkpoint


def test_reuse_repartitions_whole_videos_and_preserves_resumed_predictions(tmp_path):
    source, output, checkpoint = fixture(tmp_path)
    info = seed_predictions(source, output, list('abcde'), checkpoint, world=2)
    assert info['reused_complete_video_predictions'] == 2
    assert rows(output/'predictions.rank00.jsonl') == []
    assert [p['question_id'] for p in rows(output/'predictions.rank01.jsonl')] == ['b','d']
    write_rows(output/'predictions.rank00.jsonl', [{'question_id': 'a', 'model_response_list': [{'time': 7, 'content': 'actual saved answer'}]}])
    seed_predictions(source, output, list('abcde'), checkpoint, world=2)
    assert rows(output/'predictions.rank00.jsonl')[0]['model_response_list'][0]['content'] == 'actual saved answer'
    assert json.loads((output/'manifest.json').read_text())['question_ids'] == list('abcde')


def test_reuse_rejects_wrong_checkpoint_and_wrong_rank(tmp_path):
    source, output, checkpoint = fixture(tmp_path)
    seed_predictions(source, output, list('abcde'), checkpoint, world=2)
    write_rows(output/'predictions.rank00.jsonl', [{'question_id': 'b'}])
    with pytest.raises(ValueError, match='wrong-rank'):
        seed_predictions(source, output, list('abcde'), checkpoint, world=2)
    (checkpoint/'pointer_head.pt').write_bytes(b'different model')
    with pytest.raises(ValueError, match='different checkpoint'):
        seed_predictions(source, output, list('abcde'), checkpoint, world=2)


def test_judge_completeness_and_validity_are_required():
    requests = [{'custom_id': 'a'}, {'custom_id': 'b'}]
    one = lambda q, score: {'custom_id': q, 'response': {'body': {'choices': [{'message': {'content': score}}]}}}
    validate_judgments(requests, [one('a','3'), one('b','1')])
    for responses in ([one('a','3')], [one('a','3'),one('a','3')], [one('a','3'),one('b','bad')]):
        with pytest.raises(ValueError):
            validate_judgments(requests, responses)
