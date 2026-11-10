import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_qwen35_control import aliases, identities
from finalize_qwen35_control_preflight import estimate_steps
from run_qwen35_control import verify_web


def test_video_aliases_cover_crops_and_source_prefixes():
    assert aliases('ActivityNet/train/v_OSfMU69X3C4.mp4') & aliases('OSfMU69X3C4.7.mp4')
    assert aliases('c2d06df7-5d3a-4116-9edb-f1c81a4f669b-0-360.mp4') & aliases('c2d06df7-5d3a-4116-9edb-f1c81a4f669b.mp4')
    assert not aliases('6183717048.mp4') & aliases('6183717049.mp4')
    assert set(identities({'data': [{'video_path':'x.mp4','question':'ignore me'}]})) == {'x.mp4'}


def test_step_estimate_has_four_equal_segments_and_no_pool_reuse():
    total, milestones = estimate_steps(16, 8343088)
    assert total == 2700
    assert milestones == [675,1350,2025,2700]
    with pytest.raises(ValueError):
        estimate_steps(0, 8343088)
    with pytest.raises(ValueError):
        estimate_steps(0.01, 8343088)


def make_web(path):
    ids = [str(i) for i in range(500)]
    (path/'manifest.json').write_text(json.dumps({'question_ids':ids,'controller_adapter_sha256':'a','pointer_head_sha256':'b'}))
    (path/'predictions.jsonl').write_text(''.join(json.dumps({'question_id':i})+'\n' for i in ids))
    (path/'pauc_llm_scores-gpt-4.1.json').write_text(json.dumps([{'question_id':i} for i in ids]))
    (path/'pauc_batch_input-gpt-4.1.jsonl').write_text('{"custom_id":"request1"}\n')
    response = {'custom_id':'request1','response':{'body':{'choices':[{'message':{'content':'3'}}]}}}
    (path/'pauc_batch_output-gpt-4.1.jsonl').write_text(json.dumps(response)+'\n')
    (path/'pauc_f_summary-gpt-4.1-v1.json').write_text(json.dumps({'num_videos':500,'judge_model':'gpt-4.1','scorer_version':'pauc-f-v1','omega':0.5}))


def test_web_gate_rejects_missing_judgments(tmp_path):
    make_web(tmp_path)
    verify_web(tmp_path)
    (tmp_path/'pauc_batch_output-gpt-4.1.jsonl').write_text('')
    with pytest.raises(ValueError, match='judge request coverage'):
        verify_web(tmp_path)


def test_web_gate_rejects_duplicate_prediction_ids(tmp_path):
    make_web(tmp_path)
    with (tmp_path/'predictions.jsonl').open('a') as handle:
        handle.write('{"question_id":"0"}\n')
    with pytest.raises(ValueError, match='prediction or score IDs'):
        verify_web(tmp_path)


def test_web_gate_accepts_complete_silent_rollout_without_judge_requests(tmp_path):
    make_web(tmp_path)
    (tmp_path/'pauc_batch_input-gpt-4.1.jsonl').write_text('')
    (tmp_path/'pauc_batch_output-gpt-4.1.jsonl').write_text('')
    verify_web(tmp_path)
