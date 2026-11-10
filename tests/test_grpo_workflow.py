import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_jevany_e2_grpo as workflow


def test_training_waits_for_both_successful_complete_web_evaluations(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr(workflow,'verify_web',lambda path:seen.append(path.name))
    assert not workflow.prerequisites_complete(tmp_path)
    (tmp_path/'status.json').write_text(json.dumps({'state':'rollout'}))
    assert not workflow.prerequisites_complete(tmp_path)
    assert seen==[]
    (tmp_path/'status.json').write_text(json.dumps({'state':'complete'}))
    (tmp_path/'comparison.json').write_text(json.dumps({'complete':True,'expected_rounds':[16,8]}))
    assert workflow.prerequisites_complete(tmp_path)
    assert seen==['web-step8','web-step16']


def test_failed_or_partial_prerequisite_never_starts_training(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'state':'failed'}))
    with pytest.raises(RuntimeError,match='do not start'):
        workflow.prerequisites_complete(tmp_path)
    (tmp_path/'status.json').write_text(json.dumps({'state':'complete'}))
    (tmp_path/'comparison.json').write_text(json.dumps({'complete':False,'expected_rounds':[16,8]}))
    with pytest.raises(ValueError,match='incomplete'):
        workflow.prerequisites_complete(tmp_path)


def test_fresh_sft_initialization_and_absolute_midbatch_gate_target(tmp_path):
    argv=workflow.training_args(tmp_path,1,stop_after=1)
    assert argv[argv.index('--checkpoint')+1]==str(workflow.SFT)
    assert argv[argv.index('--stop-after-optimizer-steps')+1]=='1'
    assert '--resume' not in argv
    (tmp_path/'freeze.json').write_text('{}')
    argv=workflow.training_args(tmp_path,32)
    assert '--resume' in argv
    assert argv[argv.index('--updates')+1]=='32'
    assert argv[argv.index('--checkpoint')+1]==str(workflow.SFT)


def test_evaluates_every_eight_collection_rounds_through_32():
    assert workflow.EVALUATION_ROUNDS==(8,16,24,32)


def test_report_includes_round24_full_web_and_development(tmp_path,monkeypatch):
    import report_jevany_grpo as report
    def score(videos,value,useful,wasted):
        return {'num_videos':videos,'pauc_f':value,'num_value_adding_fires':useful,
                'num_wasted_fires':wasted,'fire_precision':70.,'pauc_recall_component':44.}
    (tmp_path/'development').mkdir()
    (tmp_path/'development/sft-baseline.json').write_text(json.dumps(score(128,54.1,184,105)))
    (tmp_path/'workflow-status.json').write_text(json.dumps({'state':'full_web_evaluation'}))
    dev=tmp_path/'evaluation/dev-step24';dev.mkdir(parents=True)
    (dev/'pauc_f_summary-gpt-4.1-v1.json').write_text(json.dumps(score(128,55.,181,70)))
    full=tmp_path/'evaluation/web-step24';full.mkdir()
    (full/'pauc_f_summary-gpt-4.1-v1.json').write_text(json.dumps(score(500,55.5,690,220)))
    monkeypatch.setattr(report,'verify_web',lambda path:(score(500,55.5,690,220) if path==full else score(500,54.67,701,290),None,None))
    monkeypatch.setattr(report,'paired_bootstrap',lambda *args:{'difference_points':.83,'percentile_95_ci':[-.2,1.8]})
    monkeypatch.setattr(sys,'argv',['report','--run',str(tmp_path)])
    report.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['development'][0]['round']==24
    assert result['full_web'][0]['round']==24
