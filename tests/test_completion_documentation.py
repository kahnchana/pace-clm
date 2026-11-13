import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from document_jevany_completed import completion_ready,full_results,pooled_domain,REQUIRED_CHECKPOINT


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_report_cannot_finalize_before_step32_and_every_evaluation(tmp_path):
    run=tmp_path
    put(run/'workflow-status.json',{'state':'training'});put(run/'latest.json',{'step':30})
    assert not completion_ready(run)[0]
    put(run/'step-000032/commit.json',{'step':32,'files':{k:'hash' for k in REQUIRED_CHECKPOINT}})
    put(run/'evaluation/full-step32/result.json',{'cohorts':{'web_all':{'episodes':500},'ego_all':{'episodes':326}}})
    put(run/'latest.json',{'step':32})
    for i in range(32):put(run/f'batch-{i:06d}/report.json',{})
    put(run/'workflow-status.json',{'state':'complete'})
    put(run/'queue/full-step32.json',{})
    assert completion_ready(run)[0]
    put(run/'queue/another-full.json',{})
    assert not completion_ready(run)[0]
    put(run/'evaluation/another-full/result.json',{})
    assert completion_ready(run)[0]
    put(run/'step-000032/commit.json',{'step':32,'files':{}})
    assert not completion_ready(run)[0]


def test_previous_run_reference_never_replaces_current_step16(tmp_path):
    cohorts={'web_all':{'pauc_f':55},'ego_all':{'pauc_f':42}}
    put(tmp_path/'evaluation/full-step16/result.json',{'job':{'kind':'full','step':16},'cohorts':cohorts})
    put(tmp_path/'evaluation/reference-mixed-v1-step16/result.json',{'job':{'kind':'reference','step':16},
        'cohorts':{'web_all':{'pauc_f':99},'ego_all':{'pauc_f':99}}})
    assert full_results(tmp_path)[16]['result']['cohorts']==cohorts


def test_combined_score_pools_counts_instead_of_averaging_pauc_f():
    cohorts={'web_train':{'useful':1,'fires':1,'gold_events':1,'temporal_recall':100,'pauc_f':100},
        'web_heldout':{'useful':0,'fires':9,'gold_events':9,'temporal_recall':0,'pauc_f':0}}
    assert pooled_domain(cohorts,'web')==10
