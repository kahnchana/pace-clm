import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from pace_clm.rl_anchor import anchor_loss, anchor_partition, calibrate_anchor

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_uneven_global_ce_plus_rl_matches_one_batch_gradient():
    torch.manual_seed(42)
    x=torch.randn(126,5);labels=torch.arange(126)%3
    weight=torch.randn(5,3,requires_grad=True)
    rl_weight=torch.randn_like(weight)
    expected=torch.autograd.grad(F.cross_entropy(x@weight,labels)+.03*(weight*rl_weight).sum(),weight)[0]
    parts=[]
    for rank in range(7):
        local=weight.detach().clone().requires_grad_(True)
        loss=anchor_loss(x[rank::7]@local,labels[rank::7],126)*7+.03*(local*rl_weight).sum()
        parts.append(torch.autograd.grad(loss,local)[0])
    torch.testing.assert_close(sum(parts)/7,expected)


def test_fixed_gradient_calibration_and_combination():
    rl=torch.tensor([3.,4.]);ce=torch.tensor([-4.,3.])
    coefficient=calibrate_anchor(float(rl.norm()),float(ce.norm()))
    assert coefficient==pytest.approx(.15)
    assert (coefficient*ce).norm()/rl.norm()==pytest.approx(.15)
    combined=rl.clone().add_(ce,alpha=coefficient)
    torch.testing.assert_close(combined,rl+coefficient*ce)
    # The saved coefficient is reused even as later gradient norms change.
    assert coefficient!=calibrate_anchor(10.,5.)


@pytest.mark.parametrize('rl,ce',[(0.,1.),(1.,0.),(float('nan'),1.),(1.,float('inf'))])
def test_invalid_calibration_fails(rl,ce):
    with pytest.raises(ValueError):calibrate_anchor(rl,ce)


def test_replay_partitions_are_disjoint_and_resume_uses_absolute_round():
    rows=list(range(126*32))
    groups=[anchor_partition(rows,8,r,7) for r in range(7)]
    assert sorted(x for group in groups for x in group)==list(range(126*8,126*9))
    assert all(len(group)==18 for group in groups)
    with pytest.raises(ValueError):anchor_partition(rows,32,0,7)


def test_selection_deterministic_and_excludes_non_training_videos(tmp_path):
    from prepare_jevany_rl_anchor import select
    rows=[{'video_path':f'/train/{v}.mp4','split':'train','label':label,'id':i}
          for v in ('allowed','denied','outsider') for label in ('Silence','Hold','Response') for i in range(5)]
    source=tmp_path/'train.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    a,counts=select(source,{'/train/allowed.mp4','/train/denied.mp4'},{'denied'},3)
    b,_=select(source,{'/train/allowed.mp4','/train/denied.mp4'},{'denied'},3)
    assert a==b
    assert counts=={'Silence':5,'Hold':5,'Response':5}
    assert all(r['video_path']=='/train/allowed.mp4' for group in a.values() for _,r in group)


def test_coverage_audit_uses_fire_timestamps_and_distinct_official_assignment(tmp_path):
    from audit_jevany_rl_coverage import audit
    batch=tmp_path/'batch-000000';(batch/'reward').mkdir(parents=True)
    traces=[];scores=[]
    for i in range(4):
        traces.append({'trajectory_id':i,'spec':{'span_id':'group','gold_points':[{},{}]},
                       'reward':.5,'reward_metrics':{'num_value_adding_fires':1},
                       'steps':[{'row':{'frame_index':101},'action':2,'old_logp':[-2.,-3.,-.2]}]})
        scores.append({'question_id':f'group:{i}','answer':[
            {'reply_timespan':[100,104],'judge_scores':{'101.0':2}},
            {'reply_timespan':[101,105],'judge_scores':{'101.0':2}}]})
    (batch/'scored.json').write_text(json.dumps(traces));(batch/'reward/scores.json').write_text(json.dumps(scores))
    s=audit(tmp_path)['strata']['event']
    assert s['useful_responses']==4
    assert s['gold_events_missed_by_all_four']==1
    assert s['useful_response_zero_advantage']==4


def test_workflow_fresh_sft_fixed_anchor_and_eight_round_evaluations(tmp_path):
    import run_jevany_e3_anchor as workflow
    args=list(map(str,workflow.training_args(tmp_path,32)))
    assert args[args.index('--checkpoint')+1]==str(workflow.SFT)
    assert args[args.index('--anchor-target-fraction')+1]=='.15'
    assert args[args.index('--anchor-per-round')+1]=='126'
    assert '--resume' not in args
    assert workflow.EVALUATION_ROUNDS==(8,16,24,32)
    (tmp_path/'freeze.json').write_text('{}')
    assert '--resume' in workflow.training_args(tmp_path,32)


def test_checkpoint_commit_rejects_modified_optimizer_or_wrong_step(tmp_path):
    import hashlib
    from pace_clm.rl_anchor import verify_committed_checkpoint
    names=['pointer_head.pt','training_state.pt','rl_state.json','controller_adapter/adapter_model.safetensors']
    for name in names:
        p=tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
    files={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names}
    (tmp_path/'commit.json').write_text(json.dumps({'step':8,'files':files}))
    assert verify_committed_checkpoint(tmp_path,8)['step']==8
    with pytest.raises(ValueError):verify_committed_checkpoint(tmp_path,9)
    (tmp_path/'training_state.pt').write_bytes(b'changed optimizer')
    with pytest.raises(ValueError,match='changed'):verify_committed_checkpoint(tmp_path,8)


@pytest.mark.parametrize('useful,accepted',[(630,False),(631,True)])
def test_report_requires_full_web_coverage_and_separates_point_gain_from_ci(tmp_path,monkeypatch,useful,accepted):
    import report_jevany_rl_anchor as report
    full=tmp_path/'evaluation/web-step24';full.mkdir(parents=True)
    (full/'pauc_f_summary-gpt-4.1-v1.json').write_text('{}')
    (tmp_path/'workflow-status.json').write_text('{"state":"full_web_evaluation"}')
    (tmp_path/'anchor-calibration.json').write_text('{"coefficient":0.01}')
    baseline={'pauc_f':54.665,'num_value_adding_fires':701,'num_wasted_fires':290}
    candidate={'pauc_f':55.,'num_value_adding_fires':useful,'num_wasted_fires':200}
    monkeypatch.setattr(report,'verify_web',lambda path:(candidate if path==full else baseline,None,None))
    monkeypatch.setattr(report,'paired_bootstrap',lambda *args:{'difference_points':.335,'percentile_95_ci':[-.1,.8]})
    monkeypatch.setattr(sys,'argv',['report','--run',str(tmp_path)])
    report.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['full_web'][0]['round']==24
    assert result['full_web'][0]['point_improvement_with_coverage']==accepted
    assert not result['full_web'][0]['positive_paired_95_ci_with_coverage']
