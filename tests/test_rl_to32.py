import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_jevany_to32 import aggregate,borrow_evaluation,schedule_to32,expanded_replay_rows
from pace_clm.rl_mixed import episode_balanced_schedule


def test_extension_preserves_every_original_batch_and_optimizer_replay_credit():
    episodes=[{'domain':d,'span_id':f'{d}:{i}','rollout_frames':n}
        for d,n in [('ours',79),('web',18),('ego',361)] for i in range(200)]
    old=episode_balanced_schedule(episodes,budget=100000)
    original=copy.deepcopy(old);extended=schedule_to32(episodes,old)
    assert old==original
    assert len(extended)==32 and extended[:len(old)]==old
    selected=[i for batch in extended for i in batch['episode_indices']]
    assert len(selected)==256==len(set(selected))
    for batch in extended:
        assert [sum(episodes[i]['domain']==d for i in batch['episode_indices'])
            for d in ('ours','web','ego')]==[4,2,2]
    assert extended[15]['groups_before']==extended[14]['groups_after']==120
    assert extended[15]['exposure_before']==extended[14]['exposure_after']
    assert extended[15]['anchor_start']==extended[14]['anchor_start']+extended[14]['anchor_count']
    assert sum(b['anchor_count'] for b in extended)==extended[-1]['exposure_after']*126//4096
    broken=copy.deepcopy(old);broken[1]['episode_indices'].reverse()
    with pytest.raises(ValueError,match='prefix'):schedule_to32(episodes,broken)


def test_borrowed_results_reaggregate_new_sampled_membership_without_touching_parent(tmp_path):
    parent=tmp_path/'parent';run=tmp_path/'continuation';source=parent/'evaluation/baseline-sft/result.json'
    source.parent.mkdir(parents=True);(run/'data').mkdir(parents=True)
    metrics={'a':{'num_value_adding_fires':1,'num_fires':2,'num_gold_turns':1,
        'pauc_recall_component':50,'pauc_f':50},
        'b':{'num_value_adding_fires':2,'num_fires':2,'num_gold_turns':2,
        'pauc_recall_component':100,'pauc_f':100}}
    original={'job':{'kind':'baseline'},'per_episode':metrics,'cohorts':{},'paired_vs_sft':{'stale':True}}
    source.write_text(json.dumps(original));before=source.read_bytes()
    cohorts={'web_all':['a','b'],'web_sampled':['b'],'web_unsampled':['a']}
    (run/'data/cohorts.json').write_text(json.dumps(cohorts))
    assert borrow_evaluation(parent,run,'baseline-sft')
    copied=json.loads((run/'evaluation/baseline-sft/result.json').read_text())
    assert copied['cohorts']['web_sampled']['pauc_f']==100
    assert copied['cohorts']['web_all']['useful']==3
    assert copied['cohorts']['web_all']['gold_events']==3
    assert copied['cohorts']['web_all']['pauc_f']==pytest.approx(100*2*.75*(2.5/3)/(.75+2.5/3))
    assert 'paired_vs_sft' not in copied
    assert source.read_bytes()==before
    assert aggregate(metrics,['a'])['pauc_f']==50


def test_extended_replay_covers_bank_boundary_without_changing_prior_credits():
    from pace_clm.rl_episode import replay_partition
    original=list(range(4032))
    schedule=[{'anchor_start':3844,'anchor_count':135},
              {'anchor_start':3979,'anchor_count':145},
              {'anchor_start':4124,'anchor_count':135}]
    expanded=expanded_replay_rows(original,schedule)
    assert len(expanded)==4259
    assert expanded[:4032]==original
    assert expanded[4032:]==original[:227]
    assert original==list(range(4032))
    for b in schedule:
        parts=[replay_partition(expanded,b,rank) for rank in range(7)]
        recovered=[None]*b['anchor_count']
        for rank,part in enumerate(parts):recovered[rank::7]=part
        assert recovered==[i%4032 for i in range(b['anchor_start'],b['anchor_start']+b['anchor_count'])]
    assert expanded[3844:3979]==original[3844:3979]
    with pytest.raises(ValueError,match='empty'):expanded_replay_rows([],schedule)
