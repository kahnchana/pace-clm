import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pace_clm.rl_mixed import (WEIGHTS,attach_loss_scales,categorical_loss,mixture_schedule,
    episode_balanced_schedule,training_schedule,observation,physical_row)
from jevany_mixed_reward import score_predictions
from zs_infer.pauc_f import score_pauc_f


def test_mixture_complete_episodes_budget_and_all_sources_per_update():
    episodes=[{'domain':domain,'rollout_frames':length} for domain in WEIGHTS for length in (7,39,70,179)]
    schedule=mixture_schedule(episodes)
    assert schedule==mixture_schedule(episodes)
    assert 97000<schedule[-1]['exposure_after']<=100000
    for batch in schedule:
        assert {episodes[i]['domain'] for i in batch['episode_indices']}==set(WEIGHTS)
        assert batch['decisions']==sum(4*episodes[i]['rollout_frames'] for i in batch['episode_indices'])
    assert sum(s['anchor_count'] for s in schedule)==schedule[-1]['exposure_after']*126//4096
    for domain,weight in WEIGHTS.items():
        assert abs(schedule[-1]['cumulative_domain_decisions'][domain]/schedule[-1]['exposure_after']-weight)<.03


def test_episode_balance_preserves_diversity_budget_and_replay_cursor():
    episodes=[{'domain':d,'span_id':f'{d}:{i}','rollout_frames':n}
        for d,n in [('ours',79),('web',18),('ego',361)] for i in range(100)]
    schedule=episode_balanced_schedule(episodes)
    assert schedule==training_schedule(episodes,sampling_unit='episode')
    assert schedule!=episode_balanced_schedule(episodes,seed=43)
    selected=[]
    for i,b in enumerate(schedule):
        batch=[episodes[j] for j in b['episode_indices']]
        assert [sum(s['domain']==d for s in batch) for d in WEIGHTS]==[4,2,2]
        assert b['decisions']==4*sum(s['rollout_frames'] for s in batch)
        assert b['exposure_before']==(schedule[i-1]['exposure_after'] if i else 0)
        assert b['groups_after']==8*(i+1)
        assert b['anchor_start']==(schedule[i-1]['anchor_start']+schedule[i-1]['anchor_count'] if i else 0)
        selected+=b['episode_indices']
    assert len(selected)==len(set(selected))
    assert schedule[-1]['exposure_after']<=100000<schedule[-1]['exposure_after']+schedule[-1]['decisions']
    assert sum(b['anchor_count'] for b in schedule)==schedule[-1]['exposure_after']*126//4096
    # Resume is the untouched suffix of the frozen schedule, never a redraw.
    assert selected[8:]==[j for b in schedule[1:] for j in b['episode_indices']]
    with pytest.raises(ValueError,match='complete'):
        episode_balanced_schedule(episodes,budget=1)
    with pytest.raises(ValueError,match='duplicate'):
        episode_balanced_schedule(episodes+[episodes[0]])


def test_episode_scheduler_never_recycles_a_small_source():
    episodes=[{'domain':d,'span_id':f'{d}:{i}','rollout_frames':n}
        for d,n,k in [('ours',79,20),('web',18,20),('ego',361,3)] for i in range(k)]
    assert len(episode_balanced_schedule(episodes))==1


def test_temporal_diagnostics_uses_official_useful_assignment():
    from jevany_mixed_diagnostics import temporal_diagnostics
    spec={'domain':'ego','timestamps':[0,120,240],
        'gold_answers':[{'content':'Check soup.','reply_timespan':[119,121]},
            {'content':'Check soup.','reply_timespan':[239,241]}]}
    predictions=[{'question_id':'q','model_response_list':[{'time':120},{'time':240}],
        'jevany_action_trace':[{'frame_index':i,'logits':[0,0,0],'input_tokens':6000,'input_frames':32,
            'memory_question':'Q\nPrior response memory:\n- At 00:30: prior'} for i in range(3)]}]
    scored=[{'question_id':'q','answer':[{'reply_timespan':[119,121],'judge_scores':{'120.0':2}},
        {'reply_timespan':[239,241],'judge_scores':{}}]}]
    result=temporal_diagnostics(predictions,{'q':spec},scored)['buckets']
    assert [b['fires'] for b in result]==[0,1,1]
    assert [b['useful'] for b in result]==[0,1,0]
    assert [b['wasted'] for b in result]==[0,0,1]
    assert sum(b['repeated_event_descriptions'] for b in result)==1
    assert result[-1]['mean_response_probability']==pytest.approx(1/3)
    assert result[-1]['input_tokens_max']==6000


def test_uneven_rank_gradients_equal_weighted_source_episode_means():
    torch.manual_seed(2)
    domains=['ours','ours','web','web','web','ego'];lengths=[3,5,2,4,6,7]
    traces=[{'spec':{'domain':d},'steps':[{} for _ in range(n)]} for d,n in zip(domains,lengths)]
    attach_loss_scales(traces)
    logits=torch.randn(sum(lengths),3,requires_grad=True)
    reference=torch.log_softmax(torch.randn_like(logits),-1)
    actions=torch.arange(len(logits))%3;advantages=torch.randn(len(logits))
    logp=logits.log_softmax(-1);prob=logp.exp()
    per=-advantages*logp[torch.arange(len(logits)),actions]+.02*(prob*(logp-reference)).sum(-1)+.001*(prob*logp).sum(-1)
    expected=0.;offset=0;loss=0.
    for domain,length,trace in zip(domains,lengths,traces):
        sl=slice(offset,offset+length)
        expected=expected+WEIGHTS[domain]/domains.count(domain)*per[sl].sum()/32
        # Each rank can hold unequal numbers and lengths of trajectories.
        value,_,_=categorical_loss(logits[sl],reference[sl],actions[sl],advantages[sl],
            total_trajectories=len(traces),scales=torch.tensor([s['loss_scale'] for s in trace['steps']]))
        loss=loss+value;offset+=length
    assert loss.item()==pytest.approx(expected.item(),abs=1e-7)
    a=torch.autograd.grad(loss,logits,retain_graph=True)[0];b=torch.autograd.grad(expected,logits)[0]
    torch.testing.assert_close(a,b)


def test_native_physical_indices_causal_memory_and_fractional_time():
    spec={'span_id':'ego:x','video_id':'x','video_path':'/x','question':'Q',
          'rollout_start':0,'rollout_stop':40,'decode_start':0,'prefix_turns':[],
          'initial_memory':None,'source_indices':list(range(5,245,6)),
          'timestamps':[i/6 for i in range(5,245,6)]}
    history=[{'frame_index':i,'emit':'</Response> opened' if i==0 else '</Silence>'} for i in range(39)]
    row=observation(spec,history,39)
    assert len(row['turns'])==32
    assert row['memory']['time_seconds']==pytest.approx(5/6)
    physical=physical_row(row)
    assert physical['turns'][-1]['frame_index']==239
    assert physical['turns'][0]['frame_index']==53
    assert physical['turns'][-1]['ts']=='00:39'
    assert row['turns'][-1]['frame_index']==39
    assert max(t['frame_index'] for t in physical['turns'])<=spec['source_indices'][39]


def test_generator_images_and_features_use_same_native_keys():
    from jevany_mixed_inputs import EpisodeFrames
    frames=object.__new__(EpisodeFrames)
    frames.frames={('/x',5):('native image','native features'),('/x',0):('wrong image','wrong features')}
    row={'video_path':'/x','turns':[{'frame_index':0}],'_source_indices':[5]}
    assert frames.images_for(row)==['native image']
    assert frames.features_for(row)==['native features']


def test_native_original_windows_and_official_aggregation(tmp_path):
    gold=[{'question_id':'native','conversation':[{'role':'user','content':'What happened?'}],
           'answer':[{'content':'It opened.','reply_timespan':[.5,12.5],'related_timespan':[0,13]}]}]
    original=copy.deepcopy(gold)
    predictions=[{'question_id':'native','model_response_list':[
        {'role':'assistant','time':t,'content':f'answer {t}'} for t in (.5,7.5,12.5,13.5)]}]
    responses=[{'custom_id':f'native=0={t}','response':{'body':{'choices':[{'message':{'content':'3'}}]}}}
               for t in (.5,7.5,12.5)]
    metrics,_=score_predictions(predictions,gold,tmp_path,tmp_path/'unused',responses)
    assert gold==original
    assert metrics['native']['num_fires']==4
    assert metrics['native']['num_value_adding_fires']==1
    direct=score_pauc_f(tmp_path/'predictions.jsonl',tmp_path/'scores.json')
    assert metrics['native']['pauc_f']==direct['pauc_f']
    req=[json.loads(s) for s in (tmp_path/'requests.jsonl').read_text().splitlines()]
    assert len(req)==3
    assert 'answer 0.5 answer 7.5' in req[1]['body']['messages'][1]['content']


def test_genuine_event_free_reward_metrics_are_finite(tmp_path):
    pred=[{'question_id':'empty','model_response_list':[{'role':'assistant','time':1.,'content':'x'}]}]
    gold=[{'question_id':'empty','conversation':[{'role':'user','content':'Q'}],'answer':[]}]
    metrics,_=score_predictions(pred,gold,tmp_path,tmp_path/'unused',[])
    assert metrics['empty']['num_fires']==1
    assert metrics['empty']['num_gold_turns']==0
    assert metrics['empty']['pauc_f']==0
