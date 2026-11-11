import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pace_clm.rl import categorical_loss, observation
from pace_clm.rl_episode import draw_schedule, replay_partition, ordered_replay
from prepare_jevany_episode_rl import episode


def test_annotation_tail_is_not_negative_and_censored_event_rejects_whole_episode():
    meta={'duration':90.,'num_frames':2700}
    text=[[i,i+1,'opened' if i in (5,6) else ''] for i in range(20)]
    spec=episode('notify',text,meta,'/train/video.mp4')
    assert spec['rollout_stop']==20
    assert spec['gold_points'][0]['time_seconds']==6
    assert spec['initial_memory'] is None and not spec['prefix_turns']
    text[-1][-1]='late event'
    with pytest.raises(ValueError,match='incomplete_event_window'):episode('notify',text,meta,'/train/video.mp4')
    text[2][0]=1.5
    with pytest.raises(ValueError,match='annotation_gap'):episode('notify',text,meta,'/train/video.mp4')


def test_real_event_free_episode_is_retained():
    spec=episode('notify',[[0,10,'']],{'duration':25,'num_frames':250},'/train/q.mp4')
    assert spec['rollout_frames']==10 and spec['gold_points']==[]


def test_schedule_uniform_draws_complete_groups_and_deterministic_replay_credit():
    episodes=[{'rollout_frames':n} for n in (6,31,67,168,179)]
    batches=draw_schedule(episodes)
    assert batches==draw_schedule(episodes)
    assert batches[-1]['exposure_after']<=131072
    exposure=0;seen=[];rows=list(range(4032))
    for batch in batches:
        assert len(batch['episode_indices'])>=2
        assert batch['decisions']==sum(4*episodes[i]['rollout_frames'] for i in batch['episode_indices'])
        exposure+=batch['decisions']
        assert batch['anchor_start']+batch['anchor_count']==exposure*126//4096
        partitions=[replay_partition(rows,batch,r) for r in range(7)]
        selected=[v for part in partitions for v in part]
        assert sorted(selected)==list(range(batch['anchor_start'],batch['anchor_start']+batch['anchor_count']))
        seen.extend(batch['episode_indices'])
    for start in range(0,len(seen)-len(episodes)+1,len(episodes)):
        assert sorted(seen[start:start+len(episodes)])==list(range(len(episodes)))


def test_unequal_trajectory_lengths_and_rank_work_preserve_global_gradient():
    torch.manual_seed(1)
    lengths=[6,31,67,12,8,41,179,13]
    x=torch.randn(sum(lengths),5);actions=torch.arange(len(x))%3
    advantages=torch.repeat_interleave(torch.tensor([1.,-1.,.5,-.5,2.,-2.,.2,-.2]),torch.tensor(lengths))
    weight=torch.randn(5,3,requires_grad=True)
    ref=torch.log_softmax(torch.randn(len(x),3),-1)
    expected=torch.autograd.grad(categorical_loss(x@weight,ref,actions,advantages,total_trajectories=8)[0],weight)[0]
    ranks=[[] for _ in range(7)];start=0
    for i,length in enumerate(lengths):ranks[i%7].extend(range(start,start+length));start+=length
    gradients=[]
    for ids in ranks:
        w=weight.detach().clone().requires_grad_(True)
        loss=categorical_loss(x[ids]@w,ref[ids],actions[ids],advantages[ids],total_trajectories=8)[0]
        gradients.append(torch.autograd.grad(loss*7,w)[0])
    torch.testing.assert_close(sum(gradients)/7,expected)


def test_chunk_boundaries_preserve_causal_history_and_aged_response_memory():
    spec={'span_id':'episode','video_id':'v','video_path':'v.mp4','question':'notify',
          'decode_start':0,'initial_memory':None}
    def run(chunks):
        history=[];rows=[]
        for start,stop in chunks:
            for frame in range(start,stop):
                row=observation(spec,history,frame);rows.append(row)
                assert all(t['frame_index']<=frame for t in row['turns'])
                history.append({'frame_index':frame,'emit':'</Response> opened' if frame==2 else '</Silence>'})
            # Simulate a chunk scheduler serializing CPU behavioral state.
            history=copy.deepcopy(history)
        return rows
    full=run([(0,80)])
    assert full==run([(0,17),(17,32),(32,33),(33,80)])
    assert full[34]['memory']['frame_index']==2
    assert len(full[-1]['turns'])==32


def test_replay_preserves_collected_peers_and_row_positions():
    rows=[{'row':{'collection_batch':batch,'collection_position':position,'collection_size':size}}
          for batch,size in ((0,6),(1,5),(2,1)) for position in range(size)]
    assert ordered_replay(rows[::-1])==rows
    with pytest.raises(ValueError,match='incomplete or duplicate'):ordered_replay(rows[1:])
    with pytest.raises(ValueError,match='incomplete or duplicate'):ordered_replay(rows+[rows[0]])
