import copy
from collections import Counter

import pytest

from pace_clm.rl_balanced160 import SOURCES, attach_source_loss_scales, balanced_schedule, source
from pace_clm.rl_estp import continuation_schedule
from pace_clm.rl_mixed import episode_balanced_schedule


def fixture_schedule():
    episodes=[dict(domain=d,span_id=f'{d}:{i}',video_id=f'{d}:{i}',rollout_frames=n)
              for d,count,n in [('ours',700,79),('web',400,18),('ego',145,361)]
              for i in range(count)]
    prefix=episode_balanced_schedule(episodes,budget=10**6)[:32]
    episodes += [dict(domain='ego',data_source='estp_it',span_id=f'it:{i}',video_id=f'it:{i}',rollout_frames=300)
                 for i in range(111)]
    prefix=continuation_schedule(episodes,prefix)
    episodes += [dict(domain='ego',data_source='estp_it',span_id=f'it:{i}',video_id=f'it:{i}',rollout_frames=300)
                 for i in range(111,175)]
    return episodes,prefix


def test_continuation_preserves_prefix_and_sampler_replay_cursors():
    episodes,prefix=fixture_schedule();original=copy.deepcopy(prefix)
    schedule=balanced_schedule(episodes,prefix)
    assert schedule[:128]==prefix==original
    assert len(schedule)==160
    assert schedule==balanced_schedule(episodes,prefix)
    previous={i for b in prefix for i in b['episode_indices']}
    selected=[i for b in schedule[128:] for i in b['episode_indices']]
    assert len(selected)==len(set(selected))==256
    assert Counter(source(episodes[i]) for i in selected if i in previous)=={'ego':64}
    for j,b in enumerate(schedule[128:],128):
        assert Counter(source(episodes[i]) for i in b['episode_indices'])==dict.fromkeys(SOURCES,2)
        assert b['groups_before']==schedule[j-1]['groups_after']
        assert b['anchor_start']==schedule[j-1]['anchor_start']+schedule[j-1]['anchor_count']
        assert b['decisions']==sum(4*episodes[i]['rollout_frames'] for i in b['episode_indices'])
        assert b['exposure_after']==b['exposure_before']+b['decisions']
    assert sum(b['anchor_count'] for b in schedule)==schedule[-1]['exposure_after']*126//4096


def test_equal_source_objective_weights_even_with_unequal_trajectory_counts():
    traces=[dict(spec=dict(domain='ego' if s=='estp_it' else s,data_source=s),steps=[{}])
            for s,n in zip(SOURCES,(4,8,12,16)) for _ in range(n)]
    attach_source_loss_scales(traces)
    for s in SOURCES:
        assert sum(t['steps'][0]['loss_scale'] for t in traces if source(t['spec'])==s)/len(traces)==pytest.approx(.25)
    with pytest.raises(ValueError,match='four sources'):
        attach_source_loss_scales([t for t in traces if source(t['spec'])!='estp_it'])


def test_reject_insufficient_source_instead_of_silent_within_phase_repetition():
    episodes,prefix=fixture_schedule()
    for s in episodes:
        if source(s)=='ego':s['data_source']='ours'
    with pytest.raises(ValueError,match='insufficient source episodes: ego'):
        balanced_schedule(episodes,prefix)
