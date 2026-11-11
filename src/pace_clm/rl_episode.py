"""Frozen episode draws and decision-based supervised replay credits."""
import random
from itertools import groupby

ANCHOR_COEFFICIENT = 0.0035140820988695176


def draw_schedule(episodes, *, budget=131072, batch_target=4096, seed=42):
    """Uniform shuffled draws, no length filtering or truncated final episodes.

    Stop before the first group that would exceed the budget. A final singleton
    group joins its preceding batch so every one of seven ranks has work.
    """
    if not episodes or budget<1 or batch_target<1:
        raise ValueError('invalid episode schedule')
    rng=random.Random(seed)
    draws=[];total=0;exhausted=False
    while not exhausted:
        order=list(range(len(episodes)));rng.shuffle(order)
        for index in order:
            decisions=4*episodes[index]['rollout_frames']
            if decisions<=0:raise ValueError('empty episode')
            if total+decisions>budget:
                exhausted=True;break
            draws.append(index);total+=decisions
    batches=[];pending=[];count=0
    for index in draws:
        pending.append(index);count+=4*episodes[index]['rollout_frames']
        if count>=batch_target and len(pending)>=2:
            batches.append(pending);pending=[];count=0
    if pending:
        if len(pending)<2 and batches:batches[-1].extend(pending)
        else:batches.append(pending)
    if not batches or any(len(batch)<2 for batch in batches):
        raise ValueError('budget cannot supply all seven learners')
    result=[];exposure=groups=0;credits=0
    for batch in batches:
        decisions=sum(4*episodes[i]['rollout_frames'] for i in batch)
        next_exposure=exposure+decisions
        next_credits=next_exposure*126//4096
        result.append({'episode_indices':batch,'decisions':decisions,
                       'exposure_before':exposure,'exposure_after':next_exposure,
                       'groups_before':groups,'groups_after':groups+len(batch),
                       'anchor_start':credits,'anchor_count':next_credits-credits})
        exposure=next_exposure;groups+=len(batch);credits=next_credits
    return result


def replay_partition(rows, batch, rank, world=7):
    start,count=batch['anchor_start'],batch['anchor_count']
    selected=rows[start:start+count]
    if len(selected)!=count or not 0<=rank<world:
        raise ValueError('invalid replay credit or rank')
    return selected[rank::world]


def ordered_replay(steps):
    """Replay complete collected batches in their original row positions."""
    result=sorted(steps,key=lambda s:(s['row']['collection_batch'],s['row']['collection_position']))
    for _,group in groupby(result,key=lambda s:s['row']['collection_batch']):
        group=list(group)
        sizes={s['row']['collection_size'] for s in group}
        if sizes!={len(group)} or [s['row']['collection_position'] for s in group]!=list(range(len(group))):
            raise ValueError('incomplete or duplicate collected replay batch')
    return result
