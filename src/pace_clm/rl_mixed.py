"""Deterministic source mixtures and native frame/annotation contracts."""
import copy
import math
import random

WEIGHTS={'ours':.5,'web':.25,'ego':.25}


def validate_spec(spec):
    n=spec['rollout_stop']
    if spec['rollout_start']!=0 or spec['decode_start']!=0 or spec['prefix_turns'] or spec['initial_memory'] or n<=0:
        raise ValueError('expected complete episode with empty initialization')
    if spec['rollout_frames']!=n or len(spec['source_indices'])!=n or len(spec['timestamps'])!=n:
        raise ValueError('frame map length mismatch')
    if any(b<=a for a,b in zip(spec['timestamps'],spec['timestamps'][1:])):
        raise ValueError('timestamps must increase strictly')
    for answer in spec['gold_answers']:
        left,right=answer['reply_timespan']
        if not all(math.isfinite(v) for v in (left,right)) or left<0 or right<=left:
            raise ValueError('invalid native reply window')


def mixture_schedule(episodes,budget=100000,batch_target=4096,seed=42):
    """Deficit scheduling by decisions; uniform shuffled episodes within source.

    Never skip a long selected episode to squeeze into the remaining budget.
    Every update includes all three sources. Partial final groups join the
    preceding batch; no individual episode is truncated.
    """
    pools={d:[i for i,s in enumerate(episodes) if s['domain']==d] for d in WEIGHTS}
    if any(not p for p in pools.values()):raise ValueError('all three training sources required')
    rng=random.Random(seed);queues={d:[] for d in pools}
    used={d:0 for d in pools};draws=[];total=0
    while True:
        domain=max(WEIGHTS,key=lambda d:WEIGHTS[d]*(total+batch_target)-used[d])
        if not queues[domain]:
            queues[domain]=pools[domain].copy();rng.shuffle(queues[domain])
        index=queues[domain].pop();cost=4*episodes[index]['rollout_frames']
        if cost<=0:raise ValueError('empty episode')
        if total+cost>budget:break
        draws.append(index);used[domain]+=cost;total+=cost
    batches=[];pending=[];count=0
    for index in draws:
        pending.append(index);count+=4*episodes[index]['rollout_frames']
        if count>=batch_target and {episodes[i]['domain'] for i in pending}==set(WEIGHTS):
            batches.append(pending);pending=[];count=0
    if pending:
        if {episodes[i]['domain'] for i in pending}==set(WEIGHTS):batches.append(pending)
        elif batches:batches[-1].extend(pending)
        else:raise ValueError('budget cannot supply three source groups')
    result=[];exposure=groups=credits=0;domain_exposure={d:0 for d in WEIGHTS}
    for batch in batches:
        delta={d:sum(4*episodes[i]['rollout_frames'] for i in batch if episodes[i]['domain']==d) for d in WEIGHTS}
        n=sum(delta.values());after=exposure+n;next_credit=after*126//4096
        for d in WEIGHTS:domain_exposure[d]+=delta[d]
        result.append({'episode_indices':batch,'decisions':n,'exposure_before':exposure,'exposure_after':after,
                       'groups_before':groups,'groups_after':groups+len(batch),
                       'anchor_start':credits,'anchor_count':next_credit-credits,
                       'domain_decisions':delta,'cumulative_domain_decisions':domain_exposure.copy()})
        exposure=after;groups+=len(batch);credits=next_credit
    if not result:raise ValueError('empty mixture schedule')
    return result


def episode_balanced_schedule(episodes,budget=100000,seed=42):
    """Without-replacement 4:2:2 episode batches, four full trajectories each."""
    composition={'ours':4,'web':2,'ego':2}
    rng=random.Random(seed)
    pools={d:[i for i,s in enumerate(episodes) if s['domain']==d] for d in composition}
    ids=[s['span_id'] for s in episodes]
    if len(set(ids))!=len(ids):raise ValueError('duplicate episode identities')
    for pool in pools.values():rng.shuffle(pool)
    cursor={d:0 for d in composition};used={d:0 for d in composition}
    result=[];exposure=groups=credits=0
    while all(cursor[d]+n<=len(pools[d]) for d,n in composition.items()):
        batch=[i for d,n in composition.items() for i in pools[d][cursor[d]:cursor[d]+n]]
        delta={d:sum(4*episodes[i]['rollout_frames'] for i in batch if episodes[i]['domain']==d) for d in composition}
        n=sum(delta.values())
        if n<=0 or any(episodes[i]['rollout_frames']<=0 for i in batch):raise ValueError('empty episode')
        if exposure+n>budget:break
        after=exposure+n;next_credit=after*126//4096
        for d,count in composition.items():cursor[d]+=count;used[d]+=delta[d]
        result.append({'episode_indices':batch,'decisions':n,'exposure_before':exposure,'exposure_after':after,
            'groups_before':groups,'groups_after':groups+len(batch),'anchor_start':credits,
            'anchor_count':next_credit-credits,'domain_decisions':delta,'cumulative_domain_decisions':used.copy()})
        exposure=after;groups+=len(batch);credits=next_credit
    if not result:raise ValueError('budget or source pools cannot supply one complete 4:2:2 batch')
    return result


def training_schedule(episodes,*,sampling_unit='decision',budget=100000,seed=42):
    if sampling_unit=='episode':return episode_balanced_schedule(episodes,budget=budget,seed=seed)
    if sampling_unit=='decision':return mixture_schedule(episodes,budget=budget,seed=seed)
    raise ValueError('unknown sampling unit')


def attach_loss_scales(traces):
    counts={d:sum(t['spec']['domain']==d for t in traces) for d in WEIGHTS}
    if any(n==0 for n in counts.values()):raise ValueError('missing source in global batch')
    for trace in traces:
        scale=WEIGHTS[trace['spec']['domain']]*len(traces)/counts[trace['spec']['domain']]
        for step in trace['steps']:step['loss_scale']=scale
    return counts


def physical_row(row):
    if '_source_indices' not in row:return row
    result=dict(row)
    result['turns']=[{**t,'frame_index':row['_source_indices'][t['frame_index']]} for t in row['turns']]
    return result


def observation(spec,history,frame):
    from pace_clm.rl import observation as base
    row=base(spec,history,frame)
    times=spec['timestamps']
    for turn in row['turns']:
        seconds=int(times[turn['frame_index']]);turn['ts']=f'{seconds//60:02d}:{seconds%60:02d}'
    if row['memory'] is not None:row['memory']['time_seconds']=times[row['memory']['frame_index']]
    row['_source_indices']=spec['source_indices']
    return row


def categorical_loss(logits,reference_log_probs,actions,advantages,*,total_trajectories,scales):
    import torch.nn.functional as F
    logp=F.log_softmax(logits.float(),dim=-1);p=logp.exp()
    kl=(p*(logp-reference_log_probs.detach().float())).sum(-1)
    entropy=-(p*logp).sum(-1);selected=logp.gather(-1,actions[:,None]).squeeze(-1)
    loss=(scales*(-advantages.detach()*selected+.02*kl-.001*entropy)).sum()/(total_trajectories*32)
    return loss,kl.detach(),entropy.detach()
