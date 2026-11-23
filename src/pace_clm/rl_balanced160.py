"""Four-source episode balancing for the step-128 to step-160 phase."""
import copy
import random
from collections import Counter

SOURCES = ('ours', 'web', 'ego', 'estp_it')


def source(spec):
    return spec.get('data_source', spec['domain'])


def balanced_schedule(episodes, prefix, seed=42):
    """Preserve the parent schedule; prefer unseen episodes, reuse EGO explicitly."""
    if len(prefix) != 128:
        raise ValueError('requires complete step-128 prefix')
    ids = [s['span_id'] for s in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode identities')
    used = {i for b in prefix for i in b['episode_indices']}
    rng = random.Random(seed)
    pools = {}
    for name in SOURCES:
        fresh = [i for i, s in enumerate(episodes) if source(s) == name and i not in used]
        seen = [i for i, s in enumerate(episodes) if source(s) == name and i in used]
        rng.shuffle(fresh)
        rng.shuffle(seen)
        pools[name] = fresh + seen
        if len(pools[name]) < 64:
            raise ValueError('insufficient source episodes: ' + name)
    result = copy.deepcopy(prefix)
    for update in range(32):
        indices = [i for name in SOURCES for i in pools[name][2*update:2*update+2]]
        delta = {d: sum(4 * episodes[i]['rollout_frames'] for i in indices
                        if episodes[i]['domain'] == d) for d in ('ours', 'web', 'ego')}
        previous = result[-1]
        after = previous['exposure_after'] + sum(delta.values())
        credits = previous['anchor_start'] + previous['anchor_count']
        result.append(dict(episode_indices=indices, decisions=sum(delta.values()),
                           exposure_before=previous['exposure_after'], exposure_after=after,
                           groups_before=previous['groups_after'], groups_after=previous['groups_after']+8,
                           anchor_start=credits, anchor_count=after*126//4096-credits,
                           domain_decisions=delta, cumulative_domain_decisions={
                               d: previous['cumulative_domain_decisions'][d]+delta[d] for d in delta}))
    return result


def attach_source_loss_scales(traces):
    """Give each source weight 0.25; retain constant-32 decision normalization."""
    counts = Counter(source(t['spec']) for t in traces)
    if set(counts) != set(SOURCES):
        raise ValueError('global batch must contain all four sources')
    for trace in traces:
        scale = .25 * len(traces) / counts[source(trace['spec'])]
        for step in trace['steps']:
            step['loss_scale'] = scale
    return dict(counts)
