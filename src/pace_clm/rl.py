"""Categorical, on-policy RL contracts; independent of model/runtime loading."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F


def mask_padding_states(hidden_states, attention_mask):
    """Mask Qwen recurrent inputs at batch one too, including left-padded decoding."""
    if attention_mask is not None and attention_mask.shape[1] > 1:
        return (hidden_states*attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def ready_collection(states, *, max_lead, min_ready_fraction, batch_size):
    """Bound progress differences so slow answers do not fragment every batch."""
    active=[s for s in states if s['frame']<s['spec']['rollout_stop']]
    if not active:return [],False,True
    progress=lambda s:s['frame']-s['spec']['rollout_start']
    slowest=min(map(progress,active))
    ready=[s for s in active if not s['pending'] and progress(s)<=slowest+max_lead]
    minimum=math.ceil(min(batch_size,len(active))*min_ready_fraction)
    wait=any(s['pending'] for s in active) and len(ready)<minimum
    return ready,wait,False


def pad_inputs(inputs: dict, length: int, pad_id: int) -> dict:
    current = inputs['input_ids'].shape[-1]
    if length < current:
        raise ValueError('padding cannot truncate an observation')
    result = dict(inputs)
    for name, fill in (('input_ids', pad_id), ('attention_mask', 0), ('mm_token_type_ids', 0)):
        result[name] = F.pad(inputs[name], (0,length-current), value=fill)
    return result


def parity_metrics(current_logp, old_logp, actions):
    """Distributional BF16 parity; near-zero actions have ill-conditioned log errors."""
    current, old = current_logp.detach().float(), old_logp.detach().float()
    error = (current-old).abs()
    kl = (old.exp()*(old-current)).sum(-1)
    selected = error.gather(-1,actions[:,None]).squeeze(-1)
    relevant = old.gather(-1,actions[:,None]).squeeze(-1).exp() >= .01
    return {'all_logp_max':float(error.max()),
            'probability_max':float((current.exp()-old.exp()).abs().max()),
            'kl_max':float(kl.max()), 'kl_sum':float(kl.sum()),
            'selected_logp_max':float(selected[relevant].max()) if relevant.any() else 0.}


def observation(spec: dict, history: list[dict], frame: int) -> dict:
    start = max(spec['decode_start'], frame - 31)
    recent = [dict(t) for t in history if start <= t['frame_index'] < frame]
    for turn in recent:
        index = turn['frame_index']
        turn['ts'] = f'{index//60:02d}:{index%60:02d}'
    if [t['frame_index'] for t in recent] != list(range(start, frame)):
        raise ValueError('history has gaps or is out of order')
    responses = ([spec['initial_memory']] if spec.get('initial_memory') else [])
    for turn in history:
        if turn['emit'].startswith('</Response>'):
            responses.append({'frame_index': turn['frame_index'],
                              'time_seconds': float(turn['frame_index']),
                              'content': turn.get('memory_content',turn['emit'][len('</Response>'):].strip())})
    aged = [r for r in responses if r['frame_index'] < start]
    memory = max(aged, key=lambda r: r['frame_index']) if aged else None
    recent.append({'frame_index': frame, 'ts': f'{frame//60:02d}:{frame%60:02d}',
                   'emit': '', 'state': 'Silence'})
    return {'episode_id': spec['span_id'], 'video_id': spec['video_id'],
            'video_path': spec['video_path'], 'question': spec['question'],
            'frame_index': frame, 'memory': copy.deepcopy(memory), 'turns': recent,
            'split': 'train'}


def leave_one_out(rewards: list[float]) -> list[float]:
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError('need at least two finite group rewards')
    total = sum(rewards)
    return [r - (total-r)/(len(rewards)-1) for r in rewards]


def categorical_loss(logits, reference_log_probs, actions, advantages, *,
                     total_trajectories: int, horizon: int = 32,
                     kl_beta: float = .02, entropy_beta: float = .001):
    """Sum contribution; divide globally, never by unequal rank-local counts."""
    logp = F.log_softmax(logits.float(), dim=-1)
    ref = reference_log_probs.detach().float()
    p = logp.exp()
    kl = (p * (logp-ref)).sum(-1)
    entropy = -(p*logp).sum(-1)
    selected = logp.gather(-1, actions[:, None]).squeeze(-1)
    loss = (-advantages.detach()*selected + kl_beta*kl - entropy_beta*entropy).sum()
    return loss/(total_trajectories*horizon), kl.detach(), entropy.detach()


def reward_ledger(gold: list[dict], fires: list[dict], semantic: list[list[float]],
                  repetition: list[float], *, window: float = 4.) -> list[dict]:
    if len(fires) != len(semantic) or len(fires) != len(repetition):
        raise ValueError('incomplete judge results')
    if any(len(scores) != len(gold) for scores in semantic):
        raise ValueError('incomplete semantic matrix')
    if not all(math.isfinite(s) and 0 <= s <= 1
               for scores in [*semantic, repetition] for s in scores):
        raise ValueError('invalid judge score')
    matched, ledger = set(), []
    denominator = max(1, len(gold))
    for i in sorted(range(len(fires)), key=lambda i: fires[i]['frame_index']):
        time = fires[i]['frame_index']
        eligible = sorted((j for j, g in enumerate(gold)
                           if j not in matched and g['time_seconds'] <= time <
                           g['time_seconds']+window and semantic[i][j] >= .5),
                          key=lambda j: gold[j]['time_seconds'])
        if eligible:
            j = eligible[0]
            matched.add(j)
            delay = time-gold[j]['time_seconds']
            value = semantic[i][j]*max(0., 1-delay/window)
            kind, settled = 'matched', gold[j]['time_seconds']+window
        else:
            # A distinct eligible recurring event was given first priority above.
            duplicate = repetition[i] >= .5 or any(semantic[i][j] >= .5 for j in matched)
            value, kind, settled = (-.35, 'duplicate', time) if duplicate else (-.25, 'unmatched', time)
            j = None
        ledger.append({'kind': kind, 'fire': i, 'gold': j, 'settled_at': settled,
                       'value': value/denominator})
    for j, g in enumerate(gold):
        if j not in matched:
            ledger.append({'kind': 'missed', 'fire': None, 'gold': j,
                           'settled_at': g['time_seconds']+window, 'value': -.5/denominator})
    return sorted(ledger, key=lambda r: r['settled_at'])
