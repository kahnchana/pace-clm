"""Auditable ESTP single-question episodes and step-32 continuation schedules."""
import copy
import hashlib
import math
import random
from collections import Counter

from pace_clm.rl_mixed import validate_spec


def priority(value):
    return hashlib.sha256(('estp-it-split-42:' + value).encode()).hexdigest()


def video_splits(videos):
    ordered = sorted(videos, key=priority)
    held = round(len(ordered) * .1)
    return dict(train=ordered[:-2 * held], dev=ordered[-2 * held:-held], test=ordered[-held:])


def estp_spec(video, clip, index, qa, path, *, training=True, frame_count=None, benchmark=False):
    """Preserve native windows; train only on complete, reachable SQ targets.

    Frames remain 1 FPS, starting at the first physical 2-FPS frame at or
    after the annotated clip start. No window is widened or gold-dependent
    sampling phase selected. Missing/invalid annotations are never negatives.
    """
    start = float(qa.get('clip_start_time', qa.get('start_time', float('nan'))))
    end = float(qa.get('clip_end_time', qa.get('end_time', float('nan'))))
    question = qa.get('question')
    users = [t for t in qa.get('conversation', []) if t.get('role') == 'user']
    if not question:
        if training or len(users) != 1:
            raise ValueError('not_single_question')
        question = users[0].get('content')
    if not question or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError('missing_or_invalid_clip_bounds')
    query = float(qa.get('question_time', qa.get('query_time', start)))
    if query > start or (training and users):
        raise ValueError('dynamic_question_requires_separate_protocol')
    first = math.ceil(start * 2 - 1e-7)
    stop = math.ceil(end * 2 - 1e-7)
    indices = list(range(first, stop, 2))
    if benchmark and frame_count is not None:
        indices = [i for i in indices if i < frame_count]
    if not indices or (frame_count is not None and indices[-1] >= frame_count):
        raise ValueError('clip_outside_media')
    times = [i / 2 - start for i in indices]
    gold = []
    for turn in qa.get('conversation', []):
        if turn.get('role') != 'assistant':
            continue
        left, right = float(turn['start_time']), float(turn['end_time'])
        valid = left <= right if benchmark else start <= left < right <= end
        if not math.isfinite(left) or not math.isfinite(right) or not valid:
            raise ValueError('invalid_or_unbounded_answer_window')
        content = str(turn.get('content', '')).strip()
        if not content:
            raise ValueError('empty_gold_answer')
        if training:
            k = max(0, math.ceil(left - first / 2 - 1e-7))
            if k >= len(indices) or first / 2 + k > right + 1e-7:
                raise ValueError('answer_window_unreachable_at_1fps')
        gold.append(dict(content=content, reply_timespan=[left - start, right - start],
                         related_timespan=[left - start, right - start]))
    if not gold:
        raise ValueError('missing_gold_answers')
    result = dict(span_id=f'estp-{ "it" if training else "eval"}:{video}/{clip}/{index}',
                  native_question_id=f'{video}/{clip}/{index}', domain='ego',
                  data_source='estp_it' if training else 'estp_bench', video_id=video,
                  video_path=str(path), question=question, decode_start=0,
                  rollout_start=0, rollout_stop=len(indices), rollout_frames=len(indices),
                  prefix_turns=[], initial_memory=None, source_indices=indices, timestamps=times,
                  gold_answers=gold, gold_points=[dict(time_seconds=g['reply_timespan'][0],
                  content=g['content']) for g in gold], source_fps=2, clip_start_time=start,
                  clip_end_time=end, clip_id=clip, qa_index=index,
                  task_type=str(qa.get('Task Type', '')).strip())
    if not benchmark:
        validate_spec(result)
    return result


def continuation_schedule(episodes, prefix, seed=42):
    """Append 96 updates: 384 original, 192 WEB, 81 EGO and 111 ESTP-IT."""
    if len(prefix) != 32:
        raise ValueError('requires the complete step-32 schedule')
    used = [i for batch in prefix for i in batch['episode_indices']]
    if len(used) != len(set(used)) or len(used) != 256:
        raise ValueError('parent schedule repeats or omits episodes')
    ids = [s['span_id'] for s in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode identity')
    rng = random.Random(seed)
    pools = {d: [i for i, s in enumerate(episodes) if i not in set(used)
                 and s.get('data_source', s['domain']) == d]
             for d in ('ours', 'web', 'ego', 'estp_it')}
    for pool in pools.values():
        rng.shuffle(pool)
    required = dict(ours=384, web=192, ego=81, estp_it=111)
    if any(len(pools[d]) < n for d, n in required.items()):
        raise ValueError('insufficient unused episodes for 128 updates')
    if len({episodes[i]['video_id'] for i in pools['estp_it'][:111]}) != 111:
        raise ValueError('ESTP episodes must come from 111 distinct training videos')
    positions = set(rng.sample(range(96), 81))
    cursor = Counter()
    result = copy.deepcopy(prefix)
    for update in range(96):
        counts = dict(ours=4, web=2, ego=int(update in positions),
                      estp_it=1 if update in positions else 2)
        batch = []
        for d, n in counts.items():
            batch += pools[d][cursor[d]:cursor[d] + n]
            cursor[d] += n
        previous = result[-1]
        delta = {d: sum(4 * episodes[i]['rollout_frames'] for i in batch
                        if episodes[i]['domain'] == d) for d in ('ours', 'web', 'ego')}
        after = previous['exposure_after'] + sum(delta.values())
        credits = previous['anchor_start'] + previous['anchor_count']
        result.append(dict(episode_indices=batch, decisions=sum(delta.values()),
                           exposure_before=previous['exposure_after'], exposure_after=after,
                           groups_before=previous['groups_after'], groups_after=previous['groups_after'] + 8,
                           anchor_start=credits, anchor_count=after * 126 // 4096 - credits,
                           domain_decisions=delta, cumulative_domain_decisions={
                               d: previous['cumulative_domain_decisions'][d] + delta[d] for d in delta}))
    assert dict(cursor) == required
    return result
