"""Deterministic training-only ESTP action targets and balanced SFT sampling."""
import bisect
import copy
import random

EXPLICIT_TASKS = (
    'Object Recognition', 'Attribute Perception', 'Text-Rich Understanding',
    'Object Localization', 'Object State Change Recognition',
    'Ego Object Localization', 'Ego Object State Change Recognition', 'Action Recognition',
)


def action_plan(spec):
    """One response at the first reachable native-window sample; Hold thereafter.

    Overlapping windows requiring two simultaneous answers are rejected rather
    than silently merged. No target or observation uses a widened answer window.
    """
    times = spec['timestamps']
    labels = ['Silence'] * len(times)
    responses = {}
    for answer in sorted(spec['gold_answers'], key=lambda a: a['reply_timespan']):
        left, right = answer['reply_timespan']
        if not left < right:
            raise ValueError('nonpositive_answer_window')
        first = bisect.bisect_left(times, left - 1e-7)
        if first >= len(times) or times[first] > right + 1e-7:
            raise ValueError('unreachable_supervised_answer')
        indices = range(first, bisect.bisect_right(times, right + 1e-7))
        if any(labels[i] != 'Silence' for i in indices):
            raise ValueError('overlapping_supervised_windows')
        for i in indices:
            labels[i] = 'Hold'
        labels[first] = 'Response'
        responses[first] = answer['content']
    return labels, responses


def training_frames(spec, seed=42):
    """Keep all responses, up to one Hold and four Silence examples per event."""
    labels, _ = action_plan(spec)
    rng = random.Random(f"{seed}:{spec['span_id']}")
    positives = [i for i, label in enumerate(labels) if label == 'Response']
    chosen = list(positives)
    for label, multiplier in [('Hold', 1), ('Silence', 4)]:
        candidates = [i for i, value in enumerate(labels) if value == label]
        chosen.extend(rng.sample(candidates, min(len(candidates), multiplier * len(positives))))
    return sorted(chosen)


def supervised_row(spec, frame):
    """Teacher-forced past answers only; current target answer is never an input."""
    labels, responses = action_plan(spec)
    start = max(0, frame - 31)
    turns = []
    for i in range(start, frame + 1):
        seconds = int(spec['timestamps'][i])
        label = labels[i]
        tag = {'Silence': '</Silence>', 'Hold': '</Standby>', 'Response': '</Response>'}[label]
        emit = tag + (' ' + responses[i] if label == 'Response' else '')
        turns.append(dict(frame_index=spec['source_indices'][i], ts=f'{seconds//60:02d}:{seconds%60:02d}',
                          emit=emit if i < frame else '', state=label if i < frame else 'Silence'))
    previous = [i for i in responses if i < start]
    memory = None
    if previous:
        last = max(previous)
        memory = dict(frame_index=spec['source_indices'][last],
                      time_seconds=spec['timestamps'][last], content=responses[last])
    return dict(episode_id=spec['span_id'], video_id=spec['video_id'], video_path=spec['video_path'],
                question=spec['question'], frame_index=spec['source_indices'][frame],
                memory=memory, turns=turns, label=labels[frame], split='train', source='estp_it')


class CyclingPermutation:
    """Random-access stream; exact reconstruction from consumed-example count."""
    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError('empty training source')
        self.size, self.seed, self.epoch, self.order = size, seed, None, None

    def at(self, position):
        epoch, offset = divmod(position, self.size)
        if epoch != self.epoch:
            self.order = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(self.order)
            self.epoch = epoch
        return self.order[offset]
