#!/usr/bin/env python3
"""Capture exact causal training contexts during ordinary raw-argmax rollouts."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

import eval_jevany_streaming as rollout
import eval_qwen35_development_sweep as sweep
from qwen35_sft_campaign import rows, write


def main():
    args = sys.argv[1:]
    config = Path(args[args.index('--conditions') + 1])
    conditions = json.loads(config.read_text())
    if len(conditions) != 1 or conditions[0].get('benchmark'):
        raise ValueError('context capture is restricted to one training-only manifest')
    condition = conditions[0]
    sources = {r['question_id']: r for r in rows(condition['manifest'])}
    if any(r.get('split') != 'train' for r in sources.values()):
        raise ValueError('refusing to collect non-training contexts')
    directory = Path(condition['output']) / 'contexts'
    directory.mkdir(parents=True, exist_ok=True)
    original_generate = rollout.web.generate
    original_inputs = rollout.decision_inputs
    current = {}

    def inputs(runtime, messages, vision, **kwargs):
        index = len(current['records'])
        source = current['source']
        start = max(0, index - 31)
        indices = source['inference_source_indices'][start:index + 1]
        times = source['inference_timestamps'][start:index + 1]
        serialized = []
        for message in messages:
            serialized.append(dict(role=message['role'], content=[
                {'type': 'image'} if part['type'] == 'image' else dict(part)
                for part in message['content']]))
        if sum(p['type'] == 'image' for m in serialized for p in m['content']) != len(indices):
            raise ValueError('causal context/frame count mismatch')
        current['records'].append(dict(
            episode_id=source['question_id'], turn_index=index, video_id=source['video_id'],
            video_path=source['resolved_video_path'], question=source['question'],
            frame_index=indices[-1], window_start=indices[0], memory=None,
            turns=[dict(frame_index=i, ts=f'{int(t)//60:02d}:{int(t)%60:02d}')
                   for i, t in zip(indices, times, strict=True)],
            controller_messages=serialized, split='train',
            source_question_id=source['question_id'], rollout_decision_index=index))
        return original_inputs(runtime, messages, vision, **kwargs)

    def generate(runtime, example, **kwargs):
        current.clear()
        current.update(source=sources[example['question_id']], records=[])
        result = original_generate(runtime, example, **kwargs)
        if len(current['records']) != current['source']['inference_frame_count']:
            raise ValueError('incomplete training context capture')
        name = hashlib.sha256(example['question_id'].encode()).hexdigest()
        write(directory / f'{name}.json', dict(question_id=example['question_id'],
                                              records=current['records']), frozen=True)
        return result

    rollout.decision_inputs = inputs
    rollout.web.generate = generate
    sweep.main()


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
