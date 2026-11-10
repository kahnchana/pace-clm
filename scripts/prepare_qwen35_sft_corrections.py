#!/usr/bin/env python3
"""Training-only rollout cohort and GPT-verified history-dependent SFT targets."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from qwen35_sft_campaign import ROOT, rows, sha, write

SG = ROOT.parent / 'stream-gemma'
sys.path[:0] = [str(SG / 'src'), str(SG)]

RUBRIC = (
    'You are verifying supervision for a streaming video controller. The current event is '
    'independently annotated as visible now. You receive its annotation and only answers '
    'the model already emitted since this event began, strictly before the current frame, '
    'and still visible in its causal window or last-response memory. '
    'Decide whether those prior answers already correctly and sufficiently report this event '
    'for the standing question. Incorrect, vague, unrelated, or contradictory answers do not '
    'cover it. Equivalent factual paraphrases do. Do not use future information. '
    'Return exactly JSON {"covered": true or false, "reason": "brief explanation"}.'
)


def cohort(out):
    from prepare_jevany_decisions import make_args
    from prepare_qwen35_control import aliases, exclusions
    from stream_gemma.data import _conversation_and_range, _parse_record, _paint_kwargs
    from stream_gemma.episode import flatten_stream_conversation, paint_v24_states
    target = out / 'data/rollout_train.jsonl'
    if target.exists():
        audit = json.loads((out / 'data/rollout_train_audit.json').read_text())
        if sha(target) != audit['sha256']:
            raise ValueError('training rollout manifest changed')
        return audit
    allowed = {str(Path(r['video_path']).resolve()) for r in rows(out / 'data/anchor.jsonl')}
    denied, _ = exclusions()
    metadata = json.loads((SG / 'datasets/markii/metadata.json').read_text())
    candidates = []
    args = make_args()
    for raw_index, line in enumerate((SG / 'datasets/markii/train.jsonl').open()):
        relative, question, text = _parse_record(json.loads(line))
        meta = metadata.get(str(Path(relative).with_suffix('')))
        if not meta:
            continue
        path = Path(meta.get('path') or SG / 'datasets/markii/videos' / relative)
        if str(path.resolve()) not in allowed or aliases(str(path)) & denied:
            continue
        built = _conversation_and_range(question=question, text_stream=text, metadata=meta,
                                        source_fps=1, keep_trailing_silence=True)
        if not built:
            continue
        conversation, source_range = built
        # Whole annotated clips retain event boundaries, without truncation at 32 frames.
        if not 16 <= len(source_range) <= 256:
            continue
        painted = paint_v24_states(flatten_stream_conversation(conversation), **_paint_kwargs(args))
        if not painted.kept_events:
            continue
        question_id = hashlib.sha256(f'sft-correction-42:{path}:{question}'.encode()).hexdigest()
        record = dict(question_id=question_id, video_id=str(path), resolved_video_path=str(path),
                      question=question, raw_index=raw_index, split='train',
                      inference_source_indices=list(source_range),
                      inference_timestamps=[float(i) for i in source_range],
                      inference_frame_count=len(source_range),
                      annotated_events=[dict(start=source_range.start + e.start,
                                             end=source_range.start + e.end,
                                             content=e.content) for e in painted.kept_events],
                      gold_states=painted.states)
        if len(painted.states) != len(source_range):
            raise ValueError('annotation/frame count mismatch')
        candidates.append(record)
    candidates.sort(key=lambda r: r['question_id'])
    selected, used = [], set()
    for record in candidates:
        canonical = str(Path(record['resolved_video_path']).resolve())
        if canonical in used:
            continue
        used.add(canonical)
        selected.append(record)
        if len(selected) == 256:
            break
    if len(selected) != 256:
        raise ValueError(f'insufficient training-only rollout videos: {len(selected)}')
    target.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in selected))
    audit = dict(videos=256, decisions=sum(r['inference_frame_count'] for r in selected),
                 sha256=sha(target), annotations_sha256=sha(SG / 'datasets/markii/train.jsonl'),
                 selection='Lowest frozen seeded question hash, one record per video, 16–256 frames, nonempty events.',
                 training_only=True, benchmark_validation_overlap=0, future_events_not_supplied_to_judge=True)
    write(out / 'data/rollout_train_audit.json', audit, frozen=True)
    return audit


def action_target(active_event, prior_answers, covered):
    if active_event is None:
        return 'Silence'
    if not prior_answers:
        return 'Response'
    if not isinstance(covered, bool):
        raise ValueError('a history-dependent label requires a valid boolean GPT judgment')
    return 'Hold' if covered else 'Response'


def visible_prior_answers(answers, frame, window_start):
    """Match the deployed 32-frame window plus its single aged-out response memory."""
    past = [answer for answer in answers if float(answer['time']) < frame]
    in_window = [answer for answer in past if float(answer['time']) >= window_start]
    aged_out = [answer for answer in past if float(answer['time']) < window_start]
    if aged_out:
        in_window = [max(aged_out, key=lambda answer: float(answer['time'])), *in_window]
    return in_window


def corrected(out):
    from judge_exp10_utility import load_gateway_key
    from zs_infer.scoring import web_sync
    from pace_clm.jevany_streaming import decision_messages
    from stream_gemma.episode import STATE_TAGS
    from stream_gemma.v24_memory import normalize_memory_content
    output_audit = out / 'data/corrections_audit.json'
    if output_audit.exists():
        result = json.loads(output_audit.read_text())
        for name in ('rollout_control', 'rollout_corrected'):
            if sha(out / f'data/{name}.jsonl') != result[name + '_sha256']:
                raise ValueError('corrective data changed')
        return result
    sources = {r['question_id']: r for r in rows(out / 'data/rollout_train.jsonl')}
    rollout_dir = out / 'training_rollouts'
    predictions = {}
    for shard in sorted(rollout_dir.glob('predictions.rank*.jsonl')):
        for r in rows(shard):
            if r['question_id'] in predictions:
                raise ValueError('duplicate training rollout')
            predictions[r['question_id']] = r
    if set(predictions) != set(sources):
        raise ValueError('training rollout coverage incomplete')
    pairs, requests = [], {}
    for context_file in sorted((rollout_dir / 'contexts').glob('*.json')):
        payload = json.loads(context_file.read_text())
        source = sources[payload['question_id']]
        prediction = predictions[payload['question_id']]
        events = source['annotated_events']
        for row in payload['records']:
            frame = row['frame_index']
            active = next((e for e in events if e['start'] <= frame <= e['end']), None)
            visible = visible_prior_answers(prediction['model_response_list'], frame, row['window_start'])
            prior = [r for r in visible
                     if active and active['start'] <= float(r['time']) < frame]
            request_key = None
            if active and prior:
                # Request contains past answers and the current annotation only.
                content = json.dumps(dict(question=row['question'], current_event=active['content'],
                                          prior_answers=[p['content'] for p in prior]), sort_keys=True)
                request_key = hashlib.sha256((RUBRIC + content).encode()).hexdigest()
                requests[request_key] = dict(custom_id=request_key, method='POST', url='/v1/chat/completions',
                    body=dict(model='gpt-4.1', temperature=0, max_tokens=256,
                              response_format={'type': 'json_object'},
                              messages=[dict(role='system', content=RUBRIC), dict(role='user', content=content)]))
            pairs.append((row, source, active, prior, request_key))
    if len(pairs) != sum(r['inference_frame_count'] for r in sources.values()):
        raise ValueError('captured-context coverage incomplete')
    judge = out / 'correction_judge'
    judge.mkdir(exist_ok=True)
    request_file = judge / 'requests.jsonl'
    text = ''.join(json.dumps(requests[k], sort_keys=True) + '\n' for k in sorted(requests))
    if request_file.exists() and request_file.read_text() != text:
        raise ValueError('frozen corrective judge requests changed')
    request_file.write_text(text)
    response_file = judge / 'responses.jsonl'
    if not response_file.exists():
        load_gateway_key()
        web_sync(dict(batch_input=request_file, batch_output=response_file),
                 concurrency=16, max_attempts=5, timeout=120.)
    judgments = {}
    for response in rows(response_file):
        if response['custom_id'] in judgments:
            raise ValueError('duplicate corrective judgment')
        body = response['response']['body']['choices'][0]['message']['content']
        value = json.loads(body)
        if type(value.get('covered')) is not bool or not isinstance(value.get('reason'), str):
            raise ValueError('invalid corrective judgment; refusing guessed labels')
        judgments[response['custom_id']] = value
    if set(judgments) != set(requests):
        raise ValueError('corrective judgment coverage incomplete')
    gold_rows, corrected_rows = [], []
    changed = 0
    for row, source, active, prior, key in pairs:
        target = action_target(active, prior, judgments[key]['covered'] if key else None)
        corrected_row = dict(row, label=target, corrective_judge_key=key,
                             target_provenance='Current annotated event plus GPT verification of causal generated-answer coverage')
        # A matched gold-history example at exactly the same video/frame position.
        events = source['annotated_events']
        origin = source['inference_source_indices'][0]
        start = row['window_start']
        prior_memory = [e for e in events if e['start'] < start]
        memory = prior_memory[-1] if prior_memory else None
        gold = {k: v for k, v in row.items() if k not in ('controller_messages',)}
        gold['memory'] = None if memory is None else dict(frame_index=memory['start'],
            time_seconds=float(memory['start']), content=normalize_memory_content(memory['content']))
        gold['turns'] = []
        for turn in row['turns']:
            index = turn['frame_index']
            state = source['gold_states'][index - origin]
            emission = STATE_TAGS[state]
            event = next((e for e in events if e['start'] == index), None)
            if state == 'Response':
                if event is None:
                    raise ValueError('missing gold event at Response')
                emission += ' ' + normalize_memory_content(event['content'])
            gold['turns'].append(dict(turn, state=state, emit=emission, learn=True))
        state = source['gold_states'][row['frame_index'] - origin]
        gold['label'] = 'Hold' if state == 'Standby' else state
        # Confirm the custom-message path exactly preserves the original observed input.
        decision_messages(gold, [None] * len(gold['turns']))
        changed += target != gold['label']
        gold_rows.append(gold)
        corrected_rows.append(corrected_row)
    anchor = list(rows(out / 'data/replay.jsonl'))[:65536]
    if len(anchor) != 65536 or len(pairs) < 1024:
        raise ValueError('insufficient corrective training coverage')
    order = list(range(len(pairs)))
    random.Random(42).shuffle(order)
    control, intervention = [], []
    used = []
    for index, row in enumerate(anchor):
        if index % 4:
            control.append(row)
            intervention.append(row)
        else:
            chosen = order[(index // 4) % len(order)]
            used.append(chosen)
            control.append(gold_rows[chosen])
            intervention.append(corrected_rows[chosen])
    for name, values in [('rollout_control', control), ('rollout_corrected', intervention)]:
        (out / f'data/{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in values))
    result = dict(rollout_control_sha256=sha(out / 'data/rollout_control.jsonl'),
                  rollout_corrected_sha256=sha(out / 'data/rollout_corrected.jsonl'),
                  training_videos=len(sources), captured_decisions=len(pairs), judgment_requests=len(requests),
                  judge_model='gpt-4.1', rubric_sha256=hashlib.sha256(RUBRIC.encode()).hexdigest(),
                  request_sha256=sha(request_file), response_sha256=sha(response_file),
                  pairs_with_changed_labels=changed, mixed_presentations=65536,
                  corrective_presentations=len(used), corrective_unique_positions=len(set(used)),
                  control_labels=dict(Counter(r['label'] for r in control)),
                  intervention_labels=dict(Counter(r['label'] for r in intervention)),
                  matched_source_positions=True, raw_argmax_inference_unchanged=True)
    result['supervision_history'] = 'Only answers visible in the recorded window or single aged-out response memory; inaccessible earlier answers never determine targets.'
    write(output_audit, result, frozen=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', action='store_true')
    args = parser.parse_args()
    print(json.dumps(corrected(args.output) if args.label else cohort(args.output)), flush=True)
