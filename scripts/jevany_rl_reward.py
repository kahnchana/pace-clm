"""Versioned training reward using the unchanged official cumulative-answer scorer.

Only text is sent to the user-authorized GPT gateway. All generated physical
fires remain in the precision denominator; prefix responses are context only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / 'stream-gemma'
sys.path[:0] = [str(ROOT / 'scripts'), str(SG)]
from zs_infer.pauc_f import OFFICIAL_SCORER, _load_official_scorer, score_pauc_f

VERSION = 'pauc-f-aligned-gpt-v1'
MODEL = 'gpt-4.1'
OFFICIAL = _load_official_scorer(OFFICIAL_SCORER)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def contract():
    return {'version': VERSION, 'judge': MODEL, 'omega': .5,
            'event_reward': 'official clip PAUC-F / 100',
            'no_event_reward': '-0.25 * physical fires',
            'window_seconds': 4, 'timestamp_origin': 'absolute',
            'window_end': 'inclusive, unchanged official request builder',
            'initial_pauc_score': .5, 'max_pauc_score': 2,
            'zero_fire_precision': 0, 'prefix_responses_scored': False,
            'overlapping_windows': 'preserved; official assignment and aggregation unchanged',
            'official_sha256': sha(OFFICIAL_SCORER),
            'pauc_f_sha256': sha(SG / 'zs_infer/pauc_f.py'),
            'request_builder_sha256': sha(SG / 'zs_infer/scoring.py')}


def artifacts(traces):
    predictions, gold = [], []
    ids = set()
    for trace in traces:
        spec = trace['spec']
        qid = f"{spec['span_id']}:{trace['trajectory_id']}"
        if qid in ids:
            raise ValueError('duplicate trajectory identity')
        ids.add(qid)
        start, stop = spec['rollout_start'], spec['rollout_stop']
        if any(not start <= g['time_seconds'] or g['time_seconds'] + 4 > stop
               for g in spec['gold_points']):
            raise ValueError('boundary-censored annotated window')
        if len(trace['steps']) != stop - start or [s['row']['frame_index'] for s in trace['steps']] != list(range(start, stop)):
            raise ValueError('incomplete or unordered trajectory')
        responses = [{'role': 'assistant', 'time': s['row']['frame_index'],
                      'content': s['answer']} for s in trace['steps'] if s['action'] == 2]
        if any(not r['content'].strip() for r in responses):
            raise ValueError('empty generated answer')
        predictions.append({'question_id': qid, 'model_response_list': responses})
        gold.append({'question_id': qid,
                     'conversation': [{'role': 'user', 'content': spec['question']}],
                     'answer': [{'content': g['content'],
                                 'reply_timespan': [g['time_seconds'], g['time_seconds'] + 4],
                                 'related_timespan': [g['time_seconds'], g['time_seconds'] + 4]}
                                for g in spec['gold_points']]})
    return predictions, gold


def request_key(body):
    return hashlib.sha256(json.dumps({'body': body, 'contract': contract()},
                                    sort_keys=True).encode()).hexdigest()


def judge_requests(requests, cache, *, concurrency=16):
    """Cache complete requests/model/rubric, validate every response, retain raw output."""
    from judge_exp10_utility import load_gateway_key
    from zs_infer.scoring import _openai_client, _chat_completion_with_retry
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cache)
    db.execute('CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, request TEXT, response TEXT)')
    by_key, hits = {}, 0
    for request in requests:
        key = request_key(request['body'])
        by_key[key] = request['body']
    found, pending = {}, {}
    for key, body in by_key.items():
        row = db.execute('SELECT response FROM judgments WHERE key=?', (key,)).fetchone()
        if row:
            found[key] = json.loads(row[0])
            hits += 1
        else:
            pending[key] = body
    if pending:
        load_gateway_key()
        client = _openai_client()

        def one(key, body):
            result = _chat_completion_with_retry(client, body, max_attempts=5, timeout=120)
            raw = result.model_dump(mode='json')
            content = str(result.choices[0].message.content or '').strip()
            if content not in ('1', '2', '3'):
                raise ValueError(f'invalid official judgment for {key}: {content!r}')
            return key, raw

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, key, body) for key, body in pending.items()]
            for completed, future in enumerate(as_completed(futures), 1):
                key, raw = future.result()
                found[key] = raw
                db.execute('INSERT OR REPLACE INTO judgments VALUES (?, ?, ?)',
                           (key, json.dumps(by_key[key]), json.dumps(raw)))
                # FSx fsync is expensive. Persist bounded chunks, retaining at
                # most fifteen successful calls to repeat after a hard crash.
                if completed % 16 == 0:
                    db.commit()
                if completed % 64 == 0:
                    print(json.dumps({'gpt_completed': completed, 'gpt_pending_total': len(pending)}), flush=True)
            db.commit()
    outputs = []
    for request in requests:
        raw = found[request_key(request['body'])]
        if str(raw['choices'][0]['message']['content']).strip() not in ('1', '2', '3'):
            raise ValueError('invalid cached judgment')
        outputs.append({'custom_id': request['custom_id'],
                        'response': {'status_code': 200, 'body': raw}, 'error': None})
    db.close()
    return outputs, {'requests': len(requests), 'unique_requests': len(by_key),
                     'cached': hits, 'new_requests': len(pending), 'backend': MODEL}


def score_traces(traces, directory, cache, *, supplied_outputs=None):
    """Build/process with official code, then score each clip with score_pauc_f itself."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predictions, gold = artifacts(traces)
    pred_path, gold_path = directory / 'predictions.jsonl', directory / 'gold.json'
    pred_path.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    write(gold_path, gold)
    request_path, output_path = directory / 'requests.jsonl', directory / 'responses.jsonl'
    OFFICIAL.create_openai_batch_input(str(pred_path), str(gold_path), str(request_path), -1, MODEL)
    requests = [json.loads(line) for line in request_path.read_text().splitlines()]
    if supplied_outputs is None:
        outputs, report = judge_requests(requests, cache)
    else:
        outputs, report = supplied_outputs, {'requests': len(requests), 'backend': 'test'}
    if len(outputs) != len(requests) or {r['custom_id'] for r in outputs} != {r['custom_id'] for r in requests}:
        raise ValueError('incomplete or duplicate GPT judge batch')
    outputs = copy.deepcopy(outputs)
    for output in outputs:
        message = output['response']['body']['choices'][0]['message']
        content = str(message['content']).strip()
        if content not in ('1', '2', '3'):
            raise ValueError('invalid official judgment')
        message['content'] = content
    output_path.write_text(''.join(json.dumps(r) + '\n' for r in outputs))
    scored_path = directory / 'scores.json'
    OFFICIAL.process_openai_batch_output(str(pred_path), str(gold_path), str(output_path), str(scored_path))
    scored = json.loads(scored_path.read_text())
    by_id = {item['question_id']: item for item in scored}
    clip_dir = directory / 'clips'
    clip_dir.mkdir(exist_ok=True)
    for trace, prediction in zip(traces, predictions, strict=True):
        if trace['spec']['gold_points']:
            clip_pred, clip_scores = clip_dir / 'prediction.jsonl', clip_dir / 'scores.json'
            clip_pred.write_text(json.dumps(prediction) + '\n')
            write(clip_scores, [by_id[prediction['question_id']]])
            summary = score_pauc_f(clip_pred, clip_scores, omega=.5, judge_model=MODEL)
            trace['reward'] = summary['pauc_f'] / 100
        else:
            fires = len(prediction['model_response_list'])
            summary = {'num_fires': fires, 'num_value_adding_fires': 0,
                       'num_wasted_fires': fires, 'num_gold_turns': 0}
            trace['reward'] = -.25 * fires
        if not math.isfinite(trace['reward']):
            raise ValueError('nonfinite reward')
        trace['reward_metrics'] = summary
        trace['reward_version'] = VERSION
    write(directory / 'contract.json', contract())
    return report
