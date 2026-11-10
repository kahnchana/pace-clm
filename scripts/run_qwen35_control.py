#!/usr/bin/env python3
"""Run a frozen step-budget control in four checkpoint/WEB/resume phases."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/jevany-qwen35-2b-mixture-12h'


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verify_web(web):
    manifest = json.loads((web / 'manifest.json').read_text())
    expected = manifest['question_ids']
    predictions = [json.loads(line) for line in (web / 'predictions.jsonl').read_text().splitlines()]
    scores = json.loads((web / 'pauc_llm_scores-gpt-4.1.json').read_text())
    if len(expected) != 500 or len(set(expected)) != 500:
        raise ValueError('WEB protocol does not contain 500 unique IDs')
    for rows in (predictions, scores):
        ids = [r['question_id'] for r in rows]
        if len(ids) != 500 or set(ids) != set(expected):
            raise ValueError('incomplete/duplicated WEB prediction or score IDs')
    requests = [json.loads(line) for line in (web / 'pauc_batch_input-gpt-4.1.jsonl').read_text().splitlines() if line]
    responses = [json.loads(line) for line in (web / 'pauc_batch_output-gpt-4.1.jsonl').read_text().splitlines() if line]
    response_ids = [r['custom_id'] for r in responses]
    if len(response_ids) != len(requests) or len(set(response_ids)) != len(response_ids) or set(response_ids) != {r['custom_id'] for r in requests}:
        raise ValueError('incomplete judge request coverage')
    for row in responses:
        content = row['response']['body']['choices'][0]['message']['content']
        if not content or content[0] not in '123':
            raise ValueError('invalid judge response')
    score = json.loads((web / 'pauc_f_summary-gpt-4.1-v1.json').read_text())
    if (score['num_videos'], score['judge_model'], score['scorer_version'], score['omega']) != (500, 'gpt-4.1', 'pauc-f-v1', 0.5):
        raise ValueError('WEB scorer protocol mismatch')
    normalized = {k:v for k,v in manifest.items() if k not in ('controller_adapter_sha256','pointer_head_sha256')}
    fingerprint = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    return score, predictions, fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=BASE / 'control-seed42')
    a = p.parse_args()
    from judge_exp10_utility import load_gateway_key
    load_gateway_key()
    out = a.output.resolve()
    lock = (out / 'orchestrator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((out / 'launch_plan.json').read_text())
    if not plan['preflight_passed'] or plan['total_steps'] % 4:
        raise ValueError('preflight incomplete or invalid step milestones')
    train = out / 'train'
    train.mkdir(exist_ok=True)
    status = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'plan': plan}
    stop_monitor = threading.Event()

    def monitor():
        with (out / 'logs/gpu_samples.csv').open('a') as log:
            while not stop_monitor.is_set():
                subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], stdout=log, stderr=log)
                log.flush()
                stop_monitor.wait(10)

    threading.Thread(target=monitor, daemon=True).start()

    def run(command, log):
        print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'command': list(map(str,command))}), flush=True)
        with (out / 'logs' / log).open('a') as handle:
            subprocess.run(list(map(str,command)), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)

    records = []
    try:
        protocol_hash = None
        for phase, target in enumerate(plan['milestones'], 1):
            status.update(phase=phase, target_step=target, state='training')
            write_json(out / 'status.json', status)
            latest_path = train / 'latest.json'
            latest = json.loads(latest_path.read_text()) if latest_path.exists() else None
            step = latest['step'] if latest else 0
            checkpoint = train / f'step-{target:06d}'
            if step < target:
                args = ['--resume'] if latest else ['--initialize-from', plan['initial_checkpoint']]
                run(['bash', ROOT / 'scripts/run_qwen35_control_command.sh', 'train',
                     '--manifest', plan['manifest'], '--output', train,
                     '--expected-manifest-sha256', plan['manifest_sha256'],
                     '--stop-after-steps', target, *args], f'train_phase{phase}.log')
            if not (checkpoint / 'training_state.pt').is_file():
                raise ValueError('missing scheduled immutable checkpoint')
            state = __import__('torch').load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
            if state['step'] != target or state['seen_per_rank'] * 8 != target * 128:
                raise ValueError('unexpected checkpoint progress')
            status.update(state='web', checkpoint=str(checkpoint))
            write_json(out / 'status.json', status)
            web = out / 'evaluation' / f'web_step{target:06d}'
            run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                 ROOT / 'scripts/eval_jevany_streaming.py', '--controller', checkpoint,
                 '--benchmark', 'web', '--output-dir', web, '--response-margin', '0',
                 '--max-new-tokens', '96', '--answer-model', 'promoted', '--record-action-logits'], f'web_phase{phase}.log')
            run([sys.executable, ROOT / 'scripts/merge_jevany_benchmark.py', '--input-dir', web, '--world-size', '8'], f'web_phase{phase}.log')
            run([sys.executable, ROOT / 'scripts/score_exp10_web_ego.py', 'web', '--pred-dir', web, '--expected-videos', '500'], f'judge_phase{phase}.log')
            score, predictions, fingerprint = verify_web(web)
            manifest = json.loads((web/'manifest.json').read_text())
            if manifest['source_sha256'] != plan['web_gold_sha256'] or manifest['original_generator_adapter_sha256'] != plan['generator_adapter_sha256']:
                raise ValueError('WEB data or generator differs from the frozen launch plan')
            if protocol_hash and fingerprint != protocol_hash:
                raise ValueError('WEB protocol changed across checkpoints')
            protocol_hash = fingerprint
            videos, positions, labels, recent = set(), set(), Counter(), Counter()
            with Path(plan['manifest']).open() as handle:
                for i,line in enumerate(islice(handle, target*128)):
                    row = json.loads(line)
                    videos.add(row['video_id'])
                    positions.add((row['video_id'],row['question'],row['frame_index']))
                    labels[row['label']] += 1
                    if i >= target*128-4096:
                        recent[row['label']] += 1
            if dict(labels) != state['global_labels']:
                raise ValueError('checkpoint exposure labels do not match frozen prefix')
            actions = sum((Counter(row['jevany_actions']) for row in predictions), Counter())
            repeated_text, repeated_times = 0, 0
            for prediction in predictions:
                fires = prediction.get('model_response_list',[])
                texts = [' '.join(str(f.get('content','')).lower().split()) for f in fires]
                times = [f['time'] for f in fires]
                repeated_text += len(texts)-len(set(texts))
                repeated_times += len(times)-len(set(times))
            record = {'phase': phase, 'step': target, 'presentations': target*128,
                      'training_hours': state['training_seconds']/3600,
                      'presentations_per_second': target*128/state['training_seconds'],
                      'unique_videos': len(videos), 'unique_positions':len(positions),
                      'labels':dict(labels), 'recent_4096_labels': dict(recent),
                      'web_actions':dict(actions), 'web_protocol_sha256': fingerprint, 'score':score}
            runtime_path = out/f'phase{phase}_training_runtime.json'
            if not runtime_path.exists():
                rank_summaries = [json.loads((train/f'summary.rank{rank}.json').read_text()) for rank in range(8)]
                if any(summary['step'] != target for summary in rank_summaries):
                    raise ValueError('training runtime summaries do not match this milestone')
                events = []
                for line in (out/f'logs/train_phase{phase}.log').read_text().splitlines():
                    if line.startswith('{'):
                        try:
                            event = json.loads(line)
                            if 'step' in event:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
                write_json(runtime_path, {'rank_summaries':rank_summaries,
                    'rank0_mean_token_padding_fraction':sum(e['rank0_token_padding_fraction'] for e in events)/max(1,len(events)),
                    'rank0_final_frame_cache_hit_fraction':events[-1]['rank0_frame_cache_hit_fraction'] if events else None})
            record['training_runtime'] = json.loads(runtime_path.read_text())
            record['web_action_rates'] = {k:v/sum(actions.values()) for k,v in actions.items()}
            record['repeated_exact_answer_text_fires'] = repeated_text
            record['repeated_timestamp_fires'] = repeated_times
            write_json(out / f'phase{phase}_report.json', record)
            records.append(record)
            write_json(out / 'report.json', {'plan':plan,'checkpoints':records})
        status.update(state='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed', error=repr(error), failed_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(out / 'status.json', status)
        stop_monitor.set()


if __name__ == '__main__':
    main()
