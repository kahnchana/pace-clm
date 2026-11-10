#!/usr/bin/env python3
"""User-requested full WEB evaluation of stopped E2-R checkpoints; no training."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import threading
import time

from jevany_rl_reward import ROOT, SG, MODEL, OFFICIAL, OFFICIAL_SCORER, judge_requests, request_key, sha, write
from report_jevany_e2 import paired_bootstrap
from run_jevany_rl_full_current_node import checkpoint_view
from run_qwen35_control import verify_web
from zs_infer.pauc_f import score_pauc_f

PY = SG/'.venv/bin/python'
NODE = 'ip-10-3-120-167'
RUN = ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1'
BASELINE = ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web'
PROTOCOL_KEYS = ('source_sha256', 'answer_model', 'answer_prompt',
                 'original_generator_adapter_sha256', 'response_margin', 'max_new_tokens',
                 'window_frames', 'memory_mode', 'memory_instruction', 'memory_select',
                 'prefix', 'history', 'controller_history', 'controller_model')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, values):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(value, sort_keys=True)+'\n' for value in values))
    temporary.replace(path)


def validate_judgments(requests, responses):
    wanted = [r['custom_id'] for r in requests]
    actual = [r['custom_id'] for r in responses]
    if len(set(wanted)) != len(wanted) or len(set(actual)) != len(actual) or set(actual) != set(wanted):
        raise ValueError('incomplete or duplicate judge requests/responses')
    for response in responses:
        content = response['response']['body']['choices'][0]['message']['content']
        if str(content).strip() not in ('1', '2', '3'):
            raise ValueError('invalid GPT judgment')


def seed_predictions(source, output, ids, checkpoint, world=8):
    """Retain complete dev video histories and assign them to the full-WEB ranks."""
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(source/'manifest.json')
    if (protocol['response_margin'] != 0 or protocol['history'] != 'self-generated' or
            protocol['answer_model'] != 'promoted' or protocol['max_new_tokens'] != 96):
        raise ValueError('incompatible development generation protocol')
    for field, name in (('controller_adapter_sha256', 'controller_adapter/adapter_model.safetensors'),
                        ('pointer_head_sha256', 'pointer_head.pt')):
        if protocol[field] != sha(checkpoint/name):
            raise ValueError('development predictions are from a different checkpoint')
    expected = {**protocol, 'rows': len(ids), 'question_ids': ids}
    predictions = rows(source/'predictions.jsonl')
    source_ids = [p['question_id'] for p in predictions]
    if (len(source_ids) != len(set(source_ids)) or set(source_ids) != set(protocol['question_ids']) or
            not set(source_ids) <= set(ids)):
        raise ValueError('incomplete or incompatible development prediction IDs')
    manifest = output/'manifest.json'
    if manifest.exists() and read(manifest) != expected:
        raise ValueError('existing full-WEB generation protocol differs')
    # Validate all shards before mutating any; a restart may already contain
    # newly generated non-dev videos, which are retained unchanged.
    shards = []
    by_id = {p['question_id']: p for p in predictions}
    for rank in range(world):
        allowed = ids[rank::world]
        path = output/f'predictions.rank{rank:02d}.jsonl'
        existing = rows(path) if path.exists() else []
        existing_ids = [p['question_id'] for p in existing]
        if len(existing_ids) != len(set(existing_ids)) or not set(existing_ids) <= set(allowed):
            raise ValueError('existing prediction shard has duplicate or wrong-rank IDs')
        merged = {p['question_id']: p for p in existing}
        for qid in allowed:
            if qid in by_id:
                if qid in merged and merged[qid] != by_id[qid]:
                    raise ValueError('saved development prediction changed')
                merged[qid] = by_id[qid]
        shards.append((path, [merged[qid] for qid in allowed if qid in merged]))
    write(manifest, expected)
    for path, values in shards:
        write_rows(path, values)
    provenance = {'source': str(source), 'source_predictions_sha256': sha(source/'predictions.jsonl'),
                  'source_protocol_sha256': sha(source/'manifest.json'),
                  'reused_complete_video_predictions': len(predictions),
                  'full_video_count': len(ids), 'generated_videos_needed_initially': len(ids)-len(predictions),
                  'reuse': 'same checkpoint and generation protocol; exact saved self-generated histories'}
    write(output/'reuse.json', provenance)
    return provenance


def seed_judge_cache(source, cache):
    requests = rows(source/'pauc_batch_input-gpt-4.1.jsonl')
    responses = rows(source/'pauc_batch_output-gpt-4.1.jsonl')
    validate_judgments(requests, responses)
    by_id = {r['custom_id']: r for r in responses}
    db = sqlite3.connect(cache)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, request TEXT, response TEXT)')
        for request in requests:
            raw = copy.deepcopy(by_id[request['custom_id']]['response']['body'])
            raw['choices'][0]['message']['content'] = raw['choices'][0]['message']['content'].strip()
            db.execute('INSERT OR IGNORE INTO judgments VALUES (?, ?, ?)',
                       (request_key(request['body']), json.dumps(request['body']), json.dumps(raw)))
        db.commit()
    finally:
        db.close()
    return {'source': str(source), 'requests': len(requests),
            'requests_sha256': sha(source/'pauc_batch_input-gpt-4.1.jsonl'),
            'responses_sha256': sha(source/'pauc_batch_output-gpt-4.1.jsonl')}


def score(output, gold, cache):
    request_path = output/'pauc_batch_input-gpt-4.1.jsonl'
    response_path = output/'pauc_batch_output-gpt-4.1.jsonl'
    OFFICIAL.create_openai_batch_input(str(output/'predictions.jsonl'), str(gold), str(request_path), -1, MODEL)
    requests = rows(request_path)
    responses, usage = judge_requests(requests, cache)
    validate_judgments(requests, responses)
    for response in responses:
        message = response['response']['body']['choices'][0]['message']
        message['content'] = message['content'].strip()
    write_rows(response_path, responses)
    scores = output/'pauc_llm_scores-gpt-4.1.json'
    OFFICIAL.process_openai_batch_output(str(output/'predictions.jsonl'), str(gold), str(response_path), str(scores))
    result = score_pauc_f(output/'predictions.jsonl', scores, omega=.5, judge_model=MODEL)
    write(output/'pauc_f_summary-gpt-4.1-v1.json', result)
    write(output/'judge-usage.json', usage)
    verify_web(output)
    return result


def report(directory, steps):
    baseline, _, _ = verify_web(BASELINE)
    comparisons = []
    lines = ['# E2-R full 500-video WEB evaluations', '',
             'User-requested evaluation of checkpoints 8 and 16 after the development coverage stop. Training remains stopped at round 16.',
             'All 500 videos include the previously inspected development subset; this is not an independent test.',
             'Each checkpoint retains its 128 complete development predictions and cached GPT judgments, and generates the remaining 372 videos.',
             'The SFT control uses retained complete predictions and GPT judgments under matched generation/scoring settings.', '',
             '| Checkpoint | PAUC-F | Delta vs SFT | Precision | Recall | Useful / wasted |',
             '|---|---:|---:|---:|---:|---:|',
             f"| SFT (retained control) | {baseline['pauc_f']:.4f} | — | {baseline['fire_precision']:.2f} | {baseline['pauc_recall_component']:.2f} | {baseline['num_value_adding_fires']} / {baseline['num_wasted_fires']} |"]
    for step in sorted(steps):
        output = directory/f'web-step{step}'
        if not (output/'pauc_f_summary-gpt-4.1-v1.json').exists():
            continue
        current, _, _ = verify_web(output)
        manifest, control = read(output/'manifest.json'), read(BASELINE/'manifest.json')
        for key in PROTOCOL_KEYS:
            if manifest[key] != control[key]:
                raise ValueError(f'unmatched full-WEB SFT protocol: {key}')
        bootstrap_path = output/'paired-bootstrap.json'
        bootstrap = read(bootstrap_path) if bootstrap_path.exists() else paired_bootstrap(output, BASELINE)
        write(bootstrap_path, bootstrap)
        coverage = current['num_value_adding_fires']/baseline['num_value_adding_fires']
        comparisons.append({'round': step, 'score': current, 'paired_bootstrap': bootstrap,
                            'useful_coverage_ratio': coverage,
                            'meets_coverage_floor': coverage >= .9,
                            'exceeds_sft_pauc_f': current['pauc_f'] > baseline['pauc_f']})
        lines.append(f"| E2-R round {step} | {current['pauc_f']:.4f} | {current['pauc_f']-baseline['pauc_f']:+.4f} | {current['fire_precision']:.2f} | {current['pauc_recall_component']:.2f} | {current['num_value_adding_fires']} / {current['num_wasted_fires']} |")
    lines += ['', 'Paired video-bootstrap intervals use 5,000 resamples, seed 42, and condition on saved generated answers and judgments. They do not measure judge or retraining uncertainty.', '']
    for comparison in comparisons:
        b = comparison['paired_bootstrap']
        lines.append(f"Round {comparison['round']}: Δ PAUC-F {b['difference_points']:+.4f}; 95% interval {b['percentile_95_ci']}; useful coverage {100*comparison['useful_coverage_ratio']:.2f}% of SFT.")
    write(directory/'comparison.json', {'baseline': baseline, 'baseline_source': str(BASELINE),
                                        'checkpoints': comparisons, 'expected_rounds': steps,
                                        'complete': len(comparisons) == len(steps)})
    (directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    directory = run/'full_web_v1'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    steps = [16, 8]  # Primary dev winner first; evaluate both regardless of coverage.
    gold = SG/'datasets/proactive-videoqa-web/anno.json'
    ids = [r['question_id'] for r in read(gold)]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('requires exactly 500 unique WEB videos')
    baseline, _, _ = verify_web(BASELINE)
    sources = [Path(__file__), ROOT/'scripts/eval_jevany_streaming.py',
               ROOT/'scripts/merge_jevany_benchmark.py', ROOT/'scripts/report_jevany_e2.py',
               ROOT/'scripts/jevany_rl_reward.py', ROOT/'scripts/judge_exp10_utility.py',
               ROOT/'scripts/run_qwen35_control.py', ROOT/'scripts/run_jevany_rl_full_current_node.py',
               ROOT/'scripts/generate_selected_responses.py',
               ROOT/'src/pace_clm/qwen_controller_runtime.py', ROOT/'src/pace_clm/evaluation_policy.py',
               ROOT/'src/pace_clm/jevany_streaming.py', SG/'zs_infer/scoring.py',
               SG/'zs_infer/pauc_f.py', OFFICIAL_SCORER,
               SG/'zs_infer/adapters/web.py', SG/'src/stream_gemma/inference/qwen_stream.py',
               SG/'src/stream_gemma/v24_memory.py', SG/'src/stream_gemma/config.py']
    hashes = {str(p): sha(p) for p in sources}
    original = read(run/'workflow-freeze.json')['source_hashes']
    for p in (SG/'zs_infer/scoring.py', SG/'zs_infer/pauc_f.py'):
        if original[str(p)] != sha(p):
            raise ValueError('official scoring source changed since development judgments')
    if read(run/'freeze.json')['reward_contract']['official_sha256'] != sha(OFFICIAL_SCORER):
        raise ValueError('official rubric changed since development judgments')
    tensor_hashes, reuse, seeds = {}, {}, []
    for step in steps:
        checkpoint = run/f'step-{step:06d}'
        for name in ('controller_adapter/adapter_model.safetensors', 'pointer_head.pt', 'model_identity.json'):
            tensor_hashes[str(checkpoint/name)] = sha(checkpoint/name)
        reuse[str(step)] = seed_predictions(run/'evaluation'/f'dev-step{step}', directory/f'web-step{step}', ids, checkpoint)
        seeds.append(seed_judge_cache(run/'evaluation'/f'dev-step{step}', directory/'gpt-cache.sqlite'))
    frozen = {'node': NODE, 'gpus': list(range(8)), 'rounds': steps,
              'videos_per_checkpoint': 500, 'judge': MODEL, 'scorer': 'pauc-f-v1', 'omega': .5,
              'gold_sha256': sha(gold), 'source_hashes': hashes, 'checkpoint_hashes': tensor_hashes,
              'baseline_manifest_sha256': sha(BASELINE/'manifest.json'),
              'baseline_summary_sha256': sha(BASELINE/'pauc_f_summary-gpt-4.1-v1.json'),
              'authorization': 'User explicitly requested full 500-video WEB evaluations on 2026-11-09',
              'training': 'none; prior coverage-gated stop remains unchanged', 'reuse': reuse}
    path = directory/'freeze.json'
    if path.exists() and read(path) != frozen:
        raise ValueError('full-WEB evaluation source/configuration differs from frozen run')
    write(path, frozen)
    write(directory/'judge-cache-seeds.json', seeds)
    snapshot = directory/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name, digest in hashes.items():
        path = snapshot/(digest+Path(name).suffix)
        if not path.exists():
            path.write_bytes(Path(name).read_bytes())
    report(directory, steps)
    if args.prepare_only:
        print(json.dumps({'prepared': str(directory), 'reused': reuse}), flush=True)
        return
    if socket.gethostname() != NODE or len(os.sched_getaffinity(0)) < 8:
        raise RuntimeError('requires original eight-GPU node and --cpu-bind=none')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('requires eight unused node-local GPUs')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).splitlines()
    if len(devices) != 8 or any('H200' not in x for x in devices):
        raise RuntimeError('requires eight H200 GPUs')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               TORCHINDUCTOR_COMPILE_THREADS='4', PACE_DISTRIBUTED_BACKEND='gloo',
               TRITON_CACHE_DIR='/tmp/pace-rl-triton',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    logs = directory/'logs'
    logs.mkdir(exist_ok=True)
    stopped = threading.Event()

    def monitor():
        with (directory/'gpu-samples.csv').open('a') as handle:
            while not stopped.is_set():
                subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits'], stdout=handle, stderr=handle)
                handle.flush()
                stopped.wait(10)

    def status(state, **fields):
        value = {'state': state, 'utc': datetime.now(timezone.utc).isoformat(),
                 'node': NODE, 'pid': os.getpid(), **fields}
        write(directory/'status.json', value)
        with (directory/'events.jsonl').open('a') as handle:
            handle.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)

    def command(argv, name):
        for file, digest in {**hashes, **tensor_hashes}.items():
            if sha(file) != digest:
                raise ValueError(f'frozen evaluation source/checkpoint changed: {file}')
        with (logs/name).open('a') as handle:
            subprocess.run(list(map(str,argv)), env=env, cwd=ROOT, check=True,
                           stdout=handle, stderr=subprocess.STDOUT)

    threading.Thread(target=monitor, daemon=True).start()
    try:
        for step in steps:
            output = directory/f'web-step{step}'
            if (output/'pauc_f_summary-gpt-4.1-v1.json').exists():
                verify_web(output)
                continue
            started = time.time()
            checkpoint = checkpoint_view(run, run/f'step-{step:06d}')
            status('rollout', round=step, videos=500, reused_videos=128)
            command([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                     ROOT/'scripts/eval_jevany_streaming.py','--controller',checkpoint,
                     '--benchmark','web','--output-dir',output,'--response-margin','0',
                     '--answer-model','promoted','--max-new-tokens','96','--record-action-logits'], f'web-step{step}-rollout.log')
            command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',output], f'web-step{step}-merge.log')
            status('gpt_scoring', round=step, videos=500)
            current = score(output, gold, directory/'gpt-cache.sqlite')
            write(output/'timing.json', {'seconds': time.time()-started,
                                        'allocated_gpu_hours': 8*(time.time()-started)/3600})
            report(directory, steps)
            status('checkpoint_complete', round=step, pauc_f=current['pauc_f'],
                   delta_vs_sft=current['pauc_f']-baseline['pauc_f'])
        report(directory, steps)
        status('complete', evaluated_rounds=steps, videos_per_checkpoint=500)
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stopped.set()


if __name__ == '__main__':
    main()
