#!/usr/bin/env python3
"""Bounded, resumable E2-R workflow on the user's current eight-H200 node."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

from jevany_rl_reward import ROOT, SG, contract, sha, write
from run_jevany_rl_full_current_node import checkpoint_view

PY = SG / '.venv/bin/python'
SFT = ROOT / 'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
NODE = 'ip-10-3-120-167'
DEFAULT = ROOT / 'outputs/jevany-2b-rl/reward-aligned-rloo-v1'


def read(path):
    return json.loads(Path(path).read_text())


def prepare_development(run):
    from prepare_qwen35_control import aliases
    directory = run / 'development'
    directory.mkdir(exist_ok=True)
    target = directory / 'selection.json'
    if target.exists():
        manifest = read(directory / 'freeze.json')
        if sha(target) != manifest['selection_sha256']:
            raise ValueError('development selection changed')
        return target
    training = ROOT / 'outputs/jevany-j1s/manifests/train.jsonl'
    cohort = set()
    with training.open() as handle:
        for line in handle:
            row = json.loads(line)
            cohort.update(aliases(row['video_path']))
    spans = ROOT / 'outputs/jevany-2b-rl/data/spans.jsonl'
    for line in spans.read_text().splitlines():
        cohort.update(aliases(json.loads(line)['video_path']))
    source = SG / 'datasets/proactive-videoqa-web/anno.json'
    eligible = [r for r in read(source) if not aliases(r['video']) & cohort]
    random.Random(42).shuffle(eligible)
    chosen, identities = [], set()
    for row in eligible:
        values = aliases(row['video'])
        if values & identities:
            continue
        chosen.append(row['question_id'])
        identities.update(values)
        if len(chosen) == 128:
            break
    if len(chosen) != 128:
        raise ValueError('insufficient video-disjoint development examples')
    write(target, chosen)
    write(directory / 'freeze.json', {
        'label': 'reused WEB development; previously inspected, not untouched test',
        'seed': 42, 'videos': len(chosen), 'source': str(source),
        'source_sha256': sha(source), 'selection_sha256': sha(target),
        'training_manifest_sha256': sha(training), 'spans_sha256': sha(spans),
        'training_aliases': len(cohort), 'training_overlap': 0,
        'policy': 'uniform shuffled eligible videos; deduplicate source-video aliases',
        'quality_checks': [8, 16, 32], 'near_silence_fire_ratio': .1,
        'coverage_floor_ratio': .9, 'coverage_patience': 2,
        'judge': 'gpt-4.1', 'scorer': 'pauc-f-v1', 'omega': .5,
        'response_margin': 0, 'history': 'actual self-generated MoE answers'})
    return target


def coverage_decision(baseline, checks):
    """Recompute chronological gates on resume; never skip a failed check."""
    low = 0
    for step, score in checks:
        ratio = score['num_value_adding_fires'] / max(1, baseline['num_value_adding_fires'])
        if score['num_fires'] < .1 * baseline['num_fires']:
            return 'near_silence', step
        low = low + 1 if ratio < .9 else 0
        if low >= 2:
            return 'coverage_below_90_percent_twice', step
    return None


def training_args(run, rounds):
    args = [str(PY), '-u', str(ROOT/'scripts/run_jevany_rl.py'), '--output', str(run),
            '--learners', '7', '--starts', '32', '--rollouts', '4',
            '--physical-batch', '24', '--microbatch', '24', '--collection-batch', '24',
            '--frame-cache', '2048', '--updates', str(rounds), '--full-run',
            '--reward-version', 'pauc-f-aligned-gpt-v1']
    if (run/'latest.json').exists():
        args.append('--resume')
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run/'full_run').mkdir(exist_ok=True)
    selection = prepare_development(run)
    if args.prepare_only:
        print(selection)
        return
    if socket.gethostname() != NODE:
        raise RuntimeError(f'user requested only current node {NODE}')
    if len(os.sched_getaffinity(0)) < 8:
        raise RuntimeError('launch with --cpu-bind=none')
    lock = (run/'workflow.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    if len(gpu_names) != 8 or any('H200' not in name for name in gpu_names):
        raise RuntimeError('requires eight local H200 GPUs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('node GPUs already in use')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               TORCHINDUCTOR_COMPILE_THREADS='4', PACE_DISTRIBUTED_BACKEND='gloo',
               TRITON_CACHE_DIR='/tmp/pace-rl-triton',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONPATH=':'.join(map(str, [ROOT/'outputs/jevany/deps032', ROOT/'src', ROOT/'scripts', SG/'src', SG])))
    sources = [Path(__file__), ROOT/'scripts/run_jevany_e2_rloo.sh',
               ROOT/'scripts/jevany_rl_reward.py', ROOT/'scripts/audit_jevany_rl_reward.py',
               ROOT/'scripts/run_jevany_rl.py', ROOT/'scripts/jevany_rl_workers.py',
               ROOT/'scripts/eval_jevany_streaming.py', ROOT/'scripts/score_exp10_web_ego.py',
               ROOT/'scripts/merge_jevany_benchmark.py', ROOT/'scripts/verify_jevany_rl_resume.py',
               ROOT/'scripts/report_jevany_e2.py', ROOT/'scripts/run_jevany_rl_full_current_node.py',
               ROOT/'scripts/judge_exp10_utility.py', SG/'zs_infer/scoring.py', SG/'zs_infer/pauc_f.py']
    hashes = {str(path): sha(path) for path in sources}
    plan_path = run/'workflow-freeze.json'
    if plan_path.exists() and read(plan_path)['source_hashes'] != hashes:
        raise ValueError('workflow source changed; inspect before resuming')
    if not plan_path.exists():
        write(plan_path, {'node': NODE, 'allocation': os.environ.get('SLURM_JOB_ID'),
                         'gpus': list(range(8)), 'source_hashes': hashes,
                         'reward': contract(), 'initial_checkpoint': str(SFT),
                         'collection_rounds': 32, 'optimizer_steps': 32,
                         'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                         'started_utc': datetime.now(timezone.utc).isoformat()})
        snapshot = run/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name, digest in hashes.items():
            (snapshot/(digest+Path(name).suffix)).write_bytes(Path(name).read_bytes())
        (snapshot/'working-tree.diff').write_bytes(subprocess.check_output(['git','diff']))
        (snapshot/'working-tree-status.txt').write_bytes(subprocess.check_output(['git','status','--short']))
    logs = run/'logs'
    logs.mkdir(exist_ok=True)

    def status(state, **fields):
        record = {'state': state, 'utc': datetime.now(timezone.utc).isoformat(),
                  'node': NODE, 'pid': os.getpid(), **fields}
        write(run/'workflow-status.json', record)
        print(json.dumps(record), flush=True)

    def command(argv, name):
        for path, digest in hashes.items():
            if sha(path) != digest:
                raise ValueError(f'frozen workflow artifact changed: {path}')
        print(json.dumps({'command': list(map(str, argv)), 'log': str(logs/name)}), flush=True)
        with (logs/name).open('a') as handle:
            subprocess.run(list(map(str, argv)), env=env, cwd=ROOT,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def evaluate(checkpoint, name, *, full=False):
        output = run/'evaluation'/name
        output.mkdir(parents=True, exist_ok=True)
        count = 500 if full else 128
        summary = output/'pauc_f_summary-gpt-4.1-v1.json'
        if not summary.exists():
            view = checkpoint if checkpoint == SFT else checkpoint_view(run, checkpoint)
            rollout = [PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                       ROOT/'scripts/eval_jevany_streaming.py', '--controller', view,
                       '--benchmark', 'web', '--output-dir', output, '--response-margin', '0',
                       '--answer-model', 'promoted', '--max-new-tokens', '96', '--record-action-logits']
            if not full:
                rollout += ['--selection-ids', selection]
            command(rollout, f'{name}-rollout.log')
            command([PY, ROOT/'scripts/merge_jevany_benchmark.py', '--input-dir', output], f'{name}-merge.log')
            command([PY, ROOT/'scripts/score_exp10_web_ego.py', 'web', '--pred-dir', output,
                     '--expected-videos', str(count)], f'{name}-gpt.log')
        score = read(summary)
        manifest = read(output/'manifest.json')
        wanted = set(read(selection)) if not full else {r['question_id'] for r in read(SG/'datasets/proactive-videoqa-web/anno.json')}
        predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
        scored = read(output/'pauc_llm_scores-gpt-4.1.json')
        requests = [json.loads(line) for line in (output/'pauc_batch_input-gpt-4.1.jsonl').read_text().splitlines() if line.strip()]
        responses = [json.loads(line) for line in (output/'pauc_batch_output-gpt-4.1.jsonl').read_text().splitlines() if line.strip()]
        if (len(responses) != len(requests) or
                len({r['custom_id'] for r in responses}) != len(responses) or
                {r['custom_id'] for r in responses} != {r['custom_id'] for r in requests} or
                any(str(r['response']['body']['choices'][0]['message']['content']).strip() not in ('1','2','3')
                    for r in responses) or len(scored) != count or
                {r['question_id'] for r in scored} != wanted):
            raise ValueError('incomplete or invalid GPT evaluation judgments')
        if (score['num_videos'] != count or score['judge_model'] != 'gpt-4.1' or
                score['scorer_version'] != 'pauc-f-v1' or score['omega'] != .5 or
                set(manifest['question_ids']) != wanted or len(predictions) != count or
                {r['question_id'] for r in predictions} != wanted or
                manifest['response_margin'] != 0 or manifest['history'] != 'self-generated'):
            raise ValueError('evaluation protocol or completeness failed')
        from pace_clm.qwen_controller_runtime import MODEL_REVISION
        if manifest['controller_model'].get('model_revision') != MODEL_REVISION:
            raise ValueError('controller revision changed')
        return score

    try:
        # Baseline evaluation is independent of the CPU/GPT reward audit and
        # may run while that audit finishes. Training remains audit-gated.
        status('sft_development_evaluation')
        baseline = evaluate(SFT, 'dev-sft')
        write(run/'development/sft-baseline.json', baseline)
        status('reward_audit')
        audit = ROOT/'outputs/jevany-2b-rl/reward-audit-v1'
        if not (audit/'summary.json').exists():
            command([PY, ROOT/'scripts/audit_jevany_rl_reward.py'], 'reward-audit.log')
        if not read(audit/'summary.json')['gate_passed'] or read(audit/'reward-spec.json') != contract():
            raise ValueError('reward audit failed or contract changed')
        status('hardware_preflight')
        command([PY, ROOT/'scripts/jevany_rl_kernel_cache.py', 'stage', '--audit', run/'kernel-cache.jsonl'], 'kernel-cache.log')
        command([PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                 ROOT/'scripts/probe_qwen35_nccl.py'], 'hardware-preflight.log')
        if not (run/'latest.json').exists():
            status('one_round_correctness_gate')
            command(training_args(run, 1), 'train-gate.log')
        if not (run/'gate-passed.json').exists():
            status('resume_verification')
            command([PY, ROOT/'scripts/verify_jevany_rl_resume.py', '--run', run], 'resume-verification.log')
            restore = read(run/'resume_verification.json')
            report = read(run/'batch-000000/report.json')
            if not restore['verified'] or restore['additional_optimizer_steps'] != 0:
                raise ValueError('resume failed')
            if report['optimizer_steps'] != 1 or report['decisions'] != 4096:
                raise ValueError('incorrect gate exposure')
            for learner in report['learner'].values():
                if (learner['head_grad_norm'] <= 0 or learner['lora_grad_norm'] <= 0 or
                        abs(learner['kl_sum']) > 1e-4 or
                        learner['parity_probability_max'] > .005):
                    raise ValueError('gate gradients, initial SFT KL or replay parity failed')
            write(run/'gate-passed.json', {'verified': True, 'step': 1,
                                         'resume_verification': restore, 'report_sha256': sha(run/'batch-000000/report.json')})
        checks = []
        for step in (8, 16, 32):
            latest = read(run/'latest.json')['step']
            if latest < step:
                status('training', target_round=step, completed_round=latest)
                command(training_args(run, step), f'train-to-{step}.log')
            status('development_evaluation', round=step)
            score = evaluate(run/f'step-{step:06d}', f'dev-step{step}')
            checks.append((step, score))
            write(run/'development/checks.json', checks)
            pause = coverage_decision(baseline, checks)
            if pause:
                status('stopped_by_coverage_gate', reason=pause[0], round=pause[1])
                command([PY, ROOT/'scripts/report_jevany_e2.py', '--run', run], 'report.log')
                return
        eligible = [(step, score) for step, score in checks
                    if score['num_value_adding_fires'] >= .9 * baseline['num_value_adding_fires']
                    and score['pauc_f'] > baseline['pauc_f']]
        if eligible:
            best_step, best = max(eligible, key=lambda pair: pair[1]['pauc_f'])
            write(run/'selected.json', {'round': best_step, 'score': best,
                                       'selection': 'highest dev PAUC-F subject to >=90% SFT useful coverage'})
            status('selected_full_web_evaluation', round=best_step)
            evaluate(run/f'step-{best_step:06d}', f'web-selected-step{best_step}', full=True)
        status('reporting', completed_round=32, promising_development=bool(eligible))
        command([PY, ROOT/'scripts/report_jevany_e2.py', '--run', run], 'report.log')
        status('complete', completed_round=32, optimizer_steps=32,
               promising_development=bool(eligible), automatic_extension=False)
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with (logs/'kernel-cache-save.log').open('a') as handle:
            subprocess.run([str(PY), str(ROOT/'scripts/jevany_rl_kernel_cache.py'), 'save',
                            '--audit', str(run/'kernel-cache.jsonl')], env=env,
                           stdout=handle, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    main()
