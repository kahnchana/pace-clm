#!/usr/bin/env python3
"""Sequential, resumable eight-GPU SFT experiments and authorized GPT evaluations."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

from qwen35_sft_campaign import HOST, ROOT, rows, sha, write

SG = ROOT.parent / 'stream-gemma'
TRAIN_PY = SG / '.venv/bin/python'
EVAL_PY = ROOT.parent / 'lmms-eval/.venv/bin/python'
BASELINE = ROOT / 'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
DEFAULT_OUT = ROOT / 'outputs/jevany-qwen35-2b-sft-campaign-20261109'
CODE_FILES = [
    'scripts/run_qwen35_sft_campaign.py', 'scripts/train_qwen35_sft_campaign.py',
    'scripts/qwen35_sft_campaign.py', 'scripts/prepare_qwen35_sft_campaign.py',
    'scripts/prepare_qwen35_sft_corrections.py', 'scripts/collect_qwen35_sft_contexts.py',
    'scripts/benchmark_qwen35_sft_cache.py', 'scripts/train_jevany_streaming.py',
    'scripts/benchmark_qwen35_2b_training.py', 'scripts/eval_qwen35_development_sweep.py',
    'scripts/eval_jevany_streaming.py', 'scripts/score_exp10_web_ego.py',
    'scripts/prepare_exp10_scoring.py', 'scripts/judge_exp10_rollout.py',
    'scripts/judge_exp10_utility.py', 'scripts/score_exp10.py',
    'scripts/qwen35_diagnostic_kernels.py', 'scripts/jevany_rl_rpc.py',
    'src/pace_clm/jevany_streaming.py', 'src/pace_clm/qwen_controller_runtime.py',
    'src/pace_clm/evaluation_policy.py', 'configs/actions_v1.yaml',
]


class Campaign:
    def __init__(self, out):
        self.out = out.resolve()
        for name in ('logs', 'data', 'reports', 'evaluation', 'preflight'):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.results_path = self.out / 'reports/results.json'
        self.results = json.loads(self.results_path.read_text()) if self.results_path.exists() else {}
        self.hashes = {}

    def status(self, state, **kwargs):
        value = dict(state=state, host=socket.gethostname(), pid=os.getpid(),
                     utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
        write(self.out / 'status.json', value)
        print(json.dumps(value), flush=True)

    def environment(self, evaluation=False):
        env = dict(os.environ)
        env.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PACE_DISTRIBUTED_BACKEND='gloo',
                   PYTHONPATH=':'.join(map(str, [ROOT / 'outputs/jevany/deps032', ROOT / 'src',
                                               ROOT / 'scripts', SG / 'src', SG])),
                   TRITON_CACHE_DIR='/tmp/pace-sft-campaign-eval-triton36' if evaluation else '/tmp/pace-sft-campaign-train-triton',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False' if evaluation else 'expandable_segments:True',
                   TORCHINDUCTOR_COMPILE_THREADS='1')
        return env

    def check_code(self):
        for filename, expected in self.hashes.items():
            if sha(ROOT / filename) != expected:
                raise ValueError(f'campaign source changed; refusing mixed-provenance execution: {filename}')

    def run(self, command, log, evaluation=False):
        self.check_code()
        log = self.out / 'logs' / log
        with log.open('a') as handle:
            handle.write(json.dumps(dict(command=list(map(str, command)), utc=time.time())) + '\n')
            handle.flush()
            subprocess.run(list(map(str, command)), cwd=ROOT, env=self.environment(evaluation),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def distributed(self, script, options, log, evaluation=False):
        deadline = time.monotonic() + 24 * 3600
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            self.status('waiting_for_free_gpus', next_script=script)
            if time.monotonic() > deadline:
                raise TimeoutError('node remained occupied; no other job was interrupted')
            time.sleep(15)
        python = EVAL_PY if evaluation else TRAIN_PY
        self.run([python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                  '--nproc_per_node=8', ROOT / 'scripts' / script, *options], log, evaluation)

    def freeze(self):
        if socket.gethostname().split('.')[0] != HOST:
            raise RuntimeError(f'GPU work is restricted to {HOST}')
        gpu_ids = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines()
        if gpu_ids != list(map(str, range(8))):
            raise RuntimeError('requires exactly this node\'s eight GPUs')
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError('GPUs are occupied; refusing to interrupt another job')
        from prepare_qwen35_sft_campaign import prepare_anchor
        audit = prepare_anchor(self.out)
        # Existing reserved cohorts have been inspected before; they are not fresh tests.
        for source, cohort in [('selection', 'development'), ('final', 'confirmation')]:
            path = ROOT / f'outputs/pace-clm-mixtures/validation/{source}.jsonl'
            gold = path.with_name(f'{source}.gold.json')
            target = self.out / f'data/{cohort}.jsonl'
            for original, destination in [(path, target), (gold, target.with_suffix('.gold.json'))]:
                if destination.exists() and sha(original) != sha(destination):
                    raise ValueError('reserved cohort changed')
                if not destination.exists():
                    shutil.copyfile(original, destination)
        self.hashes = {filename: sha(ROOT / filename) for filename in CODE_FILES}
        for filename in CODE_FILES:
            destination = self.out / 'code' / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha(destination) != self.hashes[filename]:
                raise ValueError(f'archived source differs: {filename}')
            if not destination.exists():
                shutil.copyfile(ROOT / filename, destination)
        n = audit['presentations']
        self.specs = {
            'anchor_b32': dict(data='anchor', batch=32, presentations=n),
            'anchor_b128': dict(data='anchor', batch=128, presentations=n),
            'head_control': dict(data='continuation', batch=32, presentations=65536, parent='anchor_b32'),
            'head_low': dict(data='continuation', batch=32, presentations=65536, parent='anchor_b32', head_lr=2.5e-5),
            'fresh_scale': dict(data='fresh', batch=32, presentations=n, parent='anchor_b32'),
            'replay_scale': dict(data='replay', batch=32, presentations=n, parent='anchor_b32'),
            'rollout_control': dict(data='rollout_control', batch=32, presentations=65536, parent='anchor_b32'),
            'rollout_corrected': dict(data='rollout_corrected', batch=32, presentations=65536, parent='anchor_b32'),
        }
        baseline_protocol = json.loads((ROOT / 'outputs/jevany-qwen35-2b-raw-full-web/baseline_margin0/manifest.json').read_text())
        plan = dict(host=HOST, gpus=8, seed=42, confirmation_seeds=[43, 44], arms=self.specs,
                    anchor=audit, code_hashes=self.hashes, training_python=str(TRAIN_PY), evaluation_python=str(EVAL_PY),
                    global_batch_definition='Four or sixteen examples per rank; physical batch and accumulation do not change global normalization.',
                    head_lr=1e-4, lora_lr=2e-6, constant_lr=True,
                    response_margin=0, decision_policy='raw_learned_argmax', quality_floor=54.67,
                    full_web_videos=500, web_gold_sha256=baseline_protocol['source_sha256'],
                    generator_adapter_sha256=baseline_protocol['original_generator_adapter_sha256'],
                    max_new_tokens=96, judge_model='gpt-4.1', scorer='pauc-f-v1', omega=.5,
                    selection='Frozen final endpoints only. Rank seed-42 arms on development PAUC-F@4s; ties use fixed arm order. Confirm the development-selected recipe across seeds 43/44. WEB never changes training settings or selection.',
                    cohorts={cohort: dict(manifest_sha256=sha(self.out / f'data/{cohort}.jsonl'),
                              gold_sha256=sha(self.out / f'data/{cohort}.gold.json'),
                              videos=sum(1 for _ in rows(self.out / f'data/{cohort}.jsonl')))
                             for cohort in ('development', 'confirmation')},
                    limitations=['The clean anchor excludes 5,215 previously held-out/other-benchmark rows; it is not an exact historical-anchor reproduction.',
                                 'Reserved cohorts have prior experimental exposure; confirmation is reserved within this campaign, not newly untouched data.',
                                 'WEB has prior experimental exposure. Report all frozen endpoints and training-seed variation; do not tune on WEB.'],
                    feature_cache='Separate exact-parity and preparation benchmark; never changes quality-arm inputs or effective batch.')
        write(self.out / 'launch_plan.json', plan, frozen=True)
        write(self.out / 'authorization.json', dict(
            date='2026-11-09', source='User explicitly requested all proposed SFT experiments on this node using eight GPUs and GPT scoring to compare full-WEB PAUC against the current best.',
            host=HOST, gpu_count=8, gateway='https://gateway.salesforceresearch.ai/openai/process/v1/',
            judge_payload='Questions, current gold annotations, causal generated answers for corrective supervision, and benchmark/development/confirmation scoring.',
            inference='Raw learned argmax only; no score offsets, margins, runtime suppression, or calibration.'), frozen=True)
        self.plan = plan

    def train(self, name, spec, seed=42, stop_at=None):
        output = self.out / f'training/seed{seed}/{name}'
        data = self.out / f'data/{spec["data"]}.jsonl'
        if sum(1 for _ in rows(data)) != spec['presentations']:
            raise ValueError(f'{name} data count differs from frozen budget')
        target = stop_at or math.ceil(spec['presentations'] / spec['batch'])
        expected_presentations = min(spec['presentations'], target * spec['batch'])
        checkpoint = output / f'step-{target:06d}'
        if (checkpoint / 'training_state.pt').exists() and (output / 'summary.rank0.json').exists():
            self.verify_checkpoint(checkpoint, target, expected_presentations, seed)
            return checkpoint
        self.status('training', arm=name, seed=seed, target_steps=target,
                    presentations=spec['presentations'], global_batch=spec['batch'])
        options = ['--manifest', data, '--expected-manifest-sha256', sha(data), '--output', output,
                   '--backend', 'gloo', '--microbatch', str(spec['batch'] // 8),
                   '--frame-cache', '1024', '--conv-batch-chunk', '128', '--predecode-workers', '4',
                   '--prefetch-mode', 'prepare', '--fixed-l2norm-launch', '--save-every', '256',
                   '--log-every', '25', '--stop-after-steps', str(target), '--seed', str(seed),
                   '--head-lr', str(spec.get('head_lr', 1e-4)), '--lora-lr', '2e-6']
        if (output / 'latest.json').exists():
            options += ['--resume']
        elif spec.get('parent'):
            parent = self.train(spec['parent'], self.specs[spec['parent']], seed)
            options += ['--continue-from', parent]
        else:
            options += ['--initialize-from', self.initialize(seed)]
        self.distributed('train_qwen35_sft_campaign.py', options, f'train_{seed}_{name}.log')
        self.verify_checkpoint(checkpoint, target, expected_presentations, seed)
        return checkpoint

    @staticmethod
    def verify_checkpoint(checkpoint, target, presentations, seed):
        import torch
        value = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        if value['step'] != target or value['seen_per_rank'] * 8 != presentations or value['seed'] != seed:
            raise ValueError('checkpoint counter or seed mismatch')
        for rank in range(8):
            if not (checkpoint / f'rng.rank{rank}.pt').exists():
                raise ValueError('checkpoint lacks a rank RNG state')

    def initialize(self, seed):
        output = self.out / ('shared-initial' if seed == 42 else f'shared-initial-seed{seed}')
        checkpoint = output / 'step-000000'
        if not (checkpoint / 'training_state.pt').exists():
            self.status('initializing', seed=seed)
            self.distributed('train_qwen35_sft_campaign.py', [
                '--manifest', self.out / 'data/anchor.jsonl', '--output', output,
                '--backend', 'gloo', '--microbatch', '4', '--seed', str(seed),
                '--fixed-l2norm-launch', '--initialize-only'], f'initialize_{seed}.log')
        self.verify_checkpoint(checkpoint, 0, 0, seed)
        return checkpoint

    def preflight(self):
        initial = self.initialize(42)
        self.status('preflight')
        for batch in (32, 128):
            spec = dict(data='probe', batch=batch, presentations=1024)
            # Consume all eight context tiers, with a real partial-update path tested separately.
            if batch == 32 and not (self.out / 'training/seed42/preflight_b32/latest.json').exists():
                self.train('preflight_b32', spec, stop_at=16)
            self.train('preflight_b' + str(batch), spec)
        # Verify exact restoration on every rank before using the trainer for quality runs.
        self.distributed('train_qwen35_sft_campaign.py', [
            '--manifest', self.out / 'data/probe.jsonl', '--output', self.out / 'preflight/exact_restore',
            '--initialize-from', self.out / 'training/seed42/preflight_b32/step-000032',
            '--backend', 'gloo', '--microbatch', '4',
            '--fixed-l2norm-launch', '--verify-restore-only'], 'preflight_exact_restore.log')
        for rank in range(8):
            if not json.loads((self.out / f'preflight/exact_restore/verified.rank{rank}.json').read_text())['exact_restore']:
                raise ValueError('rank restoration failed')
        # Smallest shared final tail is 112 at batch 128 for the clean anchor.
        probe = list((self.out / 'data/probe.jsonl').open('rb'))[:144]
        partial = self.out / 'data/partial.jsonl'
        if not partial.exists():
            partial.write_bytes(b''.join(probe))
        self.train('preflight_partial', dict(data='partial', batch=128, presentations=144))
        for name in ('preflight_b32', 'preflight_b128', 'preflight_partial'):
            for rank in range(8):
                value = json.loads((self.out / f'training/seed42/{name}/summary.rank{rank}.json').read_text())
                if value['peak_reserved_gib'] / value['device_memory_gib'] > .9:
                    raise ValueError('preflight lacks required 10% memory headroom')
        try:
            self.distributed('benchmark_qwen35_sft_cache.py', [
                '--manifest', self.out / 'data/probe.jsonl', '--checkpoint', initial,
                '--output', self.out / 'preflight/feature_cache'], 'feature_cache_benchmark.log')
            values = [json.loads((self.out / f'preflight/feature_cache/rank{r:02d}.json').read_text()) for r in range(8)]
            write(self.out / 'reports/feature_cache.json', dict(ranks=values,
                parity_passed=all(v['passes'][mode]['features_bit_identical'] and v['passes'][mode]['logits_bit_identical']
                                  for v in values for mode in ('cold_cache', 'warm_cache')),
                production_enabled=False, limitation='Small preparation/forward benchmark; cold and warm runs have cache/page/kernel warmup differences. No full-training speedup claim.'))
        except Exception as error:
            write(self.out / 'reports/feature_cache.json', dict(error=str(error),
                  parity_passed=False, production_enabled=False, benchmark_failed=True))
        write(self.out / 'preflight/passed.json', dict(exact_restore_all_ranks=True,
              finite_training_batches=[32, 128], partial_update_presentations=144, memory_headroom_min=.1))

    def evaluate(self, name, checkpoint, cohort):
        key = name + '/' + cohort
        output = self.out / f'evaluation/{cohort}/{name}'
        condition = dict(name=name, controller=str(checkpoint), margin=0., output=str(output))
        if cohort == 'web':
            condition['benchmark'] = 'web'
        else:
            condition['manifest'] = str(self.out / f'data/{cohort}.jsonl')
        config = self.out / f'evaluation/configs/{cohort}_{name}.json'
        write(config, [condition], frozen=True)
        if not (output / 'rollout_complete.json').exists():
            self.status('rollouts', arm=name, cohort=cohort)
            self.distributed('eval_qwen35_development_sweep.py', [
                '--conditions', config, '--require-native-moe', '--generator-kernels', 'torch'],
                f'rollout_{name}_{cohort}.log', evaluation=True)
        self.status('gpt_scoring', arm=name, cohort=cohort)
        if cohort == 'web':
            self.run([TRAIN_PY, ROOT / 'scripts/merge_jevany_benchmark.py', '--input-dir', output,
                      '--world-size', '8'], f'score_{name}_{cohort}.log')
            self.run([TRAIN_PY, ROOT / 'scripts/score_exp10_web_ego.py', 'web', '--pred-dir', output,
                      '--expected-videos', '500'], f'score_{name}_{cohort}.log')
            from run_qwen35_control import verify_web
            score, predictions, _ = verify_web(output)
            manifest = json.loads((output / 'manifest.json').read_text())
            if (manifest['source_sha256'] != self.plan['web_gold_sha256'] or
                manifest['original_generator_adapter_sha256'] != self.plan['generator_adapter_sha256']):
                raise ValueError('WEB data or generator differs from frozen contract')
            if manifest['controller_adapter_sha256'] != sha(checkpoint / 'controller_adapter/adapter_model.safetensors'):
                raise ValueError('evaluated adapter differs from trained checkpoint')
            if manifest['pointer_head_sha256'] != sha(checkpoint / 'pointer_head.pt'):
                raise ValueError('evaluated pointer differs from trained checkpoint')
            self.verify_raw(predictions, 8803)
        else:
            summary = output / 'official/results/exp10_judged_summary.json'
            if not summary.exists():
                self.run([TRAIN_PY, ROOT / 'scripts/prepare_exp10_scoring.py', '--rollout-dir', output,
                    '--manifest', condition['manifest'], '--gold', self.out / f'data/{cohort}.gold.json',
                    '--world-size', '8'], f'score_{name}_{cohort}.log')
                requests = output / 'official/judge/pauc_batch_input.jsonl'
                if requests.read_text().strip():
                    self.run([TRAIN_PY, ROOT / 'scripts/judge_exp10_rollout.py', '--rollout-dir', output,
                              '--concurrency', '16'], f'score_{name}_{cohort}.log')
                else:
                    write(output / 'official/judge/pauc_llm_scores.json', dict(
                        judge_model='gpt-4.1', scores_by_request_key={}, request_mappings=[]), frozen=True)
                    self.run([TRAIN_PY, ROOT / 'scripts/score_exp10.py', '--rollout-dir', output], f'score_{name}_{cohort}.log')
            score = json.loads(summary.read_text())
            expected = self.plan['cohorts'][cohort]['videos']
            if score['num_sampled_videos'] != expected or score['judge_model'] != 'gpt-4.1' or not score['headline']:
                raise ValueError('development/confirmation scoring coverage mismatch')
            predictions = list(rows(output / 'official/generation/predictions.jsonl'))
            if len(predictions) != expected or {r['question_id'] for r in predictions} != {r['question_id'] for r in rows(condition['manifest'])}:
                raise ValueError('development/confirmation prediction coverage mismatch')
            count = sum(r['inference_frame_count'] for r in rows(condition['manifest']))
            self.verify_raw(predictions, count)
        result = dict(name=name, cohort=cohort, checkpoint=str(checkpoint), output=str(output), score=score)
        if cohort == 'web' and name != 'historical_baseline' and 'historical_baseline/web' in self.results:
            from report_qwen35_representative import bootstrap
            comparison = bootstrap(Path(self.results['historical_baseline/web']['output']), output)
            delta = score['pauc_f'] - self.results['historical_baseline/web']['score']['pauc_f']
            if abs(delta - comparison['delta_b_minus_a']) > 1e-6:
                raise ValueError('paired WEB bootstrap does not reproduce official score delta')
            result.update(paired_vs_baseline=comparison, meets_54_67=score['pauc_f'] >= 54.67)
        self.results[key] = result
        write(self.results_path, self.results)
        self.planned_comparisons()
        self.report()
        return result

    def planned_comparisons(self):
        comparisons = {}
        for left, right in [('anchor_b32', 'anchor_b128'), ('head_control', 'head_low'),
                            ('replay_scale', 'fresh_scale'), ('rollout_control', 'rollout_corrected')]:
            for cohort in ('development', 'web'):
                a, b = self.results.get(left + '/' + cohort), self.results.get(right + '/' + cohort)
                if not a or not b:
                    continue
                if cohort == 'web':
                    from report_qwen35_representative import bootstrap
                    delta = bootstrap(Path(a['output']), Path(b['output']))
                    point = delta['delta_b_minus_a']
                else:
                    from analyze_qwen35_scaling_diagnostic import bootstrap_delta
                    def per_video(value):
                        records = json.loads((Path(value['output']) / 'official/results/exp10_per_video.json').read_text())
                        return {r['question_id']: r for r in records}
                    delta = bootstrap_delta(per_video(a), per_video(b))
                    point = delta['delta_final_minus_earlier']
                if abs(point - (b['score']['pauc_f'] - a['score']['pauc_f'])) > 1e-6:
                    raise ValueError('planned paired comparison fails scorer reproduction')
                comparisons[f'{right}_minus_{left}/{cohort}'] = delta
        write(self.out / 'reports/planned_comparisons.json', comparisons)

    @staticmethod
    def verify_raw(predictions, expected_decisions):
        count = 0
        for row in predictions:
            trace = row['jevany_action_trace']
            count += len(trace)
            if len(trace) != sum(row['jevany_actions'].values()):
                raise ValueError('incomplete action trace')
            for item in trace:
                logits = item['logits']
                if len(logits) != 3 or not all(math.isfinite(v) for v in logits):
                    raise ValueError('invalid raw action scores')
                action = ('Silence', 'Hold', 'Response')[max(range(3), key=lambda i: logits[i])]
                if item['action'] != action:
                    raise ValueError('action differs from raw learned argmax')
        if count != expected_decisions:
            raise ValueError(f'causal decision coverage mismatch: {count} != {expected_decisions}')

    def report(self):
        lines = ['# Qwen3.5-2B SFT campaign', '',
            'Eight GPUs on ip-10-3-70-177. Raw learned argmax throughout. Full WEB uses the official GPT-4.1 pauc-f-v1 scorer.',
            'Historical quality floor: 54.67 PAUC-F. Results below are completed scores only; missing arms are pending or failed.', '',
            '| Endpoint | WEB PAUC-F | Precision | Recall | Useful / wasted fires | Meets 54.67 |',
            '| --- | ---: | ---: | ---: | ---: | --- |']
        comparisons = []
        for value in self.results.values():
            if value.get('cohort') != 'web':
                continue
            score = value['score']
            lines.append(f"| {value['name']} | {score['pauc_f']:.2f} | {score['fire_precision']:.2f} | {score['pauc_recall_component']:.2f} | {score['num_value_adding_fires']} / {score['num_wasted_fires']} | {score['pauc_f'] >= 54.67} |")
            if 'paired_vs_baseline' in value:
                delta = value['paired_vs_baseline']
                comparisons.append(f"{value['name']} minus the rerun baseline: {delta['delta_b_minus_a']:+.2f}, paired-video 95% interval {delta['delta_ci95']}.")
        for comparison in comparisons:
            lines += ['', comparison]
        lines += ['', 'Development and confirmation use MarkII PAUC-F@4s, a different metric/cohort from WEB.',
                  'WEB results do not select learning rates, batches, thresholds, or checkpoints. Paired-video intervals exclude training-seed and judge variation.',
                  'Clean-anchor training removes known validation/other-benchmark overlaps. Prior margin-adjusted results are retired.']
        (self.out / 'reports/results.md').write_text('\n'.join(lines) + '\n')

    def corrections(self, parent):
        self.run([TRAIN_PY, ROOT / 'scripts/prepare_qwen35_sft_corrections.py', '--output', self.out], 'prepare_correction_cohort.log')
        condition = dict(name='training_contexts', controller=str(parent), margin=0.,
                         output=str(self.out / 'training_rollouts'), manifest=str(self.out / 'data/rollout_train.jsonl'))
        config = self.out / 'training_rollout_conditions.json'
        write(config, [condition], frozen=True)
        if not (self.out / 'training_rollouts/rollout_complete.json').exists():
            self.status('collecting_training_histories', videos=256)
            self.distributed('collect_qwen35_sft_contexts.py', [
                '--conditions', config, '--require-native-moe', '--generator-kernels', 'torch'],
                'collect_training_contexts.log', evaluation=True)
        self.status('gpt_verifying_corrective_labels')
        self.run([TRAIN_PY, ROOT / 'scripts/prepare_qwen35_sft_corrections.py', '--output', self.out, '--label'],
                 'judge_corrective_labels.log')

    def execute(self):
        self.freeze()
        stop = threading.Event()

        def monitor():
            with (self.out / 'logs/gpu_samples.csv').open('a') as handle:
                while not stop.is_set():
                    subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], stdout=handle, stderr=subprocess.STDOUT)
                    handle.flush()
                    stop.wait(10)

        threading.Thread(target=monitor, daemon=True).start()
        try:
            if not (self.out / 'preflight/passed.json').exists():
                self.preflight()
            self.evaluate('historical_baseline', BASELINE, 'development')
            self.evaluate('historical_baseline', BASELINE, 'web')
            failures = {}
            completed = {}
            for name, spec in self.specs.items():
                try:
                    if spec['data'] in ('continuation', 'fresh', 'replay', 'rollout_control', 'rollout_corrected'):
                        self.status('preparing_fresh_replay_data')
                        self.run([TRAIN_PY, ROOT / 'scripts/prepare_qwen35_sft_campaign.py', '--output', self.out,
                                  '--expansion'], 'prepare_expansion.log')
                        continuation = self.out / 'data/continuation.jsonl'
                        if not continuation.exists():
                            with (self.out / 'data/replay.jsonl').open('rb') as source, continuation.open('wb') as target:
                                for index, line in enumerate(source):
                                    if index == 65536:
                                        break
                                    target.write(line)
                    if name.startswith('rollout_'):
                        self.corrections(self.train('anchor_b32', self.specs['anchor_b32']))
                    checkpoint = self.train(name, spec)
                    completed[name] = checkpoint
                    self.evaluate(name, checkpoint, 'development')
                    self.evaluate(name, checkpoint, 'web')
                except Exception as error:
                    failures[name] = dict(error=str(error), traceback=traceback.format_exc())
                    write(self.out / 'reports/failures.json', failures)
                    self.status('arm_failed', arm=name, error=str(error), continuing_independent_arms=True)
            eligible = [name for name in completed if name + '/development' in self.results]
            if not eligible:
                raise RuntimeError('no completed SFT arms available for confirmation')
            # This choice depends exclusively on the preregistered development metric.
            selected = max(eligible, key=lambda name: self.results[name + '/development']['score']['pauc_f'])
            write(self.out / 'selected_recipe.json', dict(name=selected, criterion='Highest development PAUC-F@4s at frozen final endpoint',
                  selection_uses_web=False, training_seeds=[42, 43, 44]), frozen=True)
            self.evaluate('historical_baseline', BASELINE, 'confirmation')
            self.evaluate(selected, completed[selected], 'confirmation')
            for seed in (43, 44):
                try:
                    checkpoint = self.train(selected, self.specs[selected], seed=seed)
                    self.evaluate(selected + f'_seed{seed}', checkpoint, 'confirmation')
                    self.evaluate(selected + f'_seed{seed}', checkpoint, 'web')
                except Exception as error:
                    failures[f'{selected}_seed{seed}'] = dict(error=str(error), traceback=traceback.format_exc())
                    write(self.out / 'reports/failures.json', failures)
            selected_web = [self.results.get(name + '/web') for name in
                            (selected, selected + '_seed43', selected + '_seed44')]
            observed = [r['score']['pauc_f'] for r in selected_web if r]
            decision = dict(selected_recipe=selected, selected_web_scores=observed,
                            all_three_seeds_complete=len(observed) == 3,
                            mean_web_pauc_f=sum(observed) / len(observed) if observed else None,
                            any_new_endpoint_exceeds_54_67=any(r.get('cohort') == 'web' and r.get('name') != 'historical_baseline'
                                and r['score']['pauc_f'] > 54.67 for r in self.results.values()),
                            selected_all_seeds_meet_floor=len(observed) == 3 and min(observed) >= 54.67,
                            selected_all_seeds_beat_rerun_baseline_with_positive_paired_intervals=(
                                len(observed) == 3 and all(r and r.get('paired_vs_baseline', {}).get('delta_ci95', [-math.inf])[0] > 0
                                                           for r in selected_web)),
                            failures=list(failures), promotion_performed=False)
            write(self.out / 'reports/quality_decision.json', decision)
            self.report()
            self.status('complete' if not failures else 'completed_with_failures', **decision,
                        report=str(self.out / 'reports/results.md'))
        finally:
            stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = Campaign(args.output)
    try:
        campaign.execute()
    except BaseException as error:
        campaign.status('failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
