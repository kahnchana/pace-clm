#!/usr/bin/env python3
"""Hand off the live SFT job into the user-approved bounded diagnostic budget."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback

from qwen35_sft_campaign import HOST, ROOT, rows, sha, write
from run_qwen35_sft_campaign import Campaign, BASELINE, DEFAULT_OUT, TRAIN_PY

BOUNDED_ARMS = ('fresh_scale', 'replay_scale', 'rollout_control', 'rollout_corrected')
PAIRS = (('replay_scale', 'fresh_scale'), ('rollout_control', 'rollout_corrected'))


def positive_signal(comparison):
    """A compute-allocation criterion, never an inference-time decision adjustment."""
    point = comparison['delta_final_minus_earlier']
    lower, upper = comparison['ci95']
    if not all(math.isfinite(x) for x in (point, lower, upper)) or not lower <= upper:
        raise ValueError('invalid paired quality comparison')
    return point >= 1.0 and lower > 0.0


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return None


class BoundedCampaign(Campaign):
    def prepare(self, parent_pid):
        if socket.gethostname().split('.')[0] != HOST:
            raise RuntimeError('wrong node for SFT campaign')
        self.plan = json.loads((self.out / 'launch_plan.json').read_text())
        self.hashes = self.plan['code_hashes']
        self.specs = self.plan['arms']
        self.check_code()
        state = json.loads((self.out / 'status.json').read_text())
        if (state.get('pid'), state.get('state'), state.get('arm')) != (parent_pid, 'training', 'fresh_scale'):
            raise RuntimeError('handoff requires the original runner to be training fresh_scale')
        command = Path(f'/proc/{parent_pid}/cmdline').read_bytes().split(b'\0')
        if not any(arg.endswith(b'/run_qwen35_sft_campaign.py') or arg == b'scripts/run_qwen35_sft_campaign.py' for arg in command):
            raise RuntimeError('parent PID is not the expected campaign orchestrator')
        children = Path(f'/proc/{parent_pid}/task/{parent_pid}/children').read_text().split()
        training_children = []
        for value in children:
            child = int(value)
            args = Path(f'/proc/{child}/cmdline').read_bytes()
            if b'train_qwen35_sft_campaign.py' in args and b'fresh_scale' in args:
                training_children.append(child)
        if len(training_children) != 1:
            raise RuntimeError('cannot identify exactly one current fresh-data torchrun child')
        amendment = dict(
            date='2026-11-10', host=HOST, gpus=8,
            authorization='User accepted the recommendation to finish fresh/replay and corrected/gold-history comparisons and defer seed replications and larger runs.',
            original_plan_sha256=sha(self.out / 'launch_plan.json'), original_parent_pid=parent_pid,
            retained_training_child_pid=training_children[0], allowed_training_arms=list(BOUNDED_ARMS),
            automatic_additional_seeds=[], larger_doses_authorized=False,
            batch128_retry='Deferred outside this bounded budget; preserve its decoding-failure record.',
            quality_gate=dict(metric='MarkII PAUC-F@4s on development and separate confirmation',
                              minimum_point_gain=1.0, paired_ci95_lower_must_exceed=0.0,
                              note='Compute-allocation criterion only; response margin remains exactly zero.'),
            confirmation='Evaluate both members of any pair passing development, plus the historical baseline, on the existing reserved confirmation cohort.',
            replication='Report eligibility after confirmation; stop for review without automatically launching seeds 43/44.',
            web='Retain full 500-video GPT-4.1 WEB evaluation for each completed fixed endpoint. WEB does not select which pairs receive confirmation.',
            decision_policy='raw_learned_argmax', runner_sha256=sha(Path(__file__)),
            runtime_root=str(ROOT), frozen_runtime_code_hashes=self.hashes,
        )
        write(self.out / 'bounded_amendment.json', amendment, frozen=True)
        # Stop the orchestration process only: the current torchrun and all eight
        # training workers continue running. This prevents any later automatic
        # stage from escaping the newly narrowed budget during the handoff.
        os.kill(parent_pid, signal.SIGSTOP)
        self.status('bounded_handoff_finishing_current_training', arm='fresh_scale',
                    previous_orchestrator_pid=parent_pid, training_child_pid=training_children[0],
                    automatic_seed_replications=False, training_interrupted=False)
        return training_children[0]

    def wait_for_training(self, child, parent):
        while process_state(child) not in (None, 'Z'):
            time.sleep(15)
        # A completed torchrun releases all contexts before the next GPU launch.
        deadline = time.monotonic() + 300
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            if time.monotonic() > deadline:
                raise RuntimeError('GPU contexts remain occupied after training; refusing to overlap jobs')
            time.sleep(10)
        if process_state(parent) not in (None, 'Z'):
            os.kill(parent, signal.SIGTERM)
            os.kill(parent, signal.SIGCONT)
            deadline = time.monotonic() + 30
            while process_state(parent) not in (None, 'Z'):
                if time.monotonic() > deadline:
                    raise RuntimeError('previous orchestrator did not release campaign ownership')
                time.sleep(1)
        # Retain ownership of the original campaign lock for the bounded run.
        self.original_lock = (self.out / 'campaign.lock').open('a')
        fcntl.flock(self.original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (self.out / 'runner.pid').write_text(str(os.getpid()) + '\n')
        self.status('bounded_campaign_active', automatic_seed_replications=False)

    def compare(self, control, candidate, cohort):
        from analyze_qwen35_scaling_diagnostic import bootstrap_delta
        def per_video(name):
            result = self.results[name + '/' + cohort]
            data = json.loads((Path(result['output']) / 'official/results/exp10_per_video.json').read_text())
            return {row['question_id']: row for row in data}
        value = bootstrap_delta(per_video(control), per_video(candidate))
        expected = self.results[candidate + '/' + cohort]['score']['pauc_f'] - self.results[control + '/' + cohort]['score']['pauc_f']
        if abs(value['delta_final_minus_earlier'] - expected) > 1e-6:
            raise ValueError('paired comparison does not reproduce official score delta')
        return value

    def finish_comparisons(self):
        # Reload all results in case the prior orchestrator completed an output
        # before the handoff; retain its audit records and failed batch-128 arm.
        self.results = json.loads(self.results_path.read_text())
        failures = {}
        checkpoints = {}
        corrections_ready = False
        for name in BOUNDED_ARMS:
            try:
                if name.startswith('rollout_') and not corrections_ready:
                    self.corrections(self.train('anchor_b32', self.specs['anchor_b32']))
                    corrections_ready = True
                checkpoint = self.train(name, self.specs[name])
                checkpoints[name] = checkpoint
                self.evaluate(name, checkpoint, 'development')
                self.evaluate(name, checkpoint, 'web')
            except Exception as error:
                failures[name] = dict(error=str(error), traceback=traceback.format_exc())
                write(self.out / 'reports/bounded_failures.json', failures)
                self.status('bounded_arm_failed', arm=name, error=str(error),
                            remaining_arms=list(BOUNDED_ARMS[BOUNDED_ARMS.index(name) + 1:]))
        screening = {}
        for control, candidate in PAIRS:
            if all(name + '/development' in self.results for name in (control, candidate)):
                delta = self.compare(control, candidate, 'development')
                screening[candidate] = dict(control=control, comparison=delta, passes=positive_signal(delta))
        # Freeze this decision before inspecting any new confirmation outcomes.
        write(self.out / 'reports/bounded_development_decision.json', screening, frozen=True)
        confirmation = {}
        if any(value['passes'] for value in screening.values()):
            self.evaluate('historical_baseline', BASELINE, 'confirmation')
            for candidate, value in screening.items():
                if not value['passes']:
                    continue
                control = value['control']
                try:
                    for name in (control, candidate):
                        self.evaluate(name, checkpoints[name], 'confirmation')
                    delta = self.compare(control, candidate, 'confirmation')
                    vs_baseline = self.compare('historical_baseline', candidate, 'confirmation')
                    confirmation[candidate] = dict(control=control, comparison=delta,
                        passes=positive_signal(delta), vs_historical_baseline=vs_baseline,
                        noninferior_to_baseline=vs_baseline['ci95'][0] > -1.0)
                except Exception as error:
                    failures[candidate + '/confirmation'] = dict(error=str(error), traceback=traceback.format_exc())
                    write(self.out / 'reports/bounded_failures.json', failures)
        eligible = [name for name, value in confirmation.items()
                    if value['passes'] and value['noninferior_to_baseline']]
        decision = dict(development=screening, confirmation=confirmation,
                        recipes_eligible_for_replication_review=eligible,
                        automatic_seed_replications=False, automatic_larger_runs=False,
                        failures=failures, decision_policy='raw_learned_argmax',
                        gate_limitation='Predeclared budget criterion; paired-video intervals exclude seed and judge variation. Existing reserved cohorts have prior experimental exposure.')
        write(self.out / 'reports/bounded_decision.json', decision)
        self.report()
        self.status('bounded_complete' if not failures else 'bounded_complete_with_failures',
                    automatic_seed_replications=False, automatic_larger_runs=False,
                    recipes_eligible_for_replication_review=eligible,
                    report=str(self.out / 'reports/bounded_decision.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    lock = (args.output / 'bounded_runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = BoundedCampaign(args.output)
    stop = threading.Event()
    try:
        child = campaign.prepare(args.parent_pid)
        def monitor():
            with (campaign.out / 'logs/gpu_samples.csv').open('a') as handle:
                while not stop.is_set():
                    subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], stdout=handle, stderr=subprocess.STDOUT)
                    handle.flush()
                    stop.wait(10)
        threading.Thread(target=monitor, daemon=True).start()
        campaign.wait_for_training(child, args.parent_pid)
        campaign.finish_comparisons()
    except BaseException as error:
        campaign.status('bounded_driver_failed', error=str(error), traceback=traceback.format_exc(),
                        automatic_seed_replications=False,
                        note='Original orchestration stays stopped if handoff has not completed; the current training child was not stopped.')
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
