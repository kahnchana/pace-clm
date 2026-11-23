#!/usr/bin/env python3
"""Queue one latest committed SFT evaluation after a pinned suite completes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/jevany-qwen35-2b-pace-estp-sft'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue.resolve()
    lock = (queue / 'dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(queue / 'queue.json')
    previous = Path(config['after'])
    run = Path(config['training_run'])
    while not (queue / 'user-stopped.json').exists():
        if (queue / 'complete.json').exists():
            return
        if not (previous / 'complete.json').exists():
            write(queue / 'status.json', dict(state='waiting_for_step4550_evaluation', utc=now()))
            time.sleep(30)
            continue
        if alive(read(previous / 'launch.json')['pid']):
            write(queue / 'status.json', dict(state='waiting_for_evaluation_gpu_release', utc=now()))
            time.sleep(30)
            continue
        # Persist the chosen checkpoint once, including across dispatcher recovery.
        if not (queue / 'selection.json').exists():
            commits = sorted((run / 'train').glob('step-*/commit.json'))
            if not commits:
                raise RuntimeError('No fully committed SFT checkpoint')
            commit_path = commits[-1]
            commit = read(commit_path)
            checkpoint = commit_path.parent.resolve()
            for name in ['pointer_head.pt', 'controller_adapter/adapter_model.safetensors']:
                if sha(checkpoint / name) != commit['files'][name]:
                    raise ValueError('Checkpoint checksum mismatch: ' + name)
            # Completed evaluations anywhere in this SFT family count as already done.
            already_evaluated = None
            for done in BASE.glob('*/evaluation/*/complete.json'):
                dispatch = done.parent / 'dispatch.json'
                if dispatch.exists() and Path(read(dispatch)['checkpoint']).resolve() == checkpoint:
                    already_evaluated = done.parent
                    break
            if already_evaluated is not None:
                interval = config.get('poll_seconds_if_already_evaluated', 600)
                write(queue / 'status.json', dict(state='waiting_for_new_checkpoint', skipped_step=commit['step'], checkpoint=str(checkpoint), existing=str(already_evaluated), poll_seconds=interval, next_check_epoch=time.time()+interval, utc=now()))
                for _ in range((interval+29)//30):
                    if (queue / 'user-stopped.json').exists():
                        return
                    time.sleep(30)
                continue
            write(queue / 'selection.json', dict(step=commit['step'], checkpoint=str(checkpoint), selected_utc=now(), checkpoint_commit_sha256=sha(commit_path)))
        selected = read(queue / 'selection.json')
        out = run / 'evaluation' / f"fast-step-{selected['step']:06d}"
        template = read(previous / 'dispatch.json')
        for path, digest in template['source_hashes'].items():
            if sha(Path(path)) != digest:
                raise ValueError('Validated implementation changed: ' + path)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'code').mkdir(exist_ok=True)
        dispatch = dict(template, **selected, authorized_utc=config['authorized_utc'], comparison_step=4550)
        if (out / 'dispatch.json').exists():
            existing = read(out / 'dispatch.json')
            if existing['checkpoint'] != selected['checkpoint']:
                raise ValueError('Output checkpoint conflict')
        else:
            for path in template['source_hashes']:
                shutil.copy2(path, out / 'code' / Path(path).name)
            write(out / 'dispatch.json', dispatch)
        if not (out / 'launch.json').exists():
            with (out / 'launcher.log').open('ab', buffering=0) as handle:
                proc = subprocess.Popen(['srun', '--jobid=38177', '--overlap', '--cpu-bind=none', '--nodes=1', '--ntasks=1', 'bash', 'scripts/run_pace_sft_budget_eval.sh', '--checkpoint', selected['checkpoint'], '--output-dir', str(out)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            write(out / 'launch.json', dict(pid=proc.pid, allocation=38177, utc=now()))
        if not (out / 'watcher-launch.json').exists() or not alive(read(out / 'watcher-launch.json')['pid']):
            with (out / 'watcher.log').open('ab', buffering=0) as handle:
                watcher = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/watch_pace_sft_fast_eval.py'), '--output-dir', str(out)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            write(out / 'watcher-launch.json', dict(pid=watcher.pid, utc=now()))
        write(queue / 'status.json', dict(state='dispatched', output_dir=str(out), **selected, utc=now()))
        write(queue / 'complete.json', dict(state='dispatch_complete', output_dir=str(out), **selected, utc=now()))
        return


if __name__ == '__main__':
    main()
