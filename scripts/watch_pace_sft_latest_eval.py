#!/usr/bin/env python3
"""Serial latest-checkpoint evaluations. Preview by default; --run starts jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/jevany-qwen35-2b-pace-estp-sft'
DEFAULT_RUN = BASE / 'base-50-50-b384-to-estp-epoch5'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def latest(run):
    # commit.json is published last; ignore directories still being written.
    commits = list((run / 'train').glob('step-*/commit.json'))
    if not commits:
        return None
    commit = max(commits, key=lambda p: int(p.parent.name.split('-')[-1]))
    data = read(commit)
    if data['step'] != int(commit.parent.name.split('-')[-1]):
        raise ValueError('Checkpoint path/commit step mismatch')
    return dict(step=data['step'], checkpoint=str(commit.parent.resolve()),
                checkpoint_commit_sha256=sha(commit))


def compatible(dispatch, template):
    keys = ['source_hashes', 'population', 'implementation', 'hard_time_limit']
    return all(dispatch.get(k) == template.get(k) for k in keys)


def completed(selection, template, base=BASE):
    for path in base.glob('*/evaluation/*/complete.json'):
        dispatch = path.parent / 'dispatch.json'
        if not dispatch.exists():
            continue
        info = read(dispatch)
        if (Path(info['checkpoint']).resolve() == Path(selection['checkpoint']).resolve()
                and info.get('checkpoint_commit_sha256') == selection['checkpoint_commit_sha256']
                and compatible(info, template)):
            result = read(path)
            if result.get('step') == selection['step'] and set(result.get('scores', {})) == {'web', 'ego', 'estp_bench_explicit_200'}:
                return path.parent
    return None


def decision(run, template, base=BASE):
    selection = latest(run)
    if selection is None:
        return dict(state='waiting_for_first_commit')
    done = completed(selection, template, base)
    if done:
        final = run / 'training-complete.json'
        finished = final.exists() and read(final)['step'] == selection['step']
        return dict(state='complete' if finished else 'waiting_for_new_checkpoint',
                    evaluated_output=str(done), **selection)
    return dict(state='evaluate', **selection)


def other_gpu_job(current=None):
    # Recognize earlier manual evaluations and their recovery watchers.
    for path in BASE.glob('*/evaluation/*/launch.json'):
        # Control directories also have launch.json; only suite dispatches use GPUs.
        if not (path.parent / 'dispatch.json').exists():
            continue
        if current is not None and path.parent.resolve() == current.resolve():
            continue
        if alive(read(path)['pid']):
            return str(path.parent)
        watcher = path.parent / 'watcher-launch.json'
        if not (path.parent / 'complete.json').exists() and watcher.exists() and alive(read(watcher)['pid']):
            return str(path.parent)
    return None


def prepare_output(run, selection, template):
    checkpoint = Path(selection['checkpoint'])
    commit = read(checkpoint / 'commit.json')
    if sha(checkpoint / 'commit.json') != selection['checkpoint_commit_sha256']:
        raise ValueError('Pinned checkpoint commit changed')
    for name in ['pointer_head.pt', 'controller_adapter/adapter_model.safetensors']:
        if sha(checkpoint / name) != commit['files'][name]:
            raise ValueError('Checkpoint checksum mismatch: ' + name)
    for path, digest in template['source_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Validated evaluator changed: ' + path)
    out = run / 'evaluation' / f"fast-step-{selection['step']:06d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'user-stopped.json').exists():
        raise RuntimeError('Selected evaluation has a user stop marker')
    if (out / 'dispatch.json').exists():
        existing = read(out / 'dispatch.json')
        if (not compatible(existing, template)
                or existing['checkpoint'] != selection['checkpoint']
                or existing['checkpoint_commit_sha256'] != selection['checkpoint_commit_sha256']):
            raise ValueError('Existing evaluation has a different checkpoint/protocol')
    else:
        (out / 'code').mkdir(exist_ok=True)
        for path in template['source_hashes']:
            shutil.copy2(path, out / 'code' / Path(path).name)
        dispatch = dict(template, **selection, authorized_utc=now(),
                        scheduling='continuous newest committed checkpoint; serial suites')
        dispatch.pop('selected_utc', None)
        write(out / 'dispatch.json', dispatch)
    return out


def pause(state, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (state / 'STOP').exists():
            return
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def supervise(out, state, selection, max_attempts):
    """Recover this same pinned checkpoint, with attempts persisted across restarts."""
    watcher = out / 'watcher-launch.json'
    if watcher.exists() and alive(read(watcher)['pid']):
        write(state / 'status.json', dict(state='waiting_for_existing_evaluation_watcher', output_dir=str(out), utc=now()))
        pause(state, 30)
        return False
    launch_path = out / 'launch.json'
    if launch_path.exists() and alive(read(launch_path)['pid']):
        write(state / 'status.json', dict(state='evaluating', output_dir=str(out), **selection, utc=now()))
        pause(state, 30)
        return False
    if (out / 'complete.json').exists():
        return True
    attempts_path = out / 'continuous-attempts.json'
    attempts = read(attempts_path)['attempts'] if attempts_path.exists() else 0
    if attempts >= max_attempts:
        raise RuntimeError(f'Technical attempts exhausted for step {selection["step"]}; preserved partial results')
    busy = other_gpu_job(out)
    if busy:
        write(state / 'status.json', dict(state='waiting_for_other_evaluation', other_output=busy, utc=now()))
        pause(state, 30)
        return False
    # Read-only probe on the assigned evaluation allocation; never allocate nodes.
    check = subprocess.run(['srun', '--jobid=38177', '--overlap', '--cpu-bind=none', '--nodes=1', '--ntasks=1',
                            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    if check.stdout.strip():
        write(state / 'status.json', dict(state='waiting_for_evaluation_gpu_release', utc=now()))
        pause(state, 30)
        return False
    write(attempts_path, dict(attempts=attempts + 1, max_attempts=max_attempts, utc=now()))
    with (out / 'launcher.log').open('ab', buffering=0) as handle:
        proc = subprocess.Popen(['srun', '--jobid=38177', '--overlap', '--cpu-bind=none', '--nodes=1', '--ntasks=1',
                                 'bash', 'scripts/run_pace_sft_budget_eval.sh', '--checkpoint', selection['checkpoint'], '--output-dir', str(out)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write(launch_path, dict(pid=proc.pid, allocation=38177, utc=now()))
    write(state / 'status.json', dict(state='evaluating', output_dir=str(out), attempt=attempts + 1, **selection, utc=now()))
    pause(state, 30)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--template', type=Path, default=DEFAULT_RUN / 'evaluation/fast-step-006249/dispatch.json')
    parser.add_argument('--poll-seconds', type=int, default=600)
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--run', action='store_true', help='Launch jobs; without this flag only preview the next decision')
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_attempts < 1:
        parser.error('poll-seconds and max-attempts must be positive')
    run = args.training_run.resolve()
    if run != DEFAULT_RUN.resolve():
        parser.error('Only the current authorized five-epoch SFT continuation is supported')
    template = read(args.template)
    if not args.run:
        print(json.dumps(dict(mode='preview_only_no_jobs_started', poll_seconds=args.poll_seconds,
                              allocation=38177, next=decision(run, template)), indent=2))
        return
    state = run / 'evaluation/continuous-latest'
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Shared lock prevents two copies with different state paths from dispatching.
    global_lock = (BASE / 'continuous-evaluation.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while not (state / 'STOP').exists():
            active = state / 'active.json'
            if active.exists():
                selection = read(active)
            else:
                item = decision(run, template)
                write(state / 'status.json', dict(item, utc=now()))
                if item['state'] == 'complete':
                    return
                if item['state'] != 'evaluate':
                    pause(state, args.poll_seconds)
                    continue
                busy = other_gpu_job()
                if busy:
                    write(state / 'status.json', dict(state='waiting_for_other_evaluation', other_output=busy, utc=now()))
                    pause(state, 30)
                    continue
                selection = {k: item[k] for k in ['step', 'checkpoint', 'checkpoint_commit_sha256']}
                write(active, selection)
            out = prepare_output(run, selection, template)
            if supervise(out, state, selection, args.max_attempts):
                write(state / 'last-completed.json', dict(**selection, output_dir=str(out), utc=now()))
                active.unlink()
        write(state / 'status.json', dict(state='stopped_dispatching', note='Any already running evaluation is allowed to finish', utc=now()))
    except Exception as exc:
        write(state / 'needs-attention.json', dict(error=str(exc), utc=now()))
        raise


if __name__ == '__main__':
    main()
