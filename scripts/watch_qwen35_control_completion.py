#!/usr/bin/env python3
"""Notify the existing Codex chat after training and all four judged WEB runs finish."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run_qwen35_control import BASE, ROOT, verify_web, write_json


def inspect_run(out):
    status = json.loads((out/'status.json').read_text())
    plan = json.loads((out/'launch_plan.json').read_text())
    latest_path = out/'train/latest.json'
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else {'step':0}
    final_web = out/'evaluation'/f"web_step{plan['total_steps']:06d}"
    if latest['step'] >= plan['total_steps'] and (final_web/'pauc_f_summary-gpt-4.1-v1.json').exists():
        try:
            checkpoints = []
            for step in plan['milestones']:
                checkpoint = out/'train'/f'step-{step:06d}'
                for name in ('training_state.pt','pointer_head.pt','controller_adapter/adapter_model.safetensors'):
                    if not (checkpoint/name).is_file():
                        raise ValueError(f'missing checkpoint artifact: {checkpoint/name}')
                web = out/'evaluation'/f'web_step{step:06d}'
                score, _, _ = verify_web(web)
                protocol = json.loads((web/'manifest.json').read_text())
                if (protocol['source_sha256'],protocol['original_generator_adapter_sha256']) != (plan['web_gold_sha256'],plan['generator_adapter_sha256']):
                    raise ValueError('WEB data or generator protocol differs')
                checkpoints.append({'step':step,'pauc_f':score['pauc_f'],'videos':score['num_videos']})
            return {'condition':'complete','checkpoints':checkpoints,'run_state':status['state']}
        except (OSError,ValueError,KeyError) as error:
            if status['state'] == 'complete':
                return {'condition':'failed','error':f'Completion verification failed: {error}'}
    if status['state'] == 'failed':
        return {'condition':'failed','error':status.get('error','Runner reported failure'),'phase':status.get('phase')}
    try:
        os.kill(int(status['pid']),0)
    except ProcessLookupError:
        return {'condition':'failed','error':'Orchestrator exited before all four judged WEB evaluations were verified','phase':status.get('phase')}
    return {'condition':'waiting','run_state':status['state'],'phase':status.get('phase'),'saved_step':latest['step']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=BASE/'control-seed42')
    parser.add_argument('--thread',required=True)
    parser.add_argument('--codex',default='codex')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--check-once',action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.check_once:
        print(json.dumps(inspect_run(out)))
        return
    lock = (out/'completion_watch.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record_path = out/'completion_watch.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    if record.get('completion_queued'):
        return
    record.update(pid=os.getpid(),thread_id=args.thread,interval_seconds=args.interval,active=True)
    while True:
        record['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            result = inspect_run(out)
            record['observation'] = result
            kind = result['condition']
            fingerprint = json.dumps(result,sort_keys=True)
            if kind == 'complete' or (kind == 'failed' and fingerprint != record.get('last_failure_queued')):
                message = (
                    'The user requested a notification in this chat when the Qwen3.5-2B control run '
                    'and all four checkpoint WEB evaluations finish. The local completion watcher '
                    f'observed: {json.dumps(result)}. Run artifacts: {out}. '
                    'Read the current status and confirm the artifacts before replying. '
                    'If complete, notify the user and summarize all four checkpoint WEB scores '
                    '(PAUC-F, fire precision, PAUC recall) with links. If failed, alert the user '
                    'that the run stopped before verified completion and identify the failure. '
                    'Do not restart or change training as part of this notification.'
                )
                delivered = subprocess.run([args.codex,'queue','--thread',args.thread,'--message',message,
                                             '-C',str(ROOT)],capture_output=True,text=True,timeout=45)
                record['queue_returncode'] = delivered.returncode
                record['queue_output'] = delivered.stdout[-2000:]
                record['queue_error'] = delivered.stderr[-2000:]
                if delivered.returncode == 0:
                    if kind == 'complete':
                        record.update(completion_queued=True,active=False)
                        write_json(record_path,record)
                        return
                    record['last_failure_queued'] = fingerprint
        except Exception as error:
            record['watch_error'] = repr(error)
        write_json(record_path,record)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
