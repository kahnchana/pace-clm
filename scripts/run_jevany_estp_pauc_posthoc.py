#!/usr/bin/env python3
"""User-authorized GPT-only explicit PAUC-F rescoring of saved steps 32/128."""
import json
import fcntl
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('post-hoc scoring must have GPU visibility disabled')
    lock=(RUN/'estp-pauc-posthoc.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=RUN/'estp-pauc-posthoc-status.json'
    def status(state,**fields):
        write(status_path,dict(state=state,utc=datetime.now(timezone.utc).isoformat(),
            metric='estp-pauc-f-v1',new_inference=False,authorized_steps=[32,128],**fields))
    summaries={}
    for step,name in [(32,'final-reference-step32'),(128,'full-step128')]:
        source=RUN/'evaluation'/name;output=source/'estp_pauc_explicit'
        status('waiting_for_source_result',step=step)
        while not (source/'result.json').exists():time.sleep(10)
        output.mkdir(exist_ok=True)
        while True:
            try:
                status('gpt_scoring',step=step,output=str(output))
                with (output/'judge.log').open('a') as handle:
                    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/score_jevany_estp_pauc.py'),
                        '--source-eval',str(source),'--judge'],cwd=ROOT,stdout=handle,
                        stderr=subprocess.STDOUT,check=True)
                summary=json.loads((output/'pauc_f_summary-gpt-4.1.json').read_text())
                if not summary['headline_eligible'] or len(summary['per_task'])!=8:
                    raise ValueError('incomplete judged explicit task scores')
                summaries[str(step)]=summary
                status('checkpoint_complete',step=step,explicit_all_pauc_f=summary['all']['omega_0.5'])
                break
            except Exception as exc:
                status('retrying',step=step,error=f'{type(exc).__name__}: {exc}')
                with (output/'recovery.log').open('a') as handle:traceback.print_exc(file=handle)
                time.sleep(30)
    lines=['# RL checkpoint ESTP-Bench explicit PAUC-F','',
        'Saved predictions only; no new model inference. User authorized additional GPT-4.1 rescoring on 2026-11-16.',
        'Unchanged stream-gemma estp-pauc-f-v1: 851 cases, strict native positive-width windows, omega 0.5, eight-task macro-average.',
        'The 71 zero-duration gold turns are excluded by this PAUC protocol. Official ESTP-F1 outputs remain separate and unchanged.','',
        '| Task | Step 32 | Step 128 | Change |','|---|---:|---:|---:|']
    for task in summaries['32']['per_task']:
        a=summaries['32']['headline_task_pauc_f'][task];b=summaries['128']['headline_task_pauc_f'][task]
        lines.append(f'| {task} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
    a=summaries['32']['all']['omega_0.5'];b=summaries['128']['all']['omega_0.5']
    lines += [f'| **Explicit All (task macro)** | **{a:.3f}** | **{b:.3f}** | **{b-a:+.3f}** |','',
        'Task-level precision, temporal recall, useful/wasted fires, source hashes, requests and raw judge responses are retained in each endpoint directory under estp_pauc_explicit/.',
        'These are reused benchmark comparisons, not untouched tests or a step-160 result. No further model evaluations are queued.','']
    target=ROOT/'docs/experiments/jevany_2b_estp_explicit_pauc_results.md';target.write_text('\n'.join(lines))
    write(RUN/'estp-pauc-posthoc-comparison.json',dict(summaries=summaries,document=str(target)))
    status('complete',document=str(target),scores={k:v['all']['omega_0.5'] for k,v in summaries.items()})


if __name__=='__main__':main()
