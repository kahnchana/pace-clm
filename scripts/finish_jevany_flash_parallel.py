#!/usr/bin/env python3
"""Run the approved full Flash suite after smoke; preserve retries and report."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation'
TARGET=BASE/'web-ego-tv-subs-vad-flash-step160'
SMOKE=BASE/'web-ego-tv-subs-vad-flash-step160-smoke'

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def main():
    smoke=json.loads((SMOKE/'status.json').read_text())
    if smoke['state']!='complete': raise RuntimeError('all four smoke suites must pass first')
    for attempt in range(1,4):
        with (TARGET/f'launch-attempt-{attempt}.log').open('a') as log:
            process=subprocess.Popen(['bash',str(ROOT/'scripts/run_jevany_flash_parallel.sh'),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(TARGET/'supervisor.json',dict(state='running',pid=os.getpid(),child_pid=process.pid,attempt=attempt,utc=datetime.now(timezone.utc).isoformat()))
            code=process.wait()
        if code==0: break
        write(TARGET/'supervisor.json',dict(state='retry_pending',attempt=attempt,exit_code=code,utc=datetime.now(timezone.utc).isoformat()))
        if attempt<3: time.sleep(30)
    else:
        write(TARGET/'supervisor.json',dict(state='failed',attempts=3,utc=datetime.now(timezone.utc).isoformat()))
        raise RuntimeError('full suite failed after three attempts; completed shards and judge caches preserved')
    result=json.loads((TARGET/'result.json').read_text())
    lines=['# Step-160 controller with Gemini 3.8 Flash','',
           'All four suites completed inference, GPT judging, and unchanged official PAUC-F scoring on allocation 38177.',
           'Controller decisions use raw learned argmax with margin zero; answer history is generated causally by Flash.',
           'TV uses measured source FPS, sampling from frame zero, and current-frame-only native subtitles.',
           'Training cohorts overlap RL training; use the heldout cohorts for generalization. Timing-only scores are diagnostics.',
           '', '| Cohort | Episodes | Flash PAUC-F | Sol PAUC-F | Difference |', '|---|---:|---:|---:|---:|']
    frozen=json.loads((TARGET/'freeze.json').read_text())
    for domain in ('web','ego','tv','vad'):
        suite=result['suites'][domain]
        baseline=Path(frozen['suites'][domain]['sol_baseline'])
        sol=json.loads(baseline.read_text()) if baseline.exists() else None
        for cohort,stats in suite['cohorts'].items():
            if cohort not in (domain+'_all',domain+'_train',domain+'_heldout'): continue
            score=stats['pauc_f']
            other=sol['cohorts'].get(cohort) if sol else None
            sol_score=f"{other['pauc_f']:.2f}" if other else 'pending'
            delta=f"{score-other['pauc_f']:+.2f}" if other else 'pending'
            lines.append(f"| {cohort} | {stats['episodes']} | {score:.2f} | {sol_score} | {delta} |")
    lines.extend(['','Detailed per-episode metrics, call audits, raw action scores, cohort statistics, and timing diagnostics are in each suite directory.',''])
    (TARGET/'completion-report.md').write_text('\n'.join(lines))
    write(TARGET/'supervisor.json',dict(state='complete',pid=os.getpid(),report=str(TARGET/'completion-report.md'),utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
