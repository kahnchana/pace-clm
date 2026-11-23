#!/usr/bin/env python3
"""Report completed faster evaluation with shared startup and actual wall time."""
import json
import re
import time
from datetime import datetime,timezone
from pathlib import Path


def read(p):return json.loads(p.read_text())
def write(p,v):
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(v,indent=2)+'\n');temporary.replace(p)


def report(out):
    summary=read(out/'metrics-summary.json');freeze=read(out/'freeze.json');population=freeze['population']['sets']
    suites=['web','ego','estp-bench-explicit-200'];processing={s:{} for s in suites}
    for path in (out/'rollout').glob('predictions.rank*.jsonl'):
        rank=int(path.stem[-2:])
        with path.open() as handle:
            for line in handle:
                row=json.loads(line);qid=row['question_id'];suite='web' if qid.startswith('web:') else 'ego' if qid.startswith('ego:') else 'estp-bench-explicit-200'
                processing[suite][rank]=processing[suite].get(rank,0.)+row['jevany_episode_seconds']
    log=(out/'inference.log').read_text(errors='replace');loading={}
    for match in re.findall(r'\{"direct_gpu_moe_load_seconds"[^\n]+\}',log):
        value=json.loads(match);loading[str(value['local_rank'])]=value['direct_gpu_moe_load_seconds']
    details={s:read(out/s/'result.json') for s in suites}
    summary['total_seconds']=time.time()-read(out/'run-start.json')['started_epoch']
    summary['finished_utc']=datetime.now(timezone.utc).isoformat()
    summary['processing_seconds_by_suite_by_rank']=processing
    summary['moe_load_seconds_by_gpu']=loading
    summary['hard_time_limit']=False
    lines=[f"# SFT checkpoint {summary['step']}: faster evaluation",'',
        '| Set | Source videos | Episodes | Max worker processing time | GPT/scoring | PAUC-F |',
        '|---|---:|---:|---:|---:|---:|']
    for suite in suites:
        p=population[suite];d=details[suite]
        score=d['summary']['all']['omega_0.5'] if suite=='estp-bench-explicit-200' else d['cohorts'][suite+'_all']['pauc_f']
        lines.append(f"| {suite} | {p['unique_video_ids']} | {p['episodes']} | {max(processing[suite].values())/60:.2f} min | {d['scoring_seconds']:.2f} sec | {score:.3f} |")
    lines+=['',f"**Actual complete-run wall time: {summary['total_seconds']/60:.2f} minutes.**",
        f"Combined inference including shared model loading and merge: {summary['inference_seconds']/60:.2f} minutes. Total recorded GPT/scoring time: {summary['scoring_seconds']/60:.2f} minutes.",'',
        'Per-set processing is the largest sum of episode wall times on any worker and excludes shared model loading. Workers can overlap suite transitions, so those per-set maxima are not additive. The complete-run wall time is measured from the first worker invocation, including any recovery intervals. No three-hour cutoff was applied.','',
        'ESTP-Bench Explicit uses the fixed seed-42 diagnostic subset of 200 whole episodes, 25 per task, rather than the full 851-case benchmark. WEB/EGO are complete. All suites use the same pinned checkpoint, raw learned argmax, margin zero, unchanged native windows and histories, frozen promoted MoE greedy answers at 96 tokens, and GPT-4.1 judging.','',
        'Historical WEB/EGO train/heldout labels refer to earlier RL identities. This fresh SFT trained only on PACE and ESTP-IT.']
    if loading:lines+=['',f"Generator loading, measured independently on the eight GPUs: {min(loading.values()):.2f}–{max(loading.values()):.2f} seconds. This loading is shared across all three sets."]
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(out/'complete.json',summary)
    return summary
