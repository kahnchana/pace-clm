#!/usr/bin/env python3
"""Score one complete saved cohort while the faster GPU job continues."""
import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime,timezone
from pathlib import Path
from prepare_pace_estp_sft import ROOT,read,write,dump,sha
from jevany_mixed_reward import score_predictions
from eval_jevany_mixed_queue import stats

PARENT=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h'
DESIGN=PARENT.with_name('eval-speed-design-v1')


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--suite',choices=['web','ego'],required=True);args=parser.parse_args()
    out=args.output_dir.resolve();suite=args.suite;directory=out/suite;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return
    freeze=read(out/'freeze.json');cohorts=read(PARENT/'data/cohorts.json');wanted=set(cohorts[suite+'_all']);selection=read(DESIGN/'data/selection-web-first.json');specs={s['span_id']:s for s in rows(DESIGN/'data/specs.jsonl')}
    selected={};sources=[]
    for rank in range(8):
        expected=wanted&set(selection[rank::8]);found=set();path=out/'rollout'/f'predictions.rank{rank:02d}.jsonl'
        with path.open('rb') as handle:
            bound=path.stat().st_size
            while handle.tell()<bound and found!=expected:
                line=handle.readline()
                if not line.endswith(b'\n'):break
                p=json.loads(line);qid=p['question_id']
                if qid in wanted:
                    if qid in selected:raise ValueError('duplicate cohort prediction')
                    trace=p['jevany_action_trace']
                    if len(trace)!=specs[qid]['rollout_frames']:raise ValueError('incomplete saved episode')
                    if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):raise ValueError('non-argmax action')
                    selected[qid]=p;found.add(qid)
            sources.append(dict(path=str(path),snapshot_bytes=bound,episodes=len(found)))
        if found!=expected:raise ValueError('cohort is not fully complete on every worker')
    if set(selected)!=wanted:raise ValueError('cohort coverage mismatch')
    predictions=[selected[q] for q in sorted(wanted)];dump(directory/'saved-predictions.jsonl',predictions)
    write(directory/'early-scoring-snapshot.json',dict(step=freeze['step'],sources=sources,predictions_sha256=sha(directory/'saved-predictions.jsonl'),script_sha256=sha(Path(__file__)),new_inference=False))
    cache=out/'cache/evaluation.sqlite';old=PARENT/'evaluation/corrected-step143/cache/evaluation.sqlite'
    if old.exists() and not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with sqlite3.connect(old) as source,sqlite3.connect(cache) as target:source.backup(target)
        write(out/'cache-seed.json',dict(source=str(old),source_sha256=sha(old),rule='Exact request/unchanged scoring-contract hashes only'))
    gold=[dict(question_id=p['question_id'],conversation=[dict(role='user',content=specs[p['question_id']]['question'])],answer=specs[p['question_id']]['gold_answers']) for p in predictions]
    started=time.monotonic();utc=datetime.now(timezone.utc).isoformat();metrics,usage=score_predictions(predictions,gold,directory/'scoring',cache)
    result=dict(step=freeze['step'],historical_cohort_note='Train/heldout names refer to prior RL identities; this fresh SFT trained only PACE and ESTP-IT.',
        cohorts={k:stats(metrics,ids) for k,ids in cohorts.items() if k.startswith(suite+'_')},per_episode=metrics,judge_usage=usage,
        scoring_seconds=time.monotonic()-started,scoring_started_utc=utc,scoring_finished_utc=datetime.now(timezone.utc).isoformat(),scored_during_remaining_inference=True)
    write(directory/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='per_episode'}),flush=True)


if __name__=='__main__':main()
