#!/usr/bin/env python3
"""Report episode means alongside official global WEB scores after the pilot."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from audit_jevany_episode_aggregation import clip_statistics
from jevany_rl_reward import sha, write
from report_jevany_e2 import sufficient, metric


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args();run=args.run
    spec={'workflow_freeze_sha256':sha(run/'workflow-freeze.json'),
          'script_sha256':sha(Path(__file__)),
          'method':'arithmetic mean of single-episode PAUC-F; official sufficient statistics, all physical fires; global score checked against unchanged official summary',
          'selection':'all frozen milestone evaluations, no model selection by episode or clip mean'}
    target=run/'aggregation-report-freeze.json'
    if target.exists() and read(target)!=spec:raise ValueError('aggregation report contract changed')
    if not target.exists():write(target,spec)
    if args.watch:
        while True:
            state=read(run/'workflow-status.json')['state']
            result=run/'result.json'
            if state=='failed' or (state in ('complete','stopped_by_coverage_gate') and
                    result.exists() and read(result)['workflow']['state']==state):break
            time.sleep(30)
    results=[]
    for step in read(run/'workflow-freeze.json')['evaluation_steps']:
        directory=run/f'evaluation/web-step{step}'
        summary=directory/'pauc_f_summary-gpt-4.1-v1.json'
        if not summary.exists():continue
        stats=np.array(list(sufficient(directory).values()),dtype=float)
        global_score=float(metric(stats.sum(0)[None])[0])
        if abs(global_score-read(summary)['pauc_f'])>1e-8:raise ValueError('global scorer mismatch')
        pred=directory/'predictions.jsonl';scores=directory/'pauc_llm_scores-gpt-4.1.json'
        predictions=[json.loads(line) for line in pred.read_text().splitlines() if line.strip()]
        judged={r['question_id']:r for r in read(scores)}
        clips=np.array([row for p in predictions for row in clip_statistics(p,judged[p['question_id']])],dtype=float)
        if not np.allclose(clips.sum(0),stats.sum(0),rtol=0,atol=1e-8):raise ValueError('lost physical fires or gold windows')
        results.append({'step':step,'global_pauc_f':global_score,
                        'mean_episode_pauc_f':float(metric(stats).mean()),
                        'mean_clip_pauc_f':float(metric(clips).mean()),
                        'episodes':len(stats),'clips':len(clips),
                        'hashes':{str(p):sha(p) for p in (pred,scores,summary)}})
    write(run/'episode-vs-global.json',{'spec_sha256':sha(target),'evaluations':results,
          'limits':'Reaggregation of saved continuous histories and cumulative GPT judgments; does not simulate resetting generation history. Episode mean is not the acceptance metric.'})
    print(json.dumps({'reported_evaluations':len(results)}),flush=True)


if __name__=='__main__':main()
