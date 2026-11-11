#!/usr/bin/env python3
"""Reaggregate saved judgments without pretending to regenerate clip histories."""
import argparse
import json
from pathlib import Path

import numpy as np
from jevany_rl_reward import ROOT, OFFICIAL, sha, write
from report_jevany_e2 import metric, sufficient
from zs_infer.pauc_f import fire_times, assign_value_adding_fires


def clip_statistics(prediction, scored, width=32):
    """Extend boundaries across intersecting windows; conserve every physical fire."""
    turns=sorted(scored['answer'],key=lambda t:t['reply_timespan'][0])
    fires=fire_times(prediction)
    end=max([float(scored.get('duration',0)), *(t['reply_timespan'][1] for t in turns),
             *(t+1e-6 for t in fires)], default=0)
    output=[];start=0.
    while start<end:
        stop=min(start+width,end)
        changed=True
        while changed:
            changed=False
            for turn in turns:
                left,right=turn['reply_timespan']
                if left<stop<=right and stop<end:
                    stop=min(right+1e-6,end);changed=True
        selected=[t for t in turns if start<=t['reply_timespan'][0]<stop]
        physical=[t for t in fires if start<=t<stop]
        useful=len(assign_value_adding_fires(physical,selected))
        areas=[]
        for turn in selected:
            points=[[float(t)-turn['reply_timespan'][0],v] for t,v in turn['judge_scores'].items()]
            areas.append(OFFICIAL.area_under_line_ratio(points,turn['reply_timespan'][1]-turn['reply_timespan'][0],2,.5,.5) if points else .25)
        output.append([useful,len(physical),sum(areas),len(areas)])
        start=stop
    if sum(r[1] for r in output)!=len(fires) or sum(r[3] for r in output)!=len(turns):
        raise ValueError('clip partition lost fires or events')
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sources={
        'retained_sft':ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web',
        'anchored_clip_rloo_24':ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1/evaluation/web-step24',
        'anchored_clip_rloo_32':ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1/evaluation/web-step32',
        'reward_aligned_rloo_16':ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1/full_web_v1/web-step16',
        'fixed_grpo_16':ROOT/'outputs/jevany-2b-rl/reward-aligned-grpo-v2/evaluation/web-step16'}
    result={};missing={}
    for name,directory in sources.items():
        pred=directory/'predictions.jsonl';scores=directory/'pauc_llm_scores-gpt-4.1.json'
        if not pred.exists() or not scores.exists():missing[name]=str(directory);continue
        statistics=np.array(list(sufficient(directory).values()),dtype=float)
        predictions=[json.loads(line) for line in pred.read_text().splitlines() if line.strip()]
        judged={r['question_id']:r for r in json.loads(scores.read_text())}
        clips=np.array([r for p in predictions for r in clip_statistics(p,judged[p['question_id']])],dtype=float)
        if not np.allclose(clips.sum(0),statistics.sum(0),rtol=0,atol=1e-8):
            raise ValueError('partition statistics differ from original episodes')
        global_score=float(metric(statistics.sum(0)[None])[0])
        summary=json.loads((directory/'pauc_f_summary-gpt-4.1-v1.json').read_text())
        if abs(global_score-summary['pauc_f'])>1e-8:raise ValueError('official global scorer mismatch')
        result[name]={'global_pauc_f':global_score,'mean_episode_pauc_f':float(metric(statistics).mean()),
                      'mean_clip_pauc_f':float(metric(clips).mean()),'clips':len(clips),
                      'event_free_clips':int((clips[:,3]==0).sum()),'episodes':len(statistics),
                      'hashes':{str(p):sha(p) for p in (pred,scores)},
                      'all_physical_fires_and_complete_event_windows_preserved':True}
    rankings={field:sorted(result,key=lambda name:result[name][field],reverse=True)
              for field in ('global_pauc_f','mean_episode_pauc_f','mean_clip_pauc_f')}
    write(args.output,{'models':result,'rankings':rankings,'missing':missing,
          'scope':'aggregation only, saved full-video cumulative judgments; no clip-history regeneration',
          'clip_partition':'nominal 32s, extend boundaries to preserve complete windows, retain quiet clips and all fires'})
    print(json.dumps({'models':{n:{k:v for k,v in r.items() if k!='hashes'} for n,r in result.items()},'rankings':rankings,'missing':missing}),flush=True)


if __name__=='__main__':main()
