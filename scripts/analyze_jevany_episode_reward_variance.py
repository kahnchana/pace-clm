#!/usr/bin/env python3
"""Describe four-rollout episode reward variation without loading a GPU."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def describe(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'min':float(a.min()),
            **{f'p{int(q*100)}':float(np.quantile(a,q)) for q in (.1,.25,.5,.75,.9,.95)},
            'max':float(a.max())}


def summarize(groups):
    rewards=np.array([g['rewards_pauc_f_points'] for g in groups])
    ranges=np.ptp(rewards,axis=1);sample_vars=np.var(rewards,axis=1,ddof=1)
    advantages=4/3*(rewards-rewards.mean(1,keepdims=True))/100
    flat=ranges<1e-9
    return {'groups':len(groups),'trajectories':rewards.size,
            'sample_variance_points_squared':describe(sample_vars),
            'sample_std_points':describe(np.sqrt(sample_vars)),
            'pooled_within_group_sample_std_points':float(np.sqrt(sample_vars.mean())),
            'range_points':describe(ranges),
            'range_histogram':[
                {'label':label,'groups':int(mask.sum()),'percent':float(100*mask.mean())}
                for label,mask in (
                    ('equal',flat),('0 < range <= 5',(ranges>=1e-9)&(ranges<=5)),
                    ('5 < range <= 10',(ranges>5)&(ranges<=10)),
                    ('10 < range <= 25',(ranges>10)&(ranges<=25)),
                    ('25 < range <= 50',(ranges>25)&(ranges<=50)),
                    ('range > 50',ranges>50))],
            'flat_groups':{'all_zero':int((flat&(rewards[:,0]<1e-9)).sum()),
                           'all_100':int((flat&(rewards[:,0]>100-1e-9)).sum()),
                           'other_positive':int((flat&(rewards[:,0]>=1e-9)&(rewards[:,0]<=100-1e-9)).sum())},
            'zero_advantage_trajectories':int((np.abs(advantages)<1e-12).sum()),
            'zero_advantage_percent':float(100*(np.abs(advantages)<1e-12).mean()),
            'fraction_range_above_10':float((ranges>10).mean()),
            'fraction_range_above_25':float((ranges>25).mean()),
            'reward_points':describe(rewards.ravel()),
            'descriptive_population_variance_decomposition':{
                'total':float(np.var(rewards)),
                'within_episode':float(np.var(rewards,axis=1).mean()),
                'between_episode_means':float(np.var(rewards.mean(1)))}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();groups=[];by_step=defaultdict(list);input_hashes={}
    committed=json.loads((args.run/'latest.json').read_text())['step']
    for path in sorted(args.run.glob('batch-*/scored.json')):
        step=int(path.parent.name.split('-')[-1])+1
        if step>committed or not (path.parent/'report.json').exists():continue
        by_id=defaultdict(list)
        data=path.read_bytes();input_hashes[str(path)]=hashlib.sha256(data).hexdigest()
        traces=json.loads(data);del data
        for trace in traces:by_id[trace['trajectory_id']//4].append(trace)
        for group_id,traces in sorted(by_id.items()):
            traces.sort(key=lambda t:t['trajectory_id'])
            if len(traces)!=4 or [t['trajectory_id']%4 for t in traces]!=list(range(4)):
                raise ValueError('incomplete four-rollout group')
            if any(t['spec']!=traces[0]['spec'] for t in traces):raise ValueError('mixed episodes')
            if any(t['steps'][0]['policy_version']!=step-1 for t in traces):raise ValueError('mixed policies')
            values=[100*t['reward'] for t in traces]
            if any(abs(v-t['reward_metrics']['pauc_f'])>1e-8 for v,t in zip(values,traces)):
                raise ValueError('reward differs from episode PAUC-F')
            for t in traces:
                expected=t['reward']-(sum(x['reward'] for x in traces)-t['reward'])/3
                if abs(t['steps'][0]['advantage']-expected)>1e-10:raise ValueError('RLOO mismatch')
            spec=traces[0]['spec']
            row={'step':step,'group':group_id,'span_id':spec['span_id'],
                 'episode_seconds':spec['rollout_frames'],'gold_events':len(spec['gold_points']),
                 'rewards_pauc_f_points':values,'range_points':float(np.ptp(values)),
                 'sample_variance_points_squared':float(np.var(values,ddof=1)),
                 'sample_std_points':float(np.std(values,ddof=1))}
            groups.append(row);by_step[step].append(row)
        del traces
    strata=defaultdict(list)
    for row in groups:
        n=row['gold_events'];strata['gold_events_'+('1' if n==1 else '2_to_4' if n<=4 else '5_plus')].append(row)
        n=row['episode_seconds'];strata['length_'+('up_to_32' if n<=32 else '33_to_96' if n<=96 else '97_plus')].append(row)
    result={'scope':'committed episode-rloo-v2 training batches only; four sampled trajectories of the same episode under one policy version',
            'units':'PAUC-F points, 0 to 100; actual reward is points / 100',
            'variance_definition':'sample variance across four trajectories, denominator 3; not variance across different episodes',
            'input_sha256':input_hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'overall':summarize(groups),
            'by_step':{str(k):summarize(v) for k,v in by_step.items()},
            'strata':{k:summarize(v) for k,v in strata.items()},
            'limits':'Four samples per episode. Different episodes appear in different updates; update-level trends are not paired learning curves.'}
    (args.run/'reward-variance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.run/'reward-variance-groups.csv').open('w') as handle:
        fields=[k for k in groups[0] if k!='rewards_pauc_f_points']+['reward_1','reward_2','reward_3','reward_4']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for group in groups:
            writer.writerow({**{k:v for k,v in group.items() if k!='rewards_pauc_f_points'},
                             **{f'reward_{i+1}':v for i,v in enumerate(group['rewards_pauc_f_points'])}})
    print(json.dumps({'overall':result['overall'],'by_step':{k:{f:v[f] for f in ('groups','range_points','sample_std_points','zero_advantage_percent')} for k,v in result['by_step'].items() if k in ('1','8','16','23')},
                      'strata':{k:{f:v[f] for f in ('groups','range_points','sample_std_points')} for k,v in result['strata'].items()}}),flush=True)


if __name__=='__main__':main()
