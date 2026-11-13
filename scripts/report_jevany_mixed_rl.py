#!/usr/bin/env python3
"""Summarize immutable evaluation results and training reward distributions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from jevany_rl_reward import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    records=[];training=[]
    for path in sorted((run/'evaluation').glob('*/result.json')):
        result=json.loads(path.read_text())
        for name,summary in result.get('cohorts',{}).items():
            records.append({'evaluation':result['job']['name'],'kind':result['job']['kind'],
                'step':result['job']['step'],'decisions':result['job']['exposure'],'cohort':name,**summary})
    for path in sorted(run.glob('batch-*/report.json')):
        report=json.loads(path.read_text());traces=json.loads((path.parent/'scored.json').read_text())
        groups={}
        for t in traces:groups.setdefault(t['spec']['span_id'],[]).append(t['reward'])
        for domain in ('ours','web','ego'):
            subset=[t for t in traces if t['spec']['domain']==domain]
            rewards=[t['reward'] for t in subset]
            spreads=[max(groups[t['spec']['span_id']])-min(groups[t['spec']['span_id']]) for t in subset[::4]]
            training.append({'step':report['step'],'domain':domain,'trajectories':len(subset),
                'episodes':len({t['spec']['span_id'] for t in subset}),
                'reward_mean':float(np.mean(rewards)),'reward_std':float(np.std(rewards)),
                'best_of_four_pauc_f_quantiles':np.quantile([max(v)*100 for q,v in groups.items()
                    if q in {t['spec']['span_id'] for t in subset}],[0,.25,.5,.75,1]).tolist(),
                'within4_range_quantiles':np.quantile(spreads,[0,.25,.5,.75,1]).tolist(),
                'flat_group_fraction':float(np.mean(np.array(spreads)==0)),
                'cumulative_domain_decisions':report['episode_schedule']['cumulative_domain_decisions'][domain],
                'decisions_per_second':report['decisions']/report['seconds'],'gpu_hours':report['gpu_hours']})
    write(run/'report.json',{'evaluations':records,'training':training,
        'sampled_identities':json.loads((run/'data/sampled-identities.json').read_text()) if (run/'data/sampled-identities.json').exists() else None,
        'interpretation':'Train cohorts are seen. Heldout cohorts are excluded from this RL run, but reused historical benchmarks. Mean episode reward differs from global PAUC-F.'})
    if records:
        with (run/'evaluation-summary.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Mixed WEB/EGO episode RLOO results','',
        'WEB/EGO training cohorts intentionally enter RL. Heldout cohorts are excluded from this run; historical benchmark reuse remains.',
        '', '| Evaluation | Decisions | Cohort | PAUC-F | Useful | Wasted |','|---|---:|---|---:|---:|---:|']
    lines += [f"| {r['evaluation']} | {r['decisions']} | {r['cohort']} | {r['pauc_f']:.3f} | {r['useful']} | {r['wasted']} |" for r in records]
    (run/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
