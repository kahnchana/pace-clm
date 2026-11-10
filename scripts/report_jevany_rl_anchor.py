#!/usr/bin/env python3
"""Compare supervised-anchor RLOO against SFT and the prior unanchored RLOO."""
import argparse
import csv
import json
from pathlib import Path

from jevany_rl_reward import ROOT, write
from report_jevany_e2 import paired_bootstrap
from run_jevany_e2_full_web import BASELINE
from run_qwen35_control import verify_web


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    baseline,_,_=verify_web(BASELINE)
    curves=[]
    for path in sorted(run.glob('batch-*/report.json')):
        r=read(path);learners=list(r['learner'].values());decisions=r['decisions']
        ce=sum(v['anchor_ce_global_contribution'] for v in learners)
        counts=[sum(v['anchor_class_counts'][k] for v in learners) for k in range(3)]
        correct=[sum(v['anchor_class_correct'][k] for v in learners) for k in range(3)]
        curves.append({'round':r['step'],'optimizer_steps':r['optimizer_steps'],'rl_decisions':decisions,
            'anchor_examples':sum(v['anchor_examples'] for v in learners),
            'seconds':r['seconds'],'gpu_hours':r['gpu_hours'],
            'anchor_seconds_max_rank':max(v['anchor_seconds'] for v in learners),
            'anchor_lambda':learners[0]['anchor_lambda'],'anchor_ce':ce,
            'anchor_silence_accuracy':correct[0]/max(1,counts[0]),
            'anchor_hold_accuracy':correct[1]/max(1,counts[1]),
            'anchor_response_accuracy':correct[2]/max(1,counts[2]),
            'mean_reward':r['mean_reward'],'fires':r['fires'],
            'entropy':sum(v['entropy_sum'] for v in learners)/decisions,
            'sft_kl':sum(v['kl_sum'] for v in learners)/decisions,
            'max_parity_probability_error':max(v['parity_probability_max'] for v in learners)})
    write(run/'learning-curves.json',curves)
    if curves:
        with (run/'learning-curves.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    evaluations=[]
    for step in (8,16,24,32):
        output=run/f'evaluation/web-step{step}'
        if not (output/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
        score,_,_=verify_web(output)
        bootfile=output/'paired-bootstrap.json'
        boot=read(bootfile) if bootfile.exists() else paired_bootstrap(output,BASELINE)
        write(bootfile,boot)
        coverage=score['num_value_adding_fires']/baseline['num_value_adding_fires']
        evaluations.append({'round':step,'score':score,'paired_bootstrap':boot,
            'useful_coverage_ratio':coverage,
            'point_improvement_with_coverage':score['pauc_f']>baseline['pauc_f'] and coverage>=.9,
            'positive_paired_95_ci_with_coverage':boot['percentile_95_ci'][0]>0 and coverage>=.9})
    result={'workflow':read(run/'workflow-status.json'),'completed_rounds':len(curves),
            'optimizer_steps':len(curves),'sampled_decisions':4096*len(curves),
            'supervised_replay_decisions':126*len(curves),'baseline':baseline,'full_web':evaluations,
            'anchor_calibration':read(run/'anchor-calibration.json'),
            'core_node_hours':sum(r['seconds'] for r in curves)/3600,
            'allocated_core_gpu_hours':sum(r['gpu_hours'] for r in curves),
            'limits':'One seed; reused WEB development, not untouched test. Auxiliary class-balanced supervised replay changes the training objective. No inference heuristics.'}
    write(run/'result.json',result)
    lines=['# E3: RLOO with fixed supervised replay','',
        f"Completed rounds / optimizer steps: {len(curves)} / {len(curves)}. Training-only anchor; raw learned argmax at evaluation.",'',
        '| Full 500-video WEB | PAUC-F | Delta vs SFT | Useful / wasted | Coverage |',
        '|---|---:|---:|---:|---:|',
        f"| SFT | {baseline['pauc_f']:.4f} | — | {baseline['num_value_adding_fires']} / {baseline['num_wasted_fires']} | 100% |"]
    intervals=[]
    for item in evaluations:
        s=item['score'];b=item['paired_bootstrap']
        lines.append(f"| Anchor RLOO round {item['round']} | {s['pauc_f']:.4f} | {b['difference_points']:+.4f} | {s['num_value_adding_fires']} / {s['num_wasted_fires']} | {item['useful_coverage_ratio']:.1%} |")
        prior=ROOT/f"outputs/jevany-2b-rl/reward-aligned-rloo-v1/full_web_v1/web-step{item['round']}/pauc_f_summary-gpt-4.1-v1.json"
        if prior.exists():
            r=read(prior)
            lines.append(f"| Prior RLOO round {item['round']} | {r['pauc_f']:.4f} | {r['pauc_f']-baseline['pauc_f']:+.4f} | {r['num_value_adding_fires']} / {r['num_wasted_fires']} | {r['num_value_adding_fires']/baseline['num_value_adding_fires']:.1%} |")
        intervals.append(f"Round {item['round']} delta 95% paired video interval: [{b['percentile_95_ci'][0]:+.4f}, {b['percentile_95_ci'][1]:+.4f}].")
    lines.extend(['',*intervals])
    lines.extend(['',result['limits'],'',f"Workflow: {result['workflow']['state']}."])
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
