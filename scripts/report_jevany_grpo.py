#!/usr/bin/env python3
"""GRPO data exposure, two-pass replay cost, clipping diagnostics and dev selection."""
import argparse
import csv
import json
from pathlib import Path

from jevany_rl_reward import write
from report_jevany_e2 import paired_bootstrap
from run_jevany_e2_full_web import BASELINE
from run_qwen35_control import verify_web


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    curves=[]
    for path in sorted(run.glob('batch-*/report.json')):
        r=read(path)
        learners=[v for update in r['updates'] for v in update['learner'].values()]
        count=sum(v['decisions'] for v in learners)
        curves.append({'round':r['step'],'optimizer_steps':r['optimizer_steps'],
            'sampled_decisions':r['decisions'],'actor_replayed_decisions':count,
            'parity_verified_decisions':r['parity_verified_decisions'],
            'seconds':r['seconds'],'end_to_end_seconds':r['end_to_end_seconds'],'gpu_hours':r['gpu_hours'],
            'mean_reward':r['mean_reward'],'fires':r['fires'],
            'sft_kl':sum(v['kl_sum'] for v in learners)/count,
            'entropy':sum(v['entropy_sum'] for v in learners)/count,
            'ratio_mean':sum(v['ratio_sum'] for v in learners)/count,
            'ratio_min':min(v['ratio_min'] for v in learners),
            'ratio_max':max(v['ratio_max'] for v in learners),
            'outside_clip_fraction':sum(v['outside_clip_count'] for v in learners)/count,
            'effective_clip_fraction':sum(v['surrogate_clip_count'] for v in learners)/count,
            'max_grad_norm':max(v['grad_norm'] for v in learners)})
    write(run/'learning-curves.json',curves)
    if curves:
        with (run/'learning-curves.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(curves[0]))
            writer.writeheader();writer.writerows(curves)
    baseline=read(run/'development/sft-baseline.json')
    evaluations=[]
    for step in (8,16,24,32):
        p=run/f'evaluation/dev-step{step}/pauc_f_summary-gpt-4.1-v1.json'
        if p.exists():evaluations.append({'round':step,**read(p)})
    summary={'workflow':read(run/'workflow-status.json'),'completed_rounds':len(curves),
        'optimizer_steps':4*len(curves),'sampled_decisions':4096*len(curves),
        'actor_replayed_decisions':8192*len(curves),'baseline':baseline,'development':evaluations,
        'core_node_hours':sum(x['seconds'] for x in curves)/3600,
        'core_gpu_hours':sum(x['gpu_hours'] for x in curves),
        'limits':'One seed; reused WEB development; normalized/clipped/two-pass package comparison. RLOO stopped at 16; exposure-matched primary comparisons are at 8/16.'}
    full_results=[]
    full_baseline,_,_=verify_web(BASELINE)
    for step in (8,16,24,32):
        output=run/f'evaluation/web-step{step}'
        if not (output/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
        result,_,_=verify_web(output)
        bootstrap_file=output/'paired-bootstrap.json'
        bootstrap=read(bootstrap_file) if bootstrap_file.exists() else paired_bootstrap(output,BASELINE)
        write(bootstrap_file,bootstrap)
        full_results.append({'round':step,'score':result,'paired_bootstrap':bootstrap})
    summary['full_web']=full_results
    summary['full_web_baseline']=full_baseline
    write(run/'result.json',summary)
    lines=['# E2-G: clipped GRPO with frozen GPT/PAUC-F-aligned reward','',
        f"Completed rounds: {len(curves)}; optimizer steps: {4*len(curves)}; sampled / actor-replayed decisions: {4096*len(curves):,} / {8192*len(curves):,}.",
        f"Core training: {summary['core_node_hours']:.2f} node-hours / {summary['core_gpu_hours']:.2f} allocated GPU-hours; startup, controlled restore gates and evaluation are additional.",'',
        'Fresh original SFT, seven learners / one frozen generator, same persisted spans, GPT-4.1 reward and development cohort as E2-R. Two passes with two global 64-trajectory minibatches per pass; four optimizer steps per round. Fixed behavior likelihoods differ from the frozen SFT KL reference.','',
        '| Development checkpoint | PAUC-F | Precision | Recall | Useful / wasted fires |',
        '|---|---:|---:|---:|---:|',
        f"| SFT | {baseline['pauc_f']:.4f} | {baseline['fire_precision']:.2f} | {baseline['pauc_recall_component']:.2f} | {baseline['num_value_adding_fires']} / {baseline['num_wasted_fires']} |"]
    for score in evaluations:
        lines.append(f"| GRPO round {score['round']} | {score['pauc_f']:.4f} | {score['fire_precision']:.2f} | {score['pauc_recall_component']:.2f} | {score['num_value_adding_fires']} / {score['num_wasted_fires']} |")
        ref=run.parent/'reward-aligned-rloo-v1'/f"evaluation/dev-step{score['round']}/pauc_f_summary-gpt-4.1-v1.json"
        if ref.exists():
            control=read(ref)
            lines.append(f"| RLOO round {score['round']} (matched collected exposure) | {control['pauc_f']:.4f} | {control['fire_precision']:.2f} | {control['pauc_recall_component']:.2f} | {control['num_value_adding_fires']} / {control['num_wasted_fires']} |")
    if full_results:
        lines += ['', '| Full 500-video WEB | PAUC-F | Useful / wasted fires | Delta vs SFT |',
                  '|---|---:|---:|---:|',
                  f"| SFT retained control | {full_baseline['pauc_f']:.4f} | {full_baseline['num_value_adding_fires']} / {full_baseline['num_wasted_fires']} | — |"]
        for item in full_results:
            score=item['score']
            lines.append(f"| GRPO round {item['round']} | {score['pauc_f']:.4f} | {score['num_value_adding_fires']} / {score['num_wasted_fires']} | {item['paired_bootstrap']['difference_points']:+.4f} |")
    lines += ['',summary['limits'],'',f"Workflow: {summary['workflow']['state']}. No automatic extension beyond 32 collection rounds."]
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
