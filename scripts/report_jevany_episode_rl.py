#!/usr/bin/env python3
"""Actual exposure/cost and official paired comparisons for episode RLOO."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    freeze=read(run/'workflow-freeze.json');baseline,_,_=verify_web(BASELINE)
    curves=[];exposure=0;gpu_hours=0.;episodes=0;replay=0
    for path in sorted(run.glob('batch-*/report.json')):
        r=read(path);learners=list(r['learner'].values())
        exposure+=r['decisions'];gpu_hours+=r['gpu_hours']
        episodes+=4*len(r['episode_schedule']['episode_indices'])
        replay+=sum(v['anchor_examples'] for v in learners)
        traces=read(path.parent/'scored.json')
        curves.append({'optimizer_step':r['step'],'sampled_decisions':exposure,
                       'sampled_episodes':episodes,'supervised_examples':replay,
                       'core_gpu_hours':gpu_hours,'batch_seconds':r['seconds'],
                       'mean_episode_reward':r['mean_reward'],
                       'fires':r['fires'],
                       'useful_fires':sum(t['reward_metrics']['num_value_adding_fires'] for t in traces),
                       'wasted_fires':sum(t['reward_metrics']['num_wasted_fires'] for t in traces),
                       'sft_kl_per_decision':sum(v['kl_sum'] for v in learners)/r['decisions'],
                       'entropy_per_decision':sum(v['entropy_sum'] for v in learners)/r['decisions'],
                       'max_replay_probability_error':max(v['parity_probability_max'] for v in learners),
                       'generator_wait_seconds':max(v['answer_wait_seconds'] for v in r['collection'].values()),
                       'learner_seconds':max(v['seconds'] for v in learners),
                       'gpt_seconds':r['judge']['seconds'],
                       'padding_ratio':sum(v['preparation']['physical_tokens'] for v in learners)/max(1,sum(v['preparation']['real_tokens'] for v in learners)),
                       'max_reserved_gib':max(v['reserved_gib'] for v in learners)})
    write(run/'learning-curves.json',curves)
    if curves:
        with (run/'learning-curves.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    evaluations=[]
    anchor=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1'
    for step in freeze['evaluation_steps']:
        directory=run/f'evaluation/web-step{step}'
        if not (directory/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
        score,_,_=verify_web(directory)
        bootpath=directory/'paired-bootstrap.json'
        boot=read(bootpath) if bootpath.exists() else paired_bootstrap(directory,BASELINE)
        write(bootpath,boot)
        subset=read(run/f'evaluation/dev-step{step}/pauc_f_summary-gpt-4.1-v1.json')
        comparisons={}
        current=freeze['episode_batches'][step-1]['exposure_after']
        nearest=min((8,16,24,32),key=lambda s:abs(s*4096-current))
        for name,source,old_step in (
                ('anchored_clip_rloo_24',anchor/'evaluation/web-step24',24),
                ('nearest_decision_exposure_anchor',anchor/f'evaluation/web-step{nearest}',nearest)):
            if not (source/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
            comparison=paired_bootstrap(directory,source)
            old_reports=[read(p) for p in sorted(anchor.glob('batch-*/report.json'))][:old_step]
            comparison.update(source=str(source),reference_sampled_decisions=old_step*4096,
                              candidate_sampled_decisions=current,exposure_difference=current-old_step*4096,
                              reference_core_gpu_hours=sum(r['gpu_hours'] for r in old_reports),
                              matching='nearest saved decision exposure; exact difference disclosed; not GPU-hour matched')
            comparisons[name]=comparison
        coverage=score['num_value_adding_fires']>=631 and subset['num_value_adding_fires']>=166
        evaluations.append({'step':step,'sampled_decisions':current,'score':score,'subset_score':subset,
                            'paired_vs_sft':boot,'comparisons':comparisons,
                            'point_gain_with_coverage':score['pauc_f']>54.67 and coverage,
                            'positive_paired_95_ci_with_coverage':boot['percentile_95_ci'][0]>0 and score['pauc_f']>54.67 and coverage})
    result={'workflow':read(run/'workflow-status.json'),'optimizer_steps':len(curves),
            'sampled_decisions':exposure,'sampled_episodes':episodes,'supervised_examples':replay,
            'prior_failed_gate':read(run/'prior-gate-exposure.json'),
            'total_collected_decisions_including_failed_gate':exposure+freeze['prior_failed_gate_collected_decisions'],
            'core_gpu_hours':gpu_hours,'baseline':baseline,'full_web':evaluations,
            'limits':'One seed; WEB is reused development. Episode mean differs from global PAUC-F. Costs exclude initialization, audit, evaluation and restarts. Nearest saved decision exposure is approximate; no GPU-hour-matched result is asserted.'}
    write(run/'result.json',result)
    lines=['# Episode-level RLOO pilot','',
           f"State: {result['workflow']['state']}. {len(curves)} optimizer updates; {exposure:,} sampled decisions; {episodes} trajectories; {gpu_hours:.2f} core GPU-hours.",'',
           '| Exposure | Global PAUC-F | Useful / wasted | Paired delta vs SFT (95% interval) |',
           '|---|---:|---:|---|']
    for item in evaluations:
        s=item['score'];b=item['paired_vs_sft'];lo,hi=b['percentile_95_ci']
        lines.append(f"| {item['sampled_decisions']} | {s['pauc_f']:.4f} | {s['num_value_adding_fires']} / {s['num_wasted_fires']} | {b['difference_points']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines.extend(['',result['limits'],'','Inference: raw learned argmax, response margin zero. No automatic budget extension.'])
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('full_web','baseline')}),flush=True)


if __name__=='__main__':main()
