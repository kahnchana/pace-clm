#!/usr/bin/env python3
"""Summarize saved evaluation evidence; never launches inference or judging."""
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path


def read(path):return json.loads(path.read_text())


def summarize(run):
    results=[]
    for hour in range(2,17,2):
        path=run/'evaluation'/f'hour-{hour:02d}/result.json'
        if not path.exists():continue
        result=read(path);cohort=read(run/'data/cohorts.json')['estp_it_explicit_dev']
        metrics=[result['per_episode'][qid] for qid in cohort]
        n=len(metrics)
        results.append(dict(hour=hour,step=result['job']['step'],episodes=n,
            zero_reward_episodes=sum(m['pauc_f']==0 for m in metrics),
            no_response_episodes=sum(m['num_fires']==0 for m in metrics),
            zero_useful_response_episodes=sum(m['num_value_adding_fires']==0 for m in metrics),
            zero_reward_percent=100*sum(m['pauc_f']==0 for m in metrics)/n,
            no_response_percent=100*sum(m['num_fires']==0 for m in metrics)/n,
            pooled_pauc_f=result['cohorts']['estp_it_explicit_dev']['pauc_f'],
            task_macro_pauc_f=result['estp_it_explicit_task_macro']))
    probes=[]
    for hour in [8,16]:
        path=run/'evaluation'/f'hour-{hour:02d}-readiness/result.json'
        if path.exists():
            p=read(path);probes.append(dict(hour=hour,groups=p['groups'],all_zero_groups=p['all_zero_groups'],flat_groups=p['flat_groups']))
    comparison={}
    final=run/'evaluation/hour-16/result.json'
    if final.exists():comparison=read(final).get('paired_vs_hour02',{})
    document=dict(utc=datetime.now(timezone.utc).isoformat(),completed_checkpoints=len(results),
        training_complete=(run/'training-complete.json').exists(),workflow_complete=(run/'complete.json').exists(),
        estp_it_greedy=results,sampled_readiness=probes,paired_final_vs_hour02=comparison,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    lines=['# SFT evidence for subsequent RL','',
        f"Completed quality evaluations: {len(results)} of 8. Training complete: {document['training_complete']}. Full workflow complete: {document['workflow_complete']}.",'',
        'These are saved evaluation results. This report performs no inference, GPT requests, or weight updates.','',
        '| Estimated hour | Step | IT episodes | IT zero reward | IT no response | IT pooled PAUC-F | IT task macro PAUC-F |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in results:
        lines.append(f"| {item['hour']} | {item['step']} | {item['episodes']} | {item['zero_reward_percent']:.2f}% | {item['no_response_percent']:.2f}% | {item['pooled_pauc_f']:.3f} | {item['task_macro_pauc_f']:.3f} |")
    lines+=['','Zero reward means an episode PAUC-F of zero. No response is a separate diagnostic; responding without a useful, timely answer can still earn zero. These full-cohort numbers use greedy controller decisions. They do not measure four-trajectory RL group variance.','',
        'The fixed sampled probes provide the separate group diagnostic:']
    for item in probes:
        lines.append(f"- Estimated hour {item['hour']}: {item['all_zero_groups']} of {item['groups']} groups all zero; {item['flat_groups']} tied. Four sampled trajectories per episode.")
    if not probes:lines.append('- Pending.')
    lines+=['','The sampled probe contains only eight fixed development episodes. Its group rates are diagnostic and do not establish population-wide RL readiness. IT quality uses the native-window development proxy, including unreachable windows in the full cohort; it is not ESTP-Bench or official ESTP-F1.','']
    if comparison:
        lines+=['| Final versus hour 2 | PAUC-F change | Paired 95% interval |','|---|---:|---|']
        for cohort,label in [('web_all','WEB'),('ego_all','EGO'),('estp_it_explicit_dev','IT Explicit development')]:
            value=comparison[cohort];low,high=value['paired_95_ci']
            lines.append(f"| {label} | {value['difference_points']:+.3f} | {low:+.3f} to {high:+.3f} |")
        lines+=['','Intervals condition on saved judgments and this evaluation population. Improvement during SFT does not prove that later RL or further training will improve every benchmark.']
    for name,value in [('readiness-summary.json',json.dumps(document,indent=2)+'\n'),('readiness-summary.md','\n'.join(lines)+'\n')]:
        destination=run/name;temporary=destination.with_suffix(destination.suffix+'.tmp');temporary.write_text(value);temporary.replace(destination)
    return document['workflow_complete']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    while not (run/'user-stopped.json').exists():
        if summarize(run) or not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
