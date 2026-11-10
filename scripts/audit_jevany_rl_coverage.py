#!/usr/bin/env python3
"""Diagnose saved training trajectories; never generate or judge new WEB data."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

from jevany_rl_reward import ROOT, write, sha
from pace_clm.rl import leave_one_out
from zs_infer.pauc_f import assign_value_adding_fires


def audit(run):
    strata=defaultdict(Counter)
    per_round=[]
    inputs={}
    for path in sorted(run.glob('batch-*/scored.json')):
        inputs[str(path)]=sha(path)
        traces=json.loads(path.read_text())
        scores_path=path.parent/'reward/scores.json'
        inputs[str(scores_path)]=sha(scores_path)
        scores={x['question_id']:x for x in json.loads(scores_path.read_text())}
        groups=defaultdict(list)
        for trace in traces:groups[trace['spec']['span_id']].append(trace)
        counts=Counter()
        for members in groups.values():
            if len(members)!=4:raise ValueError('incomplete four-rollout group')
            kind='event' if members[0]['spec']['gold_points'] else 'none'
            s=strata[kind]
            rewards=[x['reward'] for x in members]
            advantages=leave_one_out(rewards)  # Same scale for both optimizer audits.
            s['groups']+=1
            s['equal_reward_groups']+=max(rewards)==min(rewards)
            counts['groups']+=1;counts['ties']+=max(rewards)==min(rewards)
            covered=[]
            for trace,adv in zip(members,advantages,strict=True):
                qid=f"{trace['spec']['span_id']}:{trace['trajectory_id']}"
                fires=[x['row']['frame_index'] for x in trace['steps'] if x['action']==2]
                useful=assign_value_adding_fires(fires,scores[qid]['answer'])
                useful_times={float(fires[i]) for i in useful}
                event_hits=set()
                consumed=set()
                ordered=sorted(enumerate(scores[qid]['answer']),key=lambda x:(*x[1]['reply_timespan'],x[0]))
                for i,turn in ordered:
                    available=[j for j in range(len(fires)) if j not in consumed]
                    matched=assign_value_adding_fires([fires[j] for j in available],[turn])
                    if matched:
                        event_hits.add(i)
                        consumed.update(available[j] for j in matched)
                if consumed!=useful or len(useful)!=trace['reward_metrics']['num_value_adding_fires']:
                    raise ValueError('audit useful-fire attribution differs from official scorer')
                covered.append(event_hits)
                s['trajectories']+=1
                s['useful_responses']+=len(useful_times)
                s['useful_response_positive_advantage']+=len(useful_times)*(adv>0)
                s['useful_response_negative_advantage']+=len(useful_times)*(adv<0)
                s['useful_response_zero_advantage']+=len(useful_times)*(adv==0)
                for step in trace['steps']:
                    response=step['action']==2
                    # Descent direction on each observation's response logit.
                    # This is a score-function diagnostic, NOT parameter-gradient attribution.
                    direction=adv*(int(response)-math.exp(step['old_logp'][2]))
                    s['response_logit_descent_sum']+=direction
                    s['response_logit_descent_absolute_sum']+=abs(direction)
                    s['actions']+=1;s['fires']+=response
            if kind=='event':
                union=set().union(*covered)
                s['gold_events']+=len(members[0]['spec']['gold_points'])
                s['gold_events_missed_by_all_four']+=len(members[0]['spec']['gold_points'])-len(union)
                for a,b in combinations(members,2):
                    ua=a['reward_metrics']['num_value_adding_fires'];ub=b['reward_metrics']['num_value_adding_fires']
                    if ua==ub:continue
                    s['pairs_with_different_useful_count']+=1
                    delta=(a['reward']-b['reward'])*(ua-ub)
                    s['more_useful_lower_reward']+=delta<0
                    s['more_useful_higher_reward']+=delta>0
        per_round.append({'round':int(path.parent.name.split('-')[-1])+1,**counts})
    return {'strata':dict(strata),'rounds':per_round,'input_sha256':inputs,
            'limitations':['Saved-policy observations only; no counterfactual answers for unsampled responses.',
                'Both runs analyzed with unnormalized leave-one-out advantages for comparable reward-signal diagnostics.',
                'Response-logit directions do not measure shared parameter gradients or establish causality.',
                'Coverage diagnostic follows official earliest-positive, distinct-fire assignment, including overlapping windows.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    result={name:audit(ROOT/'outputs/jevany-2b-rl'/name) for name in
            ('reward-aligned-rloo-v1','reward-aligned-grpo-v2')}
    write(a.output/'audit.json',result)
    lines=['# Training-only coverage audit','',
           'No new trajectories or judgments. Both runs use common RLOO-scale diagnostic advantages.','']
    for name,data in result.items():
        lines.extend([f'## {name}',''])
        for kind,s in data['strata'].items():
            lines.append(f"- {kind}: {s['groups']} groups; {s['equal_reward_groups']} tied; {s['fires']} fires; "
                         f"{s['useful_responses']} useful; positive / negative / zero advantage useful responses: "
                         f"{s['useful_response_positive_advantage']} / {s['useful_response_negative_advantage']} / {s['useful_response_zero_advantage']}.")
            if kind=='event':
                lines.append(f"- Events missed by all four: {s['gold_events_missed_by_all_four']} / {s['gold_events']}; "
                             f"more-useful lower-reward pairs: {s['more_useful_lower_reward']} / {s['pairs_with_different_useful_count']}.")
        lines.append('')
    lines.extend(['Interpretation requires comparing event and no-event signal magnitudes; do not equate logit diagnostics with optimizer gradients.',''])
    (a.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({name:data['strata'] for name,data in result.items()}))


if __name__=='__main__':main()
