#!/usr/bin/env python3
"""Historical calibration analysis and shared raw-score diagnostic statistics.

Nonzero-margin calculations describe archived experiments, not current inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-mixture-12h/scaling-diagnostic'
ACTIONS=('Silence','Hold','Response')


def average_precision(target,score):
    target=np.asarray(target,dtype=bool);score=np.asarray(score,dtype=float)
    if not target.sum():return None
    order=np.argsort(-score,kind='stable');s=score[order];y=target[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends];recall=tp/target.sum();precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def chosen_action(logits,margin):
    values=np.asarray(logits,dtype=float).copy();values[2]-=margin
    return ACTIONS[int(np.argmax(values))]


def bootstrap_delta(left,right,samples=1000):
    ids=sorted(left)
    if set(ids)!=set(right):raise ValueError('paired video IDs differ')
    def array(rows):
        return np.array([[rows[i]['total_physical_fires'],rows[i]['value_adding_fires'],
            rows[i]['num_gold_windows'],rows[i]['pauc_recall_component']/100*rows[i]['num_gold_windows']] for i in ids])
    a,b=array(left),array(right)
    rng=np.random.default_rng(42);indexes=rng.integers(0,len(ids),(samples,len(ids)))
    def metric(v):
        fires,useful,events,auc=np.moveaxis(v,-1,0)
        precision=np.divide(useful,fires,out=np.zeros_like(useful),where=fires>0)
        recall=np.divide(auc,events,out=np.zeros_like(auc),where=events>0)
        return np.divide(200*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    delta=metric(b[indexes].sum(axis=1))-metric(a[indexes].sum(axis=1))
    return dict(delta_final_minus_earlier=float(metric(b.sum(axis=0))-metric(a.sum(axis=0))),
        ci95=np.quantile(delta,[.025,.975]).tolist(),replicates=samples,seed=42,
        uncertainty='Video sampling only; excludes seed and judge repeatability.')


def read_predictions(directory):
    rows=[]
    for path in sorted(directory.glob('predictions.rank*.jsonl')):
        rows.extend(json.loads(l) for l in path.read_text().splitlines())
    if len(rows)!=32 or len({r['question_id'] for r in rows})!=32:raise ValueError('incomplete or duplicate cohort')
    return rows


def behavior(condition):
    rows=read_predictions(Path(condition['output']))
    manifests={r['question_id']:r for r in map(json.loads,Path(condition['manifest']).read_text().splitlines())}
    counter=Counter();repeated=0;short_gap=0;post=Counter();cases=[]
    for row in rows:
        trace=row['jevany_action_trace'];times=manifests[row['question_id']]['inference_timestamps']
        if len(trace)!=len(times):raise ValueError('trace/time grid mismatch')
        selected=[chosen_action(t['logits'],condition['margin']) for t in trace]
        if Counter(selected)!=Counter(row['jevany_actions']):raise ValueError('calibrated trace/action counts differ')
        responses=row['model_response_list']
        if selected.count('Response')!=len(responses):raise ValueError('fire trace mismatch')
        seen=set();last_fire=None;index=0
        for action,t in zip(selected,times):
            counter[action]+=1
            if action!='Response':continue
            answer=' '.join(responses[index]['content'].lower().split());index+=1
            if answer in seen:repeated+=1
            seen.add(answer)
            gap=None if last_fire is None else float(t)-last_fire
            post['first_fire' if gap is None else 'gap_0_4s' if gap<=4 else 'gap_4_16s' if gap<=16 else 'gap_over_16s']+=1
            if gap is not None and gap<=4:short_gap+=1
            last_fire=float(t)
        normalized=[' '.join(r['content'].lower().split()) for r in responses]
        duplicates=len(normalized)-len(set(normalized))
        if duplicates:
            cases.append(dict(question_id=row['question_id'],question=manifests[row['question_id']]['question'],
                identical_repeats=duplicates,responses=responses))
    return dict(actions=dict(counter),action_rates={k:v/sum(counter.values()) for k,v in counter.items()},
        identical_repeats=repeated,fires_within_4s_of_previous=short_gap,
        response_gap_counts=dict(post),repeat_cases=sorted(cases,key=lambda r:-r['identical_repeats'])[:8])


def gold_history():
    source=[json.loads(l) for l in (OUT/'data/gold_history.jsonl').read_text().splitlines()]
    result={};arrays={};response_scores={}
    for step in [1332,1776]:
        records=[]
        for path in sorted((OUT/'gold_history').glob(f'step-{step:06d}.rank*.jsonl')):
            records.extend(json.loads(l) for l in path.read_text().splitlines())
        records.sort(key=lambda r:r['index'])
        if [r['index'] for r in records]!=list(range(len(source))):raise ValueError('gold-history coverage mismatch')
        for row,raw in zip(records,source):
            if (row['video_id'],row['frame_index'],row['label'])!=(raw['video_id'],raw['frame_index'],raw['label']):
                raise ValueError('gold-history identity changed')
        logits=np.array([r['logits'] for r in records]);target=np.array([r['label_index'] for r in records])
        exp=np.exp(logits-logits.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
        losses=np.array([r['nll'] for r in records]);pred=logits.argmax(axis=1)
        per_class={}
        for k,name in enumerate(ACTIONS):
            mask=target==k;predmask=pred==k
            per_class[name]=dict(n=int(mask.sum()),nll=float(losses[mask].mean()),
                recall=float(np.mean(pred[mask]==k)),precision=float(np.sum(mask&predmask)/max(1,predmask.sum())),
                mean_predicted_probability=float(prob[:,k].mean()))
        confusion=np.zeros((3,3),dtype=int)
        for y,p in zip(target,pred):confusion[y,p]+=1
        confidence=prob.max(axis=1);ece=0.
        for low in np.linspace(0,1,16)[:-1]:
            mask=(confidence>=low)&(confidence<low+1/15+1e-12)
            if mask.any():ece+=mask.mean()*abs(np.mean(pred[mask]==target[mask])-confidence[mask].mean())
        groups={}
        for name,mask in {
            'gold_prior_response_in_window':np.array([any(t['state']=='Response' for t in r['turns'][:-1]) for r in source]),
            'gold_memory_present':np.array([bool(r.get('memory')) for r in source]),
            'gold_memory_absent':np.array([not r.get('memory') for r in source]),
            'context_1_8_frames':np.array([len(r['turns'])<=8 for r in source]),
            'context_9_16_frames':np.array([9<=len(r['turns'])<=16 for r in source]),
            'context_17_32_frames':np.array([len(r['turns'])>=17 for r in source]),
        }.items():
            groups[name]=dict(n=int(mask.sum()),nll=float(losses[mask].mean()),accuracy=float(np.mean(pred[mask]==target[mask])))
        result[str(step)]=dict(rows=len(source),nll=float(losses.mean()),accuracy=float(np.mean(pred==target)),
            response_average_precision=average_precision(target==2,prob[:,2]),
            response_margin_average_precision=average_precision(target==2,logits[:,2]-np.max(logits[:,:2],axis=1)),
            ece_15_bins=float(ece),per_class=per_class,confusion_true_by_predicted=confusion.tolist(),groups=groups)
        arrays[step]=losses
        response_scores[step]=prob[:,2]
    ids=sorted({r['video_id'] for r in source});components=[]
    for video in ids:
        ix=np.array([i for i,r in enumerate(source) if r['video_id']==video])
        components.append([len(ix),arrays[1332][ix].sum(),arrays[1776][ix].sum()])
    values=np.asarray(components);rng=np.random.default_rng(42);totals=values[rng.integers(0,len(ids),(1000,len(ids)))].sum(axis=1)
    delta=(totals[:,2]-totals[:,1])/totals[:,0]
    result['paired_nll_delta']=dict(final_minus_earlier=result['1776']['nll']-result['1332']['nll'],
        ci95=np.quantile(delta,[.025,.975]).tolist(),replicates=1000,seed=42)
    by_video=[np.array([i for i,r in enumerate(source) if r['video_id']==video]) for video in ids]
    truth=np.array([r['label']=='Response' for r in source])
    ap_deltas=[]
    rng=np.random.default_rng(42)
    for draw in rng.integers(0,len(ids),(1000,len(ids))):
        indexes=np.concatenate([by_video[i] for i in draw])
        left=average_precision(truth[indexes],response_scores[1332][indexes])
        right=average_precision(truth[indexes],response_scores[1776][indexes])
        if left is not None and right is not None:ap_deltas.append(right-left)
    result['paired_response_ap_delta']=dict(final_minus_earlier=result['1776']['response_average_precision']-result['1332']['response_average_precision'],
        ci95=np.quantile(ap_deltas,[.025,.975]).tolist(),replicates=len(ap_deltas),seed=42)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');args=parser.parse_args()
    report=OUT/'reports';report.mkdir(exist_ok=True)
    development=json.loads((OUT/'development_scores.json').read_text())
    conditions=[]
    for row in development:
        conditions.append(dict(name=row['name'],step=row['step'],margin=row['margin'],score=row['score'],behavior=behavior(row)))
    summary=dict(plan=json.loads((OUT/'plan.json').read_text()),semantic_judging_complete=not args.local_only,
        development=conditions,gold_history=gold_history(),limitations=[
            'Native gold-history sampling is matched between models but is not a population-distribution estimate.',
            'The 32-video cohorts are small; margins are selected on development, not WEB.',
            'Timing-only scores assume all in-window answers are correct and do not establish semantic usefulness.',
            'Comparisons diagnose checkpoint behavior, not seed variance or a causal mechanism.'])
    if not args.local_only:
        confirmation=json.loads((OUT/'confirmation_scores.json').read_text())
        summary['confirmation']=[dict(name=r['name'],step=r['step'],margin=r['margin'],score=r['score'],behavior=behavior(r)) for r in confirmation]
        selected=json.loads((OUT/'selected_margins.json').read_text());parts={}
        for row in confirmation:
            best=next(r for r in selected if r['step']==row['step'])
            if row['margin']==best['margin']:
                per=json.loads((Path(row['output'])/'official/results/exp10_per_video.json').read_text())
                parts[row['step']]={r['question_id']:r for r in per}
        summary['selected_margin_confirmation_paired_delta']=bootstrap_delta(parts[1332],parts[1776])
    (report/'diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Qwen3.5-2B checkpoint and calibration diagnosis','',
        '**Local timing-only analysis; semantic judging awaits approval.**' if args.local_only else '**Development-selected margins evaluated on separate confirmation videos.**','',
        '| Checkpoint | Margin | Development PAUC-F@4s | Precision | Recall | Fires | Identical repeats |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in conditions:
        s=row['score'];b=row['behavior']
        lines.append(f"| {row['step']} | {row['margin']:+g} | {s['pauc_f']:.2f} | {s['fire_precision']:.2f} | {s['pauc_recall_component']:.2f} | {s['total_physical_fires']} | {b['identical_repeats']} |")
    lines+=['','## Matched gold-history inputs','', '| Checkpoint | NLL | Accuracy | Response AP |','| --- | ---: | ---: | ---: |']
    for step in ['1332','1776']:
        s=summary['gold_history'][step];lines.append(f"| {step} | {s['nll']:.4f} | {s['accuracy']:.4f} | {s['response_average_precision']:.4f} |")
    lines+=['','Per-class losses, paired video intervals, history groups and repeated-fire examples are in diagnostic.json.','',*summary['limitations']]
    (report/'diagnostic.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(json.dumps({'gold_history':summary['gold_history'],'semantic_judging_complete':not args.local_only,'plot':'omitted; plotting package unavailable'}),flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for step in [1332,1776]:
        values=sorted([r for r in conditions if r['step']==step],key=lambda r:r['score']['pauc_recall_component'])
        ax.plot([r['score']['pauc_recall_component'] for r in values],[r['score']['fire_precision'] for r in values],'o-',label=f'Step {step}')
        for r in values:ax.annotate(f"m={r['margin']:+g}",(r['score']['pauc_recall_component'],r['score']['fire_precision']),xytext=(3,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='PAUC recall component (%)',ylabel='Fire precision (%)',title='Development: timing-only proxy' if args.local_only else 'Development: GPT-4.1 judged PAUC-F@4s',xlim=(0,100),ylim=(0,100))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(report/'development_frontier.png',dpi=180);plt.close(fig)
    print(json.dumps({'gold_history':summary['gold_history'],'semantic_judging_complete':not args.local_only}),flush=True)

if __name__=='__main__':main()
