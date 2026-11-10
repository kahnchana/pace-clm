#!/usr/bin/env python3
"""Local observer: enrich completed mixture checkpoints and compare available A/B runs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/jevany-qwen35-2b-mixture-12h'
OUT=BASE/'representative-seed42'


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def bootstrap(left,right):
    from zs_infer.pauc_f import OFFICIAL_SCORER,_load_official_scorer,assign_value_adding_fires,fire_times
    official=_load_official_scorer(OFFICIAL_SCORER)
    def components(web):
        scores={r['question_id']:r for r in json.loads((web/'pauc_llm_scores-gpt-4.1.json').read_text())}
        result={}
        for line in (web/'predictions.jsonl').read_text().splitlines():
            row=json.loads(line); turns=scores[row['question_id']]['answer']; fires=fire_times(row)
            auc=0.
            for turn in turns:
                judged=turn.get('judge_scores') or {}
                if judged:
                    points=[[float(t)-turn['reply_timespan'][0],v] for t,v in judged.items()]
                    auc+=official.area_under_line_ratio(points,turn['reply_timespan'][1]-turn['reply_timespan'][0],2,.5,.5)
                else:
                    auc+=.25
            result[row['question_id']]=(len(fires),len(assign_value_adding_fires(fires,turns)),auc,len(turns))
        return result
    a,b=components(left),components(right)
    ids=sorted(a)
    if len(ids)!=500 or set(a)!=set(b): raise ValueError('paired bootstrap requires identical 500 IDs')
    def metric(values,selection):
        total=np.array([values[i] for i in selection]).sum(axis=0)
        precision=total[1]/total[0] if total[0] else 0.
        recall=total[2]/total[3]
        return 200*precision*recall/(precision+recall) if precision+recall else 0.
    rng=random.Random(42)
    deltas=[]
    for _ in range(1000):
        sample=[ids[rng.randrange(500)] for _ in range(500)]
        deltas.append(metric(b,sample)-metric(a,sample))
    return dict(replicates=1000,seed=42,delta_b_minus_a=metric(b,ids)-metric(a,ids),
        delta_ci95=np.quantile(deltas,[.025,.975]).tolist(),
        uncertainty='Paired video uncertainty; excludes seed variance and judge repeatability.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--watch',action='store_true'); args=p.parse_args()
    order=np.load(BASE/'plans/representative.npy',mmap_mode='r')
    planned_labels=np.load(BASE/'plans/labels.npy',mmap_mode='r')
    audit=json.loads((BASE/'plans/audit.json').read_text())
    source=Path(audit['source']).open('rb')
    consumed=0; videos=set(); positions=set(); contexts=Counter(); memories=Counter(); sources=Counter()
    processed=set(); enriched=[]
    deadline=time.monotonic()+24*3600
    while True:
        path=OUT/'reports/results.json'
        if path.exists():
            results=json.loads(path.read_text())
            for checkpoint in results['checkpoints']:
                phase=checkpoint['phase']
                if phase in processed: continue
                count=checkpoint['presentations']
                for index in range(consumed,count):
                    offset,size=map(int,order[index]); source.seek(offset); row=json.loads(source.read(size))
                    video=row['video_id']; videos.add(video); positions.add((video,row['question'],row['frame_index']))
                    contexts[len(row['turns'])]+=1; memories[str(bool(row.get('memory')))]+=1
                    sources[str(Path(row['video_path']).parent.parent)]+=1
                consumed=count
                profile=dict(unique_videos=len(videos),unique_positions=len(positions),
                    context_frames=dict(contexts),memory_rows=dict(memories),sources=dict(sources),
                    presentations_per_second=count/(checkpoint['actual_training_hours']*3600))
                label_names=('Silence','Response','Hold')
                expected=dict(zip(label_names,np.bincount(planned_labels[:count],minlength=3).tolist()))
                if expected != checkpoint['training_labels']:
                    raise ValueError('checkpoint labels differ from frozen mixture prefix')
                profile['recent_4096_labels']=dict(zip(label_names,
                    np.bincount(planned_labels[max(0,count-4096):count],minlength=3).tolist()))
                log=OUT/f'logs/train_phase{phase}.log'
                measurements=[]
                for line in log.read_text().splitlines():
                    try: value=json.loads(line)
                    except ValueError: continue
                    if 'rank0_token_padding_fraction' in value: measurements.append(value)
                if measurements:
                    profile['rank0_mean_token_padding_fraction']=float(np.mean([v['rank0_token_padding_fraction'] for v in measurements]))
                    hits=[v['rank0_frame_cache_hit_fraction'] for v in measurements if 'rank0_frame_cache_hit_fraction' in v]
                    if hits: profile['rank0_mean_frame_cache_hit_fraction']=float(np.mean(hits))
                checkpoint['training_exposure']=profile
                control=BASE/'control-seed42/report.json'
                if control.exists():
                    a=json.loads(control.read_text())['checkpoints']
                    match=next((v for v in a if v.get('phase')==phase),None)
                    if match:
                        # Locate its immutable WEB directory from the control report/step.
                        step=match.get('target_step',match.get('step'))
                        candidates=list((BASE/'control-seed42/evaluation').glob(f'web_step{int(step):06d}')) if step is not None else []
                        if candidates:
                            checkpoint['paired_bootstrap_vs_control']=bootstrap(candidates[0],OUT/f'evaluation/web_phase{phase}')
                enriched.append(checkpoint); processed.add(phase)
                dump(OUT/'reports/detailed_results.json',dict(checkpoints=enriched,data_audit=audit,
                    step_budget_approximation=True,backend='gloo',host='ip-10-3-70-177'))
                print(json.dumps({'profiled_phase':phase,'presentations':count,'unique_videos':len(videos)}),flush=True)
        status=json.loads((OUT/'status.json').read_text())
        if status['phase'] in ('failed','complete') or not args.watch: break
        if time.monotonic()>deadline: raise TimeoutError('experiment observer exceeded 24 hours')
        time.sleep(30)

if __name__=='__main__': main()
