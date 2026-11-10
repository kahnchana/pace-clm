#!/usr/bin/env python3
"""Retired margin-sweep preparation; historical implementation retained for audit."""
from __future__ import annotations

RETIREMENT_MESSAGE = 'Retired: nonzero response-margin sweeps are disabled by the raw learned-score policy. Historical code/artifacts are retained for audit only. Use scripts/eval_qwen35_development_sweep.py with zero-margin conditions for current evaluation.'

def main():
    """Reject the retired launcher before imports, file writes, GPU work or judging."""
    raise SystemExit(RETIREMENT_MESSAGE)

if __name__ == "__main__":
    main()

from collections import Counter
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
SG=ROOT.parent/'stream-gemma'
sys.path[:0]=[str(ROOT/'scripts'),str(SG/'src'),str(SG)]
from prepare_qwen35_control import aliases
from prepare_jevany_decisions import make_args

OUT=ROOT/'outputs/jevany-qwen35-2b-mixture-12h/scaling-diagnostic'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value,jsonl=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(v,sort_keys=True)+'\n' for v in value) if jsonl else json.dumps(value,indent=2,sort_keys=True)+'\n'
    if path.exists() and path.read_text()!=text: raise FileExistsError(path)
    path.write_text(text)

def choose(rows):
    chosen=[]
    for lo,hi,n in [(0,32,8),(33,96,16),(97,10000,8)]:
        candidates=[r for r in rows if lo<=r['inference_frame_count']<=hi]
        candidates.sort(key=lambda r:hashlib.sha256(('qwen35-scaling-diagnostic-42:'+r['question_id']).encode()).digest())
        if len(candidates)<n: raise ValueError('insufficient length stratum')
        chosen.extend(candidates[:n])
    return sorted(chosen,key=lambda r:hashlib.sha256(r['question_id'].encode()).digest())

def _historical_main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((OUT.parent/'plans/audit.json').read_text())
    cohorts={}; proof={}
    benchmark_ids=set()
    for b in ['web','ego']:
        for r in json.loads((SG/f'datasets/proactive-videoqa-{b}/anno.json').read_text()):
            benchmark_ids.update(aliases(r['video']))
    for source_name,dest in [('selection','development'),('final','confirmation')]:
        manifest=ROOT/f'outputs/pace-clm-mixtures/validation/{source_name}.jsonl'
        gold=manifest.with_name(source_name+'.gold.json')
        if audit['exclusion_files'][str(manifest)]!=sha(manifest): raise ValueError('training exclusion source changed')
        rows=choose([json.loads(l) for l in manifest.read_text().splitlines()])
        for r in rows:
            if aliases(r['video_id']) & benchmark_ids: raise ValueError('benchmark video overlap')
            if not Path(r['resolved_video_path']).is_file(): raise FileNotFoundError(r['resolved_video_path'])
        ids={r['question_id'] for r in rows}
        records=[r for r in json.loads(gold.read_text())['records'] if r['question_id'] in ids]
        assert len(records)==len(rows)==32
        write(OUT/f'data/{dest}.jsonl',rows,True)
        write(OUT/f'data/{dest}.gold.json',dict(protocol_version='markii-paucf-200-zss7-yesno-v1',window_seconds=4.,boundary_semantics='[start,end)',records=records))
        cohorts[dest]=rows
        proof[dest]=dict(source=str(manifest),source_sha256=sha(manifest),manifest_sha256=sha(OUT/f'data/{dest}.jsonl'),
            videos=32,decisions=sum(r['inference_frame_count'] for r in rows),gold_events=sum(len(r['windows']) for r in records))
    assert not {r['video_id'] for r in cohorts['development']} & {r['video_id'] for r in cohorts['confirmation']}
    # Reproduce native gold-history training examples from just the development records.
    wanted={int(r['raw_index']) for r in cohorts['development']}
    original=SG/'datasets/markii/val.jsonl'
    raw=[]
    for i,l in enumerate(original.open()):
        if i in wanted: raw.append(json.loads(l))
    write(OUT/'data/development_annotations.jsonl',raw,True)
    from stream_gemma.data import build_episode_specs
    config=make_args();config.annotation_path=str(OUT/'data/development_annotations.jsonl')
    config.video_sample_stride=1;config.target_episode_count=0
    specs,report=build_episode_specs(config)
    rows=[];eligible={(Path(r['resolved_video_path']).stem,r['question']):r for r in cohorts['development']}
    for episode,spec in enumerate(specs):
        key=(spec.video_id,spec.question)
        if key not in eligible: raise ValueError('unexpected gold-history video')
        source=eligible[key]
        turns=[dict(t) for t in spec.turns]
        for i,t in enumerate(turns):
            if not t['learn']: continue
            label='Hold' if t['state']=='Standby' else t['state']
            if label=='Hold':
                previous=max((j for j in range(i) if turns[j]['state']=='Response'),default=-1)
                if previous<0 or not all(x['state']=='Standby' for x in turns[previous+1:i+1]): continue
            rows.append(dict(episode_id=episode,turn_index=i,video_id=source['video_id'],
                video_path=source['resolved_video_path'],question=spec.question,label=label,
                frame_index=int(t['frame_index']),window_start=int(spec.chunk_start),
                memory=None if spec.memory is None else dict(frame_index=spec.memory.frame_index,time_seconds=spec.memory.time_seconds,content=spec.memory.content),
                turns=turns[:i+1],split='development'))
    if not rows or len({r['label'] for r in rows})!=3: raise ValueError('gold-history coverage missing')
    write(OUT/'data/gold_history.jsonl',rows,True)
    checkpoints=[1332,1776];margins=[-2.,-1.,0.,1.,2.]
    conditions=[dict(name=f'step{s}_margin{m:+g}',step=s,margin=m,
        controller=str(OUT.parent/f'representative-seed42/train/step-{s:06d}'),
        manifest=str(OUT/'data/development.jsonl'),output=str(OUT/f'rollouts/development/step{s}_margin{m:+g}')) for s in checkpoints for m in margins]
    for c in conditions:
        if not Path(c['controller'],'training_state.pt').is_file(): raise FileNotFoundError(c['controller'])
    write(OUT/'development_conditions.json',conditions)
    write(OUT/'plan.json',dict(host='ip-10-3-70-177',seed=42,checkpoints=checkpoints,response_margins=margins,
        cohorts=proof,gold_history_rows=len(rows),gold_history_labels=dict(Counter(r['label'] for r in rows)),
        gold_history_sha256=sha(OUT/'data/gold_history.jsonl'),
        note='Gold-history subset follows native episode sampling and excludes ambiguous memory-only Hold; report per-class metrics, not a population loss estimate.',
        selection='Maximize development judged PAUC-F@4s per checkpoint; ties prefer margin nearest zero. Freeze selected margins before confirmation.',
        judge='Existing GPT-4.1 MarkII PAUC-F@4s scorer for these new development and confirmation rollouts.',
        scope='No WEB tuning and no new training run; recommend the next experiment after diagnosis.'))
    print(json.dumps(dict(cohorts=proof,gold_history_rows=len(rows),gold_history_labels=dict(Counter(r['label'] for r in rows)))))
