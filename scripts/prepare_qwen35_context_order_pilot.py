#!/usr/bin/env python3
"""Freeze a matched-exposure context-order pilot and fresh evaluation cohorts."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from qwen35_context_order import shuffle_update_blocks,sha256

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/jevany-qwen35-2b-mixture-12h'
OUT=ROOT/'outputs/jevany-qwen35-2b-context-order-pilot'

def freeze(path,value,jsonl=False):
    text=''.join(json.dumps(v,sort_keys=True)+'\n' for v in value) if jsonl else json.dumps(value,indent=2,sort_keys=True)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:raise FileExistsError(path)
    path.write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    recipe_path=BASE/'representative-seed42/train/recipe.json';recipe=json.loads(recipe_path.read_text())
    original=Path(recipe['row_order'])
    if sha256(original)!=recipe['row_order_sha256']:raise ValueError('original plan hash changed')
    old=np.load(original,mmap_mode='r');shuffled,windows=shuffle_update_blocks(old)
    dest=OUT/'plans/shuffled.npy';dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        if not np.array_equal(np.load(dest,mmap_mode='r'),shuffled):raise FileExistsError(dest)
    else:np.save(dest,shuffled)
    lengths=np.load(BASE/'plans/lengths.npy',mmap_mode='r')
    before=lengths[1344*128:1792*128].reshape(14,32,128).mean(axis=2)
    after=np.array([before[i,w['permutation']] for i,w in enumerate(windows)])
    plan=dict(host='ip-10-3-70-177',seed=42,parent_checkpoint=str(BASE/'representative-seed42/train/step-001332'),
        parent_recipe=str(recipe_path),source=recipe['manifest'],source_sha256=recipe['manifest_sha256'],
        reference_order=str(original),reference_order_sha256=sha256(original),
        shuffled_order=str(dest),shuffled_order_sha256=sha256(dest),
        shared_step=1344,start_step=1344,end_step=1792,updates_per_arm=448,presentations_per_arm=57344,
        final_cumulative_presentations=229376,global_batch=128,microbatch=16,microbatch_ratio=1.,backend='gloo',
        pointer_lr=1e-4,lora_lr=2e-6,gradient_clip=1.,save_every=16,
        reference_policy='Fresh matched continuation from the same shared state; no historical quality reference substitution.',
        training_arm_order=['shuffled','reference'],
        audit=dict(exact_window_row_multiset=True,complete_global_batches_preserved=True,rank_assignment_within_batch_preserved=True,
            consumed_prefix_identical=True,suffix_after_pilot_identical=True,windows=windows,
            mean_length_phase_correlation_before=float(np.corrcoef(np.arange(32),before.mean(axis=0))[0,1]),
            mean_length_phase_correlation_after=float(np.corrcoef(np.arange(32),after.mean(axis=0))[0,1])),
        evaluation=dict(primary_margin=0.,secondary_development_margins=[-2.,-1.,0.,1.,2.],
            semantic_judge='gpt-4.1',gateway='https://gateway.salesforceresearch.ai/openai/process/v1/',
            criterion='Compare both fixed-zero-margin and development-selected-margin confirmation scores; paired-video intervals.',
            phase_probes=[1744,1760,1776,1792],phase_probe_scope='Matched 512-row native gold-history diagnostic; no rollout-quality claims from classifier loss.'))
    audit=json.loads((BASE/'plans/audit.json').read_text());cohorts={}
    for source_name,name,previous in [('selection','development','development'),('final','confirmation','confirmation')]:
        src=ROOT/f'outputs/pace-clm-mixtures/validation/{source_name}.jsonl'
        if audit['exclusion_files'][str(src)]!=sha256(src):raise ValueError('reserved source hash changed')
        old_ids={json.loads(l)['video_id'] for l in (BASE/f'scaling-diagnostic/data/{previous}.jsonl').read_text().splitlines()}
        rows=[json.loads(l) for l in src.read_text().splitlines()]
        available=[r for r in rows if r['video_id'] not in old_ids];selected=[]
        for lo,hi,n in [(0,32,16),(33,96,32),(97,10000,16)]:
            candidates=[r for r in available if lo<=r['inference_frame_count']<=hi]
            candidates.sort(key=lambda r:hashlib.sha256(f'context-order-pilot-42:{name}:{r["question_id"]}'.encode()).digest())
            if len(candidates)<n:raise ValueError('insufficient fresh evaluation stratum')
            selected.extend(candidates[:n])
        selected.sort(key=lambda r:hashlib.sha256(r['question_id'].encode()).digest())
        if len({r['video_id'] for r in selected})!=64:raise ValueError('cohort has duplicate videos')
        if any(not Path(r['resolved_video_path']).is_file() for r in selected):raise FileNotFoundError('cohort video')
        ids={r['question_id'] for r in selected}
        gold=json.loads(src.with_name(source_name+'.gold.json').read_text())
        gold['records']=[r for r in gold['records'] if r['question_id'] in ids]
        if len(gold['records'])!=64:raise ValueError('gold coverage')
        freeze(OUT/f'evaluation/data/{name}.jsonl',selected,True);freeze(OUT/f'evaluation/data/{name}.gold.json',gold)
        cohorts[name]=dict(videos=64,decisions=sum(r['inference_frame_count'] for r in selected),events=sum(len(r['windows']) for r in gold['records']),
            manifest_sha256=sha256(OUT/f'evaluation/data/{name}.jsonl'),gold_sha256=sha256(OUT/f'evaluation/data/{name}.gold.json'),
            source_sha256=sha256(src),excluded_previous_diagnostic_video_ids=sorted(old_ids))
    dev={json.loads(l)['video_id'] for l in (OUT/'evaluation/data/development.jsonl').read_text().splitlines()}
    conf={json.loads(l)['video_id'] for l in (OUT/'evaluation/data/confirmation.jsonl').read_text().splitlines()}
    assert not dev&conf
    plan['cohorts']=cohorts
    source_gold=[json.loads(l) for l in (BASE/'scaling-diagnostic/data/gold_history.jsonl').read_text().splitlines()]
    indexes=sorted(range(len(source_gold)),key=lambda i:hashlib.sha256(f'context-probe-42:{i}'.encode()).digest())[:512]
    freeze(OUT/'evaluation/data/phase_probe.jsonl',[source_gold[i] for i in sorted(indexes)],True)
    freeze(OUT/'launch_plan.json',plan)
    freeze(OUT/'authorization.json',dict(date='2026-11-07',source='User explicitly approved any GPT scoring as needed and launch of the next training run on this node using eight GPUs.',
        scope='Two matched context-order continuations and their development/confirmation evaluation.',
        judge_payload='Text-only questions, gold answers and newly generated answers.',gateway=plan['evaluation']['gateway']))
    print(json.dumps(dict(output=str(OUT),start=1344,end=1792,updates_per_arm=448,cohorts=cohorts,
        context_phase_correlation_before=plan['audit']['mean_length_phase_correlation_before'],context_phase_correlation_after=plan['audit']['mean_length_phase_correlation_after'])))

if __name__=='__main__':main()
