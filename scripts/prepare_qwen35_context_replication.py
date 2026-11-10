#!/usr/bin/env python3
"""Freeze a new continuation seed, unused heldouts, and a gated full-source ladder."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from qwen35_context_order import sha256
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'outputs/jevany-qwen35-2b-context-order-pilot'
OUT=ROOT/'outputs/jevany-qwen35-2b-context-replication-seed43'
BASE=ROOT/'outputs/jevany-qwen35-2b-mixture-12h'

def freeze(path,value):
    text=json.dumps(value,indent=2,sort_keys=True)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:raise ValueError(f'frozen artifact differs: {path}')
    path.write_text(text)
def save(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path,mmap_mode='r'),array):raise ValueError(f'frozen plan differs: {path}')
    else:np.save(path,array)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'launch_plan.json').exists():
        print('Frozen plan already prepared:',OUT);return
    parent=json.loads((PILOT/'launch_plan.json').read_text());audit=json.loads((BASE/'plans/audit.json').read_text())
    original=np.load(parent['reference_order'],mmap_mode='r');n=len(original)//128;full_end=n*128
    ids=np.arange(n);permutations=[]
    for start in range(1344,n,32):
        end=min(start+32,n);perm=np.random.default_rng(np.random.SeedSequence([43,start//32])).permutation(end-start)
        ids[start:end]=np.arange(start,end)[perm];permutations.append(dict(start_step=start,permutation=perm.tolist()))
    replication=np.array(original,copy=True);fresh=np.array(original,copy=True)
    batches=original[:full_end].reshape(n,128,2)
    replication[1344*128:1792*128]=batches[ids[1344:1792]].reshape(-1,2)
    fresh[:full_end]=batches[ids].reshape(-1,2)
    assert np.array_equal(replication[:1344*128],original[:1344*128])
    assert np.array_equal(fresh[:1792*128],replication[:1792*128])
    # Reuse complete already-seen global batches, with exactly matched class
    # counts at every subsequent optimizer update. This is deliberately replay.
    labels=np.load(BASE/'plans/labels.npy',mmap_mode='r')[:full_end].reshape(n,128)
    counts=np.stack([(labels==k).sum(1) for k in range(3)],axis=1)
    groups={}
    for i in range(1792):groups.setdefault(tuple(counts[ids[i]]),[]).append(int(ids[i]))
    rng=np.random.default_rng(4301);replay_ids=ids.copy()
    for i in range(1792,n):
        key=tuple(counts[ids[i]])
        if key not in groups:raise ValueError(f'no already-seen replay batch with class quota {key}')
        replay_ids[i]=rng.choice(groups[key])
    assert np.array_equal(counts[ids[1792:]],counts[replay_ids[1792:]])
    replay=np.array(original,copy=True);replay[:full_end]=batches[replay_ids].reshape(-1,2)
    for name,array in [('replication',replication),('fresh',fresh),('replay',replay)]:save(OUT/f'plans/{name}.npy',array)
    save(OUT/'plans/fresh_original_batch_ids.npy',ids);save(OUT/'plans/replay_original_batch_ids.npy',replay_ids)
    used=set()
    for directory in [BASE/'scaling-diagnostic/data',PILOT/'evaluation/data']:
        for name in ['development','confirmation']:
            used.update(json.loads(l)['video_id'] for l in (directory/f'{name}.jsonl').read_text().splitlines())
    available={};gold_by_id={};source_hashes={}
    for name in ['selection','final']:
        f=ROOT/f'outputs/pace-clm-mixtures/validation/{name}.jsonl'
        if sha256(f)!=audit['exclusion_files'][str(f)]:raise ValueError('heldout exclusion source changed')
        source_hashes[str(f)]=sha256(f)
        gold=json.loads(f.with_name(name+'.gold.json').read_text())
        for r in gold['records']:gold_by_id[r['question_id']]=r
        for line in f.read_text().splitlines():
            r=json.loads(line)
            if r['video_id'] not in used:available[r['video_id']]=r
    # All three cohorts are frozen before predictions. Remaining heldout length
    # availability prevents reproducing the previous 16/32/16 allocation.
    pools=[]
    for lo,hi,num in [(0,32,42),(33,96,6),(97,10000,16)]:
        rows=[r for r in available.values() if lo<=r['inference_frame_count']<=hi]
        rows.sort(key=lambda r:hashlib.sha256(f'context-replication-43:{r["question_id"]}'.encode()).digest())
        if len(rows)<3*num:raise ValueError('insufficient unused heldout stratum')
        pools.append((rows,num))
    cohorts={};all_selected=set()
    for j,name in enumerate(['replication','scaling_development','scaling_confirmation']):
        selected=[r for rows,num in pools for r in rows[j*num:(j+1)*num]]
        selected.sort(key=lambda r:hashlib.sha256(r['question_id'].encode()).digest())
        selected_ids={r['video_id'] for r in selected};assert len(selected_ids)==64 and not selected_ids&(used|all_selected)
        all_selected.update(selected_ids)
        if any(not Path(r['resolved_video_path']).is_file() for r in selected):raise FileNotFoundError('heldout video')
        f=OUT/f'evaluation/data/{name}.jsonl';f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
        g=dict(gold);g['records']=[gold_by_id[r['question_id']] for r in selected]
        freeze(f.with_name(name+'.gold.json'),g)
        cohorts[name]=dict(videos=64,decisions=sum(r['inference_frame_count'] for r in selected),
            events=sum(len(r['windows']) for r in g['records']),manifest_sha256=sha256(f),gold_sha256=sha256(f.with_name(name+'.gold.json')),
            length_quota=dict(short=42,medium=6,long=16))
    plan=dict(host='ip-10-3-70-177',base_training_seed=42,continuation_seed=43,shuffle_seed=43,
        independent_initialization_seed=False,seed_scope='New rank-specific continuation RNG streams and batch-window permutations, conditional on the same seed-42 parent; not an independent from-scratch training seed.',
        parent_checkpoint=str(PILOT/'shared/train/step-001344'),parent_recipe=str(PILOT/'shared/train/recipe.json'),
        source=parent['source'],source_sha256=parent['source_sha256'],reference_order=parent['reference_order'],reference_order_sha256=parent['reference_order_sha256'],
        replication_order=str(OUT/'plans/replication.npy'),fresh_order=str(OUT/'plans/fresh.npy'),replay_order=str(OUT/'plans/replay.npy'),
        plan_hashes={name:sha256(OUT/f'plans/{name}.npy') for name in ['replication','fresh','replay']},
        start_step=1344,end_step=1792,updates_per_arm=448,global_batch=128,microbatch=16,microbatch_ratio=1.,save_every=16,
        pointer_lr=1e-4,lora_lr=2e-6,backend='gloo',primary_margin=1.,training_arm_order=['reference','shuffled'],
        cohorts=cohorts,heldout_source_hashes=source_hashes,excluded_previously_used_video_ids=sorted(used),
        cohort_limitations='Fresh to this diagnostic series. Availability-constrained 42/6/16 length mix differs from previous 16/32/16 cohorts; compare arms within cohorts, not absolute scores across studies.',
        replication_gate=dict(minimum_delta_points=1.,ci95_lower_must_exceed=0.,otherwise='complete_replication_not_confirmed; do not launch scaling'),
        scaling=dict(source_rows=8343088,eligible_rows=len(original),epoch_rows=full_end,terminal_rows_reserved=len(original)-full_end,
            milestones=[4096,8192,16384,32768,n],replay_control_through_step=8192,
            first_milestone_safety='Stop if fresh-minus-replay or fresh-minus-anchor paired CI upper bound is below -1 point; an inconclusive result is not evidence of benefit.',
            million_presentation_gate='At step 8192 require fresh-minus-replay paired CI lower > 0 and fresh-minus-anchor lower > -1 before larger doses.',
            later_safety='Stop if a later dose is clearly worse than the step-8192 fresh model by more than 1 point (paired CI upper < -1).',
            final_comparison='Frozen scaling confirmation: full-dose fresh vs step-8192 fresh, step-8192 replay, and replication shuffled endpoint; all at fixed +1.',
            full_data_caveat='One full-batch pass over all eligible source rows, reserving the last 86 rows for batch alignment. All heldout exclusions remain in force; no source relabeling or deduplication.',
            budget='Fixed step doses, not a wall-clock stop. Full ladder is a multi-day experiment on eight GPUs; ETA measured from current-node timings.'),
        audit=dict(consumed_prefix_unchanged=True,replication_complete_batches_preserved=True,fresh_full_batch_permutation_verified=bool(np.array_equal(np.sort(ids),np.arange(n))),
            replay_exact_per_update_class_quotas=True,disjoint_new_cohorts=True,full_window_permutations=permutations))
    freeze(OUT/'launch_plan.json',plan)
    freeze(OUT/'authorization.json',dict(date='2026-11-08',source='User: Okay - let us try the next idea and then see if this scales up with our full dataset.',
        scope='Local eight-GPU continuation replication followed by preregistered gated full-source scaling, and required GPT-4.1 judging.',gateway=parent['evaluation']['gateway']))
    (OUT/'logs').mkdir(exist_ok=True);(OUT/'reports').mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
    print(json.dumps(dict(output=str(OUT),cohorts=cohorts,full_source_updates=n,epoch_rows=full_end,terminal_rows_reserved=len(original)-full_end)))
if __name__=='__main__':main()
