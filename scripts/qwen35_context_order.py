"""Auditable batch-order forks for fixed-global-batch context-order experiments."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

BATCH=128
WINDOW_UPDATES=32

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def shuffle_update_blocks(order,start_step=1344,end_step=1792,seed=42):
    if start_step%32 or end_step%32 or end_step<=start_step:raise ValueError('fork interval must contain complete 32-update windows')
    if order.ndim!=2 or order.shape[1]!=2 or len(order)<end_step*BATCH:raise ValueError('invalid or insufficient byte-offset plan')
    result=np.array(order,copy=True);windows=[]
    for step in range(start_step,end_step,WINDOW_UPDATES):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step//WINDOW_UPDATES]))
        permutation=rng.permutation(WINDOW_UPDATES)
        original=order[step*BATCH:(step+WINDOW_UPDATES)*BATCH].reshape(WINDOW_UPDATES,BATCH,2)
        emitted=original[permutation]
        result[step*BATCH:(step+WINDOW_UPDATES)*BATCH]=emitted.reshape(-1,2)
        assert np.array_equal(np.sort(permutation),np.arange(WINDOW_UPDATES))
        for index,old in enumerate(permutation):assert np.array_equal(emitted[index],original[old])
        windows.append(dict(start_step=step,permutation=permutation.tolist()))
    assert np.array_equal(result[:start_step*BATCH],order[:start_step*BATCH])
    assert np.array_equal(result[end_step*BATCH:],order[end_step*BATCH:])
    return result,windows

def validate_fork(parent_recipe_path,new_order_path,manifest_hash,step,seen,world,current):
    parent_recipe_path=Path(parent_recipe_path)
    parent=json.loads(parent_recipe_path.read_text())
    if parent['manifest_sha256']!=manifest_hash:raise ValueError('fork manifest checksum mismatch')
    if parent.get('batch_plan') is not None or parent.get('batch_plan_sha256') is not None:raise ValueError('fork requires a fixed-batch parent')
    for key in ('seed','backend','microbatch','microbatch_ratio','require_global_batch','frame_cache_per_gpu','vision_chunk','conv_batch_chunk','prefetch_mode','predecode_workers'):
        if parent.get(key)!=current.get(key):raise ValueError(f'fork training contract mismatch: {key}')
    if current['require_global_batch']!=BATCH or world!=8 or seen*world!=step*BATCH:raise ValueError('fork cursor/global batch mismatch')
    original_path=parent.get('row_order')
    if not original_path or not new_order_path:raise ValueError('fork requires parent and child byte-offset plans')
    if sha256(original_path)!=parent['row_order_sha256']:raise ValueError('parent byte-offset plan was modified')
    original=np.load(original_path,mmap_mode='r');new=np.load(new_order_path,mmap_mode='r')
    cursor=seen*world
    if original.shape!=new.shape or original.dtype!=new.dtype or cursor>len(new):raise ValueError('fork plan shape/dtype mismatch')
    if not np.array_equal(original[:cursor],new[:cursor]):raise ValueError('fork changes already-consumed rows')
    return dict(parent_recipe=str(parent_recipe_path.resolve()),parent_recipe_sha256=sha256(parent_recipe_path),
        parent_row_order_sha256=parent['row_order_sha256'],fork_step=step,seen_per_rank=seen,
        verified_consumed_prefix_rows=cursor,consumed_prefix_sha256=hashlib.sha256(new[:cursor].tobytes()).hexdigest())
