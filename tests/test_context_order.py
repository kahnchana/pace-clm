import sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from qwen35_context_order import shuffle_update_blocks,validate_fork,sha256

def test_shuffle_preserves_batches_ranks_and_prefix():
    order=np.arange(128*128*2,dtype=np.uint64).reshape(-1,2)
    new,windows=shuffle_update_blocks(order,32,96)
    again,_=shuffle_update_blocks(order,32,96)
    assert np.array_equal(new,again)
    assert np.array_equal(new[:32*128],order[:32*128])
    assert np.array_equal(new[96*128:],order[96*128:])
    for w in windows:
        start=w['start_step'];a=order[start*128:(start+32)*128].reshape(32,128,2)
        b=new[start*128:(start+32)*128].reshape(32,128,2)
        assert sorted(w['permutation'])==list(range(32))
        for i,j in enumerate(w['permutation']):assert np.array_equal(b[i],a[j])

def test_bad_window_boundaries():
    with pytest.raises(ValueError):shuffle_update_blocks(np.zeros((10000,2)),31,64)

def test_fork_rejects_consumed_prefix_changes(tmp_path):
    order=np.arange(128*96*2,dtype=np.uint64).reshape(-1,2)
    parent=tmp_path/'parent.npy';new=tmp_path/'new.npy';np.save(parent,order)
    shuffled,_=shuffle_update_blocks(order,32,64);np.save(new,shuffled)
    current=dict(seed=42,backend='gloo',microbatch=16,microbatch_ratio=1.,require_global_batch=128,
        frame_cache_per_gpu=1024,vision_chunk=0,conv_batch_chunk=128,prefetch_mode='prepare',predecode_workers=4)
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps(dict(**current,manifest_sha256='source',row_order=str(parent),row_order_sha256=sha256(parent))))
    result=validate_fork(recipe,new,'source',32,512,8,current)
    assert result['verified_consumed_prefix_rows']==4096
    shuffled[0,0]+=1;np.save(new,shuffled)
    with pytest.raises(ValueError,match='already-consumed'):validate_fork(recipe,new,'source',32,512,8,current)
    with pytest.raises(ValueError,match='manifest'):validate_fork(recipe,new,'changed',32,512,8,current)
