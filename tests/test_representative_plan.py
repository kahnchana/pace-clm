import importlib.util
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

spec = importlib.util.spec_from_file_location('mixture', Path(__file__).parents[1]/'scripts/plan_qwen35_representative.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_quota_coverage_and_determinism():
    labels = np.repeat(np.arange(3), [16392,3128,480])
    lengths = np.random.default_rng(1).integers(144,4800,len(labels))
    order = module.make_order(labels,lengths)
    assert np.array_equal(order,module.make_order(labels,lengths))
    assert np.array_equal(np.sort(order),np.arange(len(labels)))
    report = module.audit(labels,order)
    assert report['longest_updates_without_class'] == [0,0,0]
    rank_hold = [int(np.sum(labels[order][rank::8] == 2)) for rank in range(8)]
    assert max(rank_hold)-min(rank_hold) < 20


def test_small_and_terminal_windows():
    for counts in ([101,19,3],[10000,1900,297],[5000,1,0]):
        labels = np.repeat(np.arange(3),counts)
        order = module.make_order(labels,np.ones(len(labels)))
        module.audit(labels,order)


def test_benchmark_aliases():
    assert module.aliases('OSfMU69X3C4.7.mp4') & module.aliases('OSfMU69X3C4.mp4')
    assert module.aliases('c2d06df7-5d3a-4116-9edb-f1c81a4f669b-0-360.mp4') & module.aliases('c2d06df7-5d3a-4116-9edb-f1c81a4f669b-360-720.mp4')
