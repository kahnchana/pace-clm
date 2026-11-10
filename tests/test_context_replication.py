"""Budget gates must reject inconclusive replication and weak scaling evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_qwen35_context_replication import replicate_passes,million_passes,clearly_worse

def result(point,lo,hi):return {'delta_shuffled_minus_reference':point,'ci95':[lo,hi]}
def test_replication_requires_positive_interval_and_practical_effect():
    assert not replicate_passes(result(4.8,-.1,9.))
    assert not replicate_passes(result(4.8,0.,9.))
    assert not replicate_passes(result(.9,.1,1.8))
    assert replicate_passes(result(1.,.01,2.))
def test_scaling_requires_fresh_gain_and_anchor_noninferiority():
    assert not million_passes(result(2.,0.,4.),result(0.,-.9,.9))
    assert not million_passes(result(2.,.1,4.),result(0.,-1.,1.))
    assert million_passes(result(2.,.1,4.),result(0.,-.99,1.))
def test_regression_stop_does_not_treat_uncertainty_as_established_harm():
    assert not clearly_worse(result(-2.,-5.,1.))
    assert not clearly_worse(result(-2.,-5.,-1.))
    assert clearly_worse(result(-2.,-5.,-1.01))
