import importlib.util
from pathlib import Path
import numpy as np
import subprocess
import sys
import pytest

spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).parents[1]/'scripts/analyze_qwen35_scaling_diagnostic.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_margin_changes_response_decision_and_preserves_ties():
    assert m.chosen_action([1,0,2],0)=='Response'
    assert m.chosen_action([1,0,2],2)=='Silence'
    assert m.chosen_action([0,2,1],-2)=='Response'
    assert m.chosen_action([1,1,1],0)=='Silence'

def test_average_precision_ties_have_threshold_semantics():
    assert m.average_precision([1,0,1,0],[4,2,3,1])==1
    assert m.average_precision([1,0],[1,1])==.5
    assert m.average_precision([0,0],[1,2]) is None

def test_paired_bootstrap_identical_videos_zero_delta():
    rows={str(i):dict(total_physical_fires=i+1,value_adding_fires=1,num_gold_windows=2,pauc_recall_component=40+i) for i in range(8)}
    value=m.bootstrap_delta(rows,rows)
    assert value['delta_final_minus_earlier']==0
    assert value['ci95']==[0,0]


@pytest.mark.parametrize('script', ['prepare_qwen35_scaling_diagnostic.py', 'run_qwen35_scaling_diagnostic.py'])
def test_retired_margin_launchers_exit_before_imports_or_artifact_access(script,tmp_path):
    path=Path(__file__).parents[1]/'scripts'/script
    # -S disables site packages: rejection must happen before heavy model imports.
    result=subprocess.run([sys.executable,'-S',str(path)],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==1
    assert 'Retired: nonzero response-margin sweeps' in result.stderr
    assert not list(tmp_path.iterdir())
