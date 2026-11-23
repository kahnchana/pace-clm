"""Keep official PO placeholder behavior separate from real-response timing."""
from run_jevany_sb_moe import po_counts


def test_blank_placeholder_has_only_official_timing_credit():
    records=[dict(qid='q',dialog_history=[dict(time=10.,content='')],all_fires=[])]
    examples={'q':dict(ground_truth_time=10.,keywords='answer')}
    result=po_counts(records,examples)
    assert result['official_time_accuracy']==100
    assert result['corrected_time_accuracy']==0
    assert result['answer_accuracy']==0
    assert result['blank_placeholder_timing_credit']==1


def test_first_response_controls_timing_and_case_sensitive_answer():
    first=dict(time=5.,content='Answer')
    later=dict(time=10.,content='answer')
    records=[dict(qid='q',dialog_history=[first],all_fires=[first,later])]
    examples={'q':dict(ground_truth_time=10.,keywords='answer')}
    result=po_counts(records,examples)
    assert result['official_time_accuracy']==0
    assert result['corrected_time_accuracy']==0
    assert result['answer_accuracy']==0
    first['time']=8.
    result=po_counts(records,examples)
    assert result['corrected_time_accuracy']==100
    assert result['answer_accuracy']==0
