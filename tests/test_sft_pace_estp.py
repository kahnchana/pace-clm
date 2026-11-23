import pytest
from pace_clm.sft_pace_estp import action_plan, supervised_row, training_frames, CyclingPermutation


def spec():
    return dict(span_id='it:a',video_id='a',video_path='a.mp4',question='Report the event',
        source_indices=list(range(0,100,2)),timestamps=list(map(float,range(50))),
        gold_answers=[dict(reply_timespan=[2.2,5.2],content='past answer'),
                      dict(reply_timespan=[40,42],content='future answer')])


def test_native_window_and_history():
    s=spec();labels,responses=action_plan(s)
    assert labels[2:7]==['Silence','Response','Hold','Hold','Silence']
    row=supervised_row(s,40)
    assert row['label']=='Response' and row['memory']['content']=='past answer'
    assert row['turns'][-1]['emit']==''
    assert all(t['frame_index']<=80 for t in row['turns'])
    assert 'future answer' not in str(row)
    assert len(row['turns'])==32


def test_unreachable_and_overlap_rejected():
    s=spec();s['gold_answers'][0]['reply_timespan']=[2.2,2.4]
    with pytest.raises(ValueError,match='unreachable'):action_plan(s)
    s=spec();s['gold_answers'][1]['reply_timespan']=[4,7]
    with pytest.raises(ValueError,match='overlapping'):action_plan(s)


def test_sampling_preserves_positives_and_reconstructs():
    s=spec();frames=training_frames(s)
    assert {3,40}<=set(frames) and len(frames)<=12
    assert frames==training_frames(s)
    first=CyclingPermutation(19,42);restored=CyclingPermutation(19,42)
    values=[first.at(i) for i in range(57)]
    assert len(set(values[:19]))==19
    assert [restored.at(i) for i in range(23,57)]==values[23:]
