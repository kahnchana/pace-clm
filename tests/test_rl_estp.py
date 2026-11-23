import copy
from collections import Counter

import pytest

from pace_clm.rl_estp import continuation_schedule, estp_spec, video_splits
from pace_clm.rl_mixed import attach_loss_scales, episode_balanced_schedule


def qa(start=10.2, end=15.2, left=11.1, right=11.6):
    return dict(clip_start_time=start, clip_end_time=end, question='What changes?',
                conversation=[dict(role='assistant', content='It opens.', start_time=left, end_time=right)])


def test_native_frame_mapping_and_gold_windows_do_not_shift_to_make_targets_easier():
    spec = estp_spec('v', 'c', 0, qa(), '/video.mp4')
    assert spec['source_indices'] == [21, 23, 25, 27, 29]
    assert spec['timestamps'] == pytest.approx([.3, 1.3, 2.3, 3.3, 4.3])
    assert spec['gold_answers'][0]['reply_timespan'] == pytest.approx([.9, 1.4])
    with pytest.raises(ValueError, match='unreachable'):
        estp_spec('v', 'c', 0, qa(left=11.7, right=11.9), '/video.mp4')
    # Evaluation keeps unreachable windows in its denominator.
    spec = estp_spec('v', 'c', 0, qa(left=11.7, right=11.9), '/video.mp4', training=False)
    assert spec['gold_answers'][0]['reply_timespan'] == pytest.approx([1.5, 1.7])


def test_invalid_bounds_and_dynamic_queries_are_not_silent_training_examples():
    item = qa()
    del item['clip_end_time']
    with pytest.raises(ValueError, match='bounds'):
        estp_spec('v', 'c', 0, item, '/video.mp4')
    with pytest.raises(ValueError, match='outside_media'):
        estp_spec('v', 'c', 0, qa(), '/video.mp4', frame_count=22)
    with pytest.raises(ValueError, match='dynamic_question'):
        estp_spec('v', 'c', 0, {**qa(), 'question_time': 12}, '/video.mp4')


def test_benchmark_retains_native_zero_width_and_outside_clip_targets():
    item = qa(left=16, right=16)
    spec = estp_spec('v', 'c', 0, item, '/video.mp4', training=False, benchmark=True)
    assert spec['gold_answers'][0]['reply_timespan'] == pytest.approx([5.8, 5.8])
    with pytest.raises(ValueError, match='answer_window'):
        estp_spec('v', 'c', 0, item, '/video.mp4')
    spec=estp_spec('v','c',0,item,'/video.mp4',training=False,benchmark=True,frame_count=25)
    assert spec['source_indices']==[21,23]
    assert spec['gold_answers'][0]['reply_timespan']==pytest.approx([5.8,5.8])


def test_split_keeps_source_video_groups_together():
    splits = video_splits([str(i) for i in range(1166)])
    assert list(map(len, splits.values())) == [932, 117, 117]
    assert len(set(sum(splits.values(), []))) == 1166
    assert splits == video_splits(list(reversed([str(i) for i in range(1166)])))


def test_128_schedule_preserves_parent_and_replay_cursor_without_repeating_episodes():
    episodes = [dict(domain=d, span_id=f'{d}:{i}', video_id=f'{d}:{i}', rollout_frames=n)
                for d, count, n in [('ours', 600, 79), ('web', 400, 18), ('ego', 145, 361)]
                for i in range(count)]
    prefix = episode_balanced_schedule(episodes, budget=10**6)[:32]
    episodes += [dict(domain='ego', data_source='estp_it', span_id=f'it:{i}', video_id=f'it:{i}',
                      rollout_frames=300) for i in range(111)]
    original = copy.deepcopy(prefix)
    schedule = continuation_schedule(episodes, prefix)
    assert prefix == original == schedule[:32]
    assert len(schedule) == 128
    selected = [i for b in schedule for i in b['episode_indices']]
    assert len(selected) == len(set(selected)) == 1024
    assert Counter(episodes[i].get('data_source', episodes[i]['domain'])
                   for b in schedule[32:] for i in b['episode_indices']) == dict(ours=384, web=192, ego=81, estp_it=111)
    assert sum(b['anchor_count'] for b in schedule) == schedule[-1]['exposure_after'] * 126 // 4096
    for i, batch in enumerate(schedule[32:], 32):
        assert Counter(episodes[j]['domain'] for j in batch['episode_indices']) == dict(ours=4, web=2, ego=2)
        assert batch['anchor_start'] == schedule[i-1]['anchor_start'] + schedule[i-1]['anchor_count']


def test_estp_uses_egocentric_family_weight_with_separate_source_identity():
    traces = [dict(spec=dict(domain=d, data_source=source), steps=[{}])
              for d, source, n in [('ours', 'ours', 16), ('web', 'web', 8),
                                    ('ego', 'ego', 4), ('ego', 'estp_it', 4)] for _ in range(n)]
    assert attach_loss_scales(traces) == dict(ours=16, web=8, ego=8)
    assert all(t['steps'][0]['loss_scale'] == 1 for t in traces)


def test_estp_clip_relative_reward_uses_native_window_and_all_physical_fires(tmp_path):
    from jevany_mixed_reward import score_predictions
    spec=estp_spec('v','c',0,qa(start=10.5,end=15.5,left=11.5,right=11.8),'/video.mp4')
    original=copy.deepcopy(spec['gold_answers'])
    predictions=[dict(question_id='it',model_response_list=[
        dict(role='assistant',time=1.,content='It opens.'),
        dict(role='assistant',time=2.,content='It opens.')])]
    gold=[dict(question_id='it',conversation=[dict(role='user',content=spec['question'])],answer=spec['gold_answers'])]
    outputs=[dict(custom_id='it=0=1.0',response=dict(body=dict(choices=[dict(message=dict(content='3'))])))]
    metrics,_=score_predictions(predictions,gold,tmp_path,tmp_path/'unused',outputs)
    assert spec['gold_answers']==original
    assert metrics['it']['num_fires']==2
    assert metrics['it']['num_value_adding_fires']==1
    assert metrics['it']['num_gold_turns']==1


def test_failed_attempts_are_archived_without_moving_committed_parent(tmp_path):
    import json
    from run_jevany_estp128 import archive_uncommitted
    (tmp_path/'latest.json').write_text(json.dumps(dict(step=32)))
    parent=tmp_path/'parent';parent.mkdir()
    (tmp_path/'step-000032').symlink_to(parent,target_is_directory=True)
    (tmp_path/'batch-000032').mkdir();(tmp_path/'batch-000032/rollouts.json').write_text('[]')
    (tmp_path/'step-000033').mkdir()
    archive_uncommitted(tmp_path)
    assert (tmp_path/'step-000032').is_symlink()
    assert len(list((tmp_path/'audit').glob('*/batch-000032/rollouts.json')))==1
    assert not (tmp_path/'step-000033').exists()


def test_endpoint_documentation_requires_step128_and_completed_judging(tmp_path):
    import json
    from run_jevany_estp128 import document_completion
    (tmp_path/'latest.json').write_text(json.dumps(dict(step=32)))
    with pytest.raises(ValueError,match='before step128'):
        document_completion(tmp_path)
    (tmp_path/'latest.json').write_text(json.dumps(dict(step=128)))
    (tmp_path/'queue').mkdir();(tmp_path/'queue/full-step128.json').write_text('{}')
    with pytest.raises(ValueError,match='pending evaluations'):
        document_completion(tmp_path)


def test_quality_warnings_cannot_stop_training_and_endpoint_stays_local(tmp_path):
    import json
    from jevany_estp_control import before_batch, publish_checkpoint
    from eval_jevany_estp128 import chosen_cohorts
    (tmp_path/'continuation-gate-passed.json').write_text('{"verified":true}')
    (tmp_path/'stop-request.json').write_text('{"reasons":["low_quality"]}')
    assert before_batch(tmp_path,33) is True
    (tmp_path/'workflow-freeze.json').write_text(json.dumps(dict(full_steps=[48,64,80,96,112,128])))
    publish_checkpoint(tmp_path,128,tmp_path/'step-000128',dict(episode_schedule=dict(exposure_after=500000)))
    assert json.loads((tmp_path/'queue/full-step128.json').read_text())['kind']=='final'
    cohorts=dict(ego_heldout=['e'],estp_it_dev=['d'],estp_it_test=['t'],estp_bench=['b'])
    assert set(chosen_cohorts(cohorts,False))=={'ego_heldout','estp_it_dev'}
    assert chosen_cohorts(cohorts,True)==cohorts
