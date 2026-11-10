import copy
import queue
import time

from pace_clm.rl_broker import FrozenGeneratorBroker, AnswerSink


def row():
    return {'video_path':'v.mp4','question':'q','frame_index':2,'memory':None,
            'turns':[{'frame_index':2,'ts':'00:02','emit':''}]}


def test_shared_inflight_and_hot_cache_deliver_every_original_request():
    answers=[queue.Queue(),queue.Queue()]
    broker=FrozenGeneratorBroker(None,answers,start=False)
    a=(0,'a',row(),time.monotonic());b=(1,'b',row(),time.monotonic())
    broker.submit(a);broker.submit(b)
    assert broker.get(timeout=.1)==a
    try:broker.get(timeout=.01)
    except queue.Empty:pass
    else:raise AssertionError('duplicate work reached the generator')
    AnswerSink(broker,0).put(('a','frozen answer',{'gpu':7,'batch_id':'batch0','batch':1,'seconds':.2}))
    left,right=answers[0].get_nowait(),answers[1].get_nowait()
    assert left[:2]==('a','frozen answer') and right[:2]==('b','frozen answer')
    assert not left[2]['deduplicated'] and right[2]['deduplicated']
    broker.submit((1,'c',row(),time.monotonic()))
    cached=answers[1].get_nowait()
    assert cached[:2]==('c','frozen answer') and cached[2]['served_from_cache'] and cached[2]['batch']==0
    stats=broker.snapshot()
    assert stats['logical_requests']==3 and stats['unique_computations']==1
    assert stats['cache_hits']==1 and stats['inflight_duplicates']==1 and stats['pending_contexts']==0


def test_cache_keys_preserve_actual_history_images_question_and_memory():
    key=FrozenGeneratorBroker.context_key
    original=row();metadata={**original,'physical_batch':4,'padded_length':512,'policy_version':1,'episode_id':'other'}
    assert key(metadata)==key(original)
    for field,value in [('video_path','other.mp4'),('question','other'),('frame_index',3),
                        ('memory',{'content':'prior answer'}),('turns',[{'frame_index':2,'ts':'00:02','emit':'</Response> actual answer'}])]:
        changed=copy.deepcopy(original);changed[field]=value
        assert key(changed)!=key(original)


def test_judge_and_shutdown_messages_bypass_the_answer_cache():
    broker=FrozenGeneratorBroker(None,[queue.Queue()],start=False)
    task=('judge',0,[],'output')
    broker.submit(task);broker.submit(None)
    assert broker.get(timeout=.1)==task and broker.get(timeout=.1) is None
    assert broker.snapshot()['logical_requests']==0
