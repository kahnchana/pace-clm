import pytest
import torch

from pace_clm.rl import categorical_loss, leave_one_out, observation, reward_ledger, pad_inputs, parity_metrics, mask_padding_states


def test_parity_distinguishes_rare_log_rounding_from_policy_drift():
    old=torch.tensor([[1.-2e-8,1e-8,1e-8]]).log()
    current=torch.tensor([[1.-4e-8,2e-8,2e-8]]).log()
    stats=parity_metrics(current,old,torch.tensor([0]))
    assert stats['all_logp_max'] > .5
    assert stats['probability_max'] < 1e-6
    assert stats['selected_logp_max'] < 1e-6
    stats=parity_metrics(torch.tensor([[.8,.1,.1]]).log(),
                         torch.tensor([[.5,.25,.25]]).log(),torch.tensor([0]))
    assert stats['probability_max'] > .2
    assert stats['kl_max'] > .01


def test_bucket_padding_preserves_observation_and_masks_new_tokens():
    inputs = {'input_ids':torch.tensor([[3,4]]),'attention_mask':torch.ones(1,2,dtype=torch.long),
              'mm_token_type_ids':torch.tensor([[0,1]]),'image_grid_thw':torch.tensor([[1,2,2]])}
    result = pad_inputs(inputs,4,99)
    assert result['input_ids'].tolist() == [[3,4,99,99]]
    assert result['attention_mask'].tolist() == [[1,1,0,0]]
    assert result['mm_token_type_ids'].tolist() == [[0,1,0,0]]
    assert result['image_grid_thw'] is inputs['image_grid_thw']
    assert inputs['input_ids'].shape[-1] == 2
    with pytest.raises(ValueError): pad_inputs(inputs,1,99)


def test_group_credit_and_gradient_sign():
    assert leave_one_out([1., 1., 1., 1.]) == [0.]*4
    assert leave_one_out([1., 0.]) == [1., -1.]
    logits = torch.zeros(2, 3, requires_grad=True)
    loss, kl, _ = categorical_loss(logits, torch.log_softmax(logits.detach(), -1),
                                 torch.tensor([2, 2]), torch.tensor([1., -1.]),
                                 total_trajectories=2, entropy_beta=0)
    loss.backward()
    assert logits.grad[0, 2] < 0 < logits.grad[1, 2]
    assert torch.allclose(kl, torch.zeros(2))


def test_microbatch_global_weighting():
    logits = torch.randn(7, 3, requires_grad=True)
    ref = torch.log_softmax(torch.randn(7, 3), -1)
    actions, adv = torch.arange(7)%3, torch.randn(7)
    whole = categorical_loss(logits, ref, actions, adv, total_trajectories=4)[0]
    chunks = sum(categorical_loss(logits[s], ref[s], actions[s], adv[s],
                                  total_trajectories=4)[0]
                 for s in (slice(0, 2), slice(2, 7)))
    assert torch.allclose(whole, chunks)


def test_reward_recurring_events_and_mutually_exclusive_penalties():
    gold = [{'time_seconds': 2}, {'time_seconds': 10}]
    fires = [{'frame_index': f} for f in (2, 3, 10, 17)]
    rows = reward_ledger(gold, fires, [[1, 1], [1, 1], [1, 1], [0, 0]], [0, 1, 1, 0])
    assert [r['kind'] for r in rows].count('matched') == 2
    assert sum(r['value'] for r in rows) == pytest.approx(.7)
    assert sum(r['kind'] == 'duplicate' for r in rows) == 1
    assert sum(r['kind'] == 'unmatched' for r in rows) == 1
    assert sum(r['value'] for r in reward_ledger(gold, [], [], [])) == -.5
    assert reward_ledger([], [], [], []) == []


def test_reward_boundaries_and_failed_judge():
    assert reward_ledger([{'time_seconds': 2}], [{'frame_index': 6}], [[1]], [0])[0]['kind'] == 'unmatched'
    with pytest.raises(ValueError):
        reward_ledger([], [{'frame_index': 0}], [], [])
    with pytest.raises(ValueError):
        reward_ledger([], [{'frame_index': 0}], [[]], [float('nan')])


def test_exact_history_and_aged_memory():
    spec = {'decode_start': 0, 'span_id': 's', 'video_id': 'v', 'video_path': 'v.mp4',
            'question': 'q', 'initial_memory': None}
    history = [{'frame_index': i, 'emit': '</Response> real answer' if i == 0 else '</Silence>',
                'ts': f'00:{i:02d}'} for i in range(33)]
    row = observation(spec, history, 33)
    assert len(row['turns']) == 32
    assert row['turns'][0]['frame_index'] == 2
    assert row['memory']['content'] == 'real answer'
    with pytest.raises(ValueError):
        observation(spec, history[3:], 33)


def test_span_prefix_timestamp_conversion():
    spec = {'decode_start': 59, 'span_id': 's', 'video_id': 'v',
            'video_path': 'v.mp4', 'question': 'q', 'initial_memory': None}
    row = observation(spec, [{'frame_index':59,'time_seconds':59.,
                              'state':'Silence','emit':'</Silence>'}],60)
    assert [t['ts'] for t in row['turns']] == ['00:59','01:00']


def test_singleton_left_padding_is_masked_like_batched_recurrent_inputs():
    x=torch.randn(1,5,3,dtype=torch.bfloat16,requires_grad=True)
    mask=torch.tensor([[0,0,1,1,1]])
    one=mask_padding_states(x,mask)
    many=mask_padding_states(x.expand(4,-1,-1),mask.expand(4,-1))
    assert torch.equal(one,many[:1])
    assert torch.equal(one[:,:2],torch.zeros_like(one[:,:2]))
    assert torch.equal(one[:,2:],x[:,2:])
    assert one.dtype==x.dtype
    one.float().sum().backward()
    assert torch.equal(x.grad[:,:2],torch.zeros_like(x.grad[:,:2]))
    assert mask_padding_states(x,None) is x


def test_lightweight_image_inputs_require_complete_vision_cache():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from pace_clm.rl_inputs import install_cached_vision_inputs
    entered=[]
    @contextmanager
    def original(vision):
        entered.append(vision)
        yield
    runtime=SimpleNamespace(_cached_image_features=original)
    install_cached_vision_inputs(runtime)
    with pytest.raises(RuntimeError,match='all causal frozen vision features'):
        with runtime._cached_image_features([SimpleNamespace(image_embeds=None)]):pass
    complete=[SimpleNamespace(image_embeds=torch.ones(1))]
    with runtime._cached_image_features(complete):assert entered==[complete]


def test_collection_batches_bound_relative_progress_and_wait_for_fill():
    from pace_clm.rl import ready_collection
    states=[{'frame':103,'pending':False,'spec':{'rollout_start':100,'rollout_stop':132}},
            {'frame':200,'pending':True,'spec':{'rollout_start':200,'rollout_stop':232}}]
    ready,wait,finished=ready_collection(states,max_lead=2,min_ready_fraction=.75,batch_size=24)
    assert ready==[] and wait and not finished
    states[1].update(frame=201,pending=False)
    ready,wait,finished=ready_collection(states,max_lead=2,min_ready_fraction=.75,batch_size=24)
    assert ready==states and not wait and not finished
    states[0]['frame']=132;states[1]['frame']=232
    assert ready_collection(states,max_lead=2,min_ready_fraction=.75,batch_size=24)==([],False,True)


def test_collection_waits_for_a_useful_batch_when_answers_are_pending():
    from pace_clm.rl import ready_collection
    states=[{'frame':0,'pending':i<10,'spec':{'rollout_start':0,'rollout_stop':32}} for i in range(19)]
    assert ready_collection(states,max_lead=2,min_ready_fraction=.75,batch_size=24)[1]
    for s in states[4:]:s['pending']=False
    ready,wait,_=ready_collection(states,max_lead=2,min_ready_fraction=.75,batch_size=24)
    assert len(ready)==15 and not wait
