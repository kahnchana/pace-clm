import json
import math
import copy

import pytest
import torch

from pace_clm.grpo import (group_advantages, clipped_loss, update_schedule,
                           partition_trajectories, recover_progress, validate_progress, sha, batch_identity, BATCH_FILES)


def test_population_normalization_and_exact_zero_equal_reward_groups():
    rewards = [0., 0., 0., 1.]
    mean = .25
    std = math.sqrt(3/16)
    assert group_advantages(rewards) == pytest.approx([(x-mean)/(std+1e-6) for x in rewards])
    assert group_advantages([1/7]*4) == [0.]*4
    with pytest.raises(ValueError):group_advantages([0., float('nan')])


def test_ratios_start_at_one_and_behavior_reference_stay_detached():
    logits = torch.tensor([[.1,.2,.3],[.3,.2,.1]], requires_grad=True)
    old = torch.log_softmax(logits.detach(),-1)[:,2].clone().requires_grad_()
    ref = torch.log_softmax(logits.detach(),-1).clone().requires_grad_()
    loss, kl, entropy, metrics = clipped_loss(logits,ref,torch.tensor([2,2]),torch.tensor([1.,-1.]),old,total_trajectories=2)
    assert metrics['ratios'].tolist() == [1.,1.]
    assert kl.tolist() == [0.,0.]
    loss.backward()
    assert old.grad is None and ref.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('advantage,ratio,expected', [(1.,1.5,-1.2),(-1.,.5,.8),(1.,.5,-.5),(-1.,1.5,1.5)])
def test_clipping_branches_include_negative_advantages(advantage,ratio,expected):
    logits=torch.zeros(1,3,requires_grad=True)
    old=torch.tensor([math.log(1/3)-math.log(ratio)])
    loss,_,_,metrics=clipped_loss(logits,torch.log_softmax(logits.detach(),-1),torch.tensor([2]),
        torch.tensor([advantage]),old,total_trajectories=1,horizon=1,kl_beta=0,entropy_beta=0)
    assert float(loss)==pytest.approx(expected)
    assert metrics['outside_clip'].item()
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_zero_advantage_keeps_regularization_and_global_denominator():
    logits=torch.tensor([[1.,0.,0.],[0.,1.,0.]],requires_grad=True)
    ref=torch.log_softmax(torch.zeros(2,3),-1)
    old=torch.log_softmax(logits.detach(),-1)[:,2]
    loss,*_=clipped_loss(logits,ref,torch.tensor([2,2]),torch.zeros(2),old,total_trajectories=4)
    pieces=sum(clipped_loss(logits[i:i+1],ref[i:i+1],torch.tensor([2]),torch.zeros(1),old[i:i+1],total_trajectories=4)[0] for i in range(2))
    assert torch.allclose(loss,pieces)
    loss.backward()
    assert float(logits.grad.abs().sum())>0
    with pytest.raises(ValueError,match='nonfinite'):
        clipped_loss(logits,ref,torch.tensor([2,2]),torch.ones(2),torch.full((2,),-1e30),total_trajectories=4)


def test_two_passes_shuffle_whole_trajectories_and_handle_seven_ranks():
    schedule=update_schedule(range(128),42,0)
    assert schedule==update_schedule(range(128),42,0)
    assert schedule!=update_schedule(range(128),42,1)
    assert [x['pass'] for x in schedule]==[0,0,1,1]
    for actor_pass in range(2):
        assert sorted(sum([x['trajectory_ids'] for x in schedule if x['pass']==actor_pass],[]))==list(range(128))
    trajectories=[{'trajectory_id':i,'steps':[{'id':i} for _ in range(32)]} for i in range(128)]
    ranks=partition_trajectories(trajectories,schedule[0]['trajectory_ids'])
    for rank,steps in enumerate(ranks):
        assert all(step['id']%7==rank for step in steps)
        assert len(steps)==32*sum(identity%7==rank for identity in schedule[0]['trajectory_ids'])
    assert sum(map(len,ranks))==64*32


def test_atomic_commit_recovery_does_not_repeat_saved_update(tmp_path):
    parent=tmp_path/'optimizer_checkpoints'
    parent.mkdir()
    checkpoint=parent/'update-000001'
    checkpoint.mkdir()
    (checkpoint/'model').write_text('updated-weights')
    progress={'step':0,'optimizer_steps':1,'cursor':{'round':0,'next_update':1},'checkpoint':str(checkpoint.resolve())}
    (checkpoint/'commit.json').write_text(json.dumps({'parent_checkpoint':None,'progress':progress,'files':{'model':sha(checkpoint/'model')}}))
    assert recover_progress(tmp_path)==progress
    assert recover_progress(tmp_path)==progress
    # An interrupted second save is never adopted.
    incomplete=parent/'update-000002.incomplete'
    incomplete.mkdir()
    (incomplete/'model').write_text('not committed')
    assert recover_progress(tmp_path)==progress
    (checkpoint/'model').write_text('corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        recover_progress(tmp_path)


def test_round_counter_is_separate_from_four_optimizer_steps():
    validate_progress({'step':1,'optimizer_steps':4,'cursor':None})
    validate_progress({'step':1,'optimizer_steps':6,'cursor':{'round':1,'next_update':2}})
    with pytest.raises(ValueError):validate_progress({'step':1,'optimizer_steps':1,'cursor':None})


def test_midbatch_optimizer_restore_preserves_behavior_and_matches_uninterrupted_updates():
    torch.manual_seed(42)
    features=torch.randn(4,2)
    model=torch.nn.Linear(2,3,bias=False)
    initial=copy.deepcopy(model.state_dict())
    actions=torch.tensor([0,1,2,0])
    reference=torch.log_softmax(model(features).detach(),-1)
    behavior=reference.gather(-1,actions[:,None]).squeeze(-1).clone()
    advantages=torch.tensor([-1.,.5,1.,-.5])
    batches=[[0,1],[2,3],[0,1],[2,3]]
    def update(actor,optimizer,indices,old):
        optimizer.zero_grad()
        loss,*_=clipped_loss(actor(features[indices]),reference[indices],actions[indices],
                            advantages[indices],old[indices],total_trajectories=2,horizon=1)
        loss.backward();optimizer.step()
    full=torch.nn.Linear(2,3,bias=False);full.load_state_dict(initial)
    full_optimizer=torch.optim.AdamW(full.parameters(),lr=.02)
    for indices in batches:update(full,full_optimizer,indices,behavior)
    partial=torch.nn.Linear(2,3,bias=False);partial.load_state_dict(initial)
    optimizer=torch.optim.AdamW(partial.parameters(),lr=.02)
    update(partial,optimizer,batches[0],behavior)
    checkpoint=copy.deepcopy({'model':partial.state_dict(),'optimizer':optimizer.state_dict(),
                              'next_update':1,'behavior':behavior})
    restored=torch.nn.Linear(2,3,bias=False);restored.load_state_dict(checkpoint['model'])
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.02)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    for indices in batches[checkpoint['next_update']:]:update(restored,restored_optimizer,indices,checkpoint['behavior'])
    assert torch.equal(restored.weight,full.weight)
    assert torch.equal(behavior,checkpoint['behavior'])


def test_clipping_activates_after_a_policy_change_with_fixed_behavior():
    logits=torch.nn.Parameter(torch.zeros(1,3))
    reference=torch.log_softmax(logits.detach(),-1)
    old=reference[:,0].clone()
    optimizer=torch.optim.SGD([logits],lr=2.)
    loss,*_=clipped_loss(logits,reference,torch.tensor([0]),torch.ones(1),old,
                         total_trajectories=1,horizon=1,kl_beta=0,entropy_beta=0)
    loss.backward();optimizer.step()
    _,_,_,metrics=clipped_loss(logits,reference,torch.tensor([0]),torch.ones(1),old,
                               total_trajectories=1,horizon=1,kl_beta=0,entropy_beta=0)
    assert metrics['ratios'].item()>1.2
    assert metrics['surrogate_clipped'].item()


def test_unequal_rank_gradient_accumulation_matches_global_minibatch():
    torch.manual_seed(42)
    logits=torch.randn(64,3,requires_grad=True)
    actions=torch.arange(64)%3
    old=torch.log_softmax(logits.detach(),-1).gather(-1,actions[:,None]).squeeze(-1)+torch.linspace(-.5,.5,64)
    reference=torch.log_softmax(torch.randn(64,3),-1)
    advantages=torch.linspace(-2,2,64)
    whole=clipped_loss(logits,reference,actions,advantages,old,total_trajectories=64)[0]
    # Exactly what seven rank-local backward calls and the averaged reducer do.
    distributed=sum(clipped_loss(logits[r::7],reference[r::7],actions[r::7],advantages[r::7],old[r::7],
                                  total_trajectories=64)[0]*7 for r in range(7))/7
    assert torch.allclose(torch.autograd.grad(whole,logits,retain_graph=True)[0],
                          torch.autograd.grad(distributed,logits)[0])


def test_behavior_and_batch_hashes_survive_resume_and_reject_mutation(tmp_path):
    for _,name in BATCH_FILES:(tmp_path/name).write_text(json.dumps({'original':name}))
    identity=batch_identity(tmp_path)
    cursor={**identity,'round':0,'next_update':1}
    assert batch_identity(tmp_path,cursor)==identity
    (tmp_path/'behavior.json').write_text('refreshed behavior log-probabilities')
    with pytest.raises(ValueError,match='immutable'):
        batch_identity(tmp_path,cursor)
