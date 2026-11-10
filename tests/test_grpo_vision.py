from types import SimpleNamespace

import pytest
import torch

from pace_clm.grpo_vision import FrozenVisionBank


def sample(path='video.mp4',index=2,embeds=None):
    frame=SimpleNamespace(grid=torch.tensor([[1,2,4]]),num_tokens=2,image_embeds=embeds)
    return {'video_path':path,'turns':[{'frame_index':index}]}, (None,[frame],None,None)


def test_exact_bfloat16_features_restore_after_rebatching_and_disk_resume(tmp_path):
    torch.manual_seed(42)
    original=torch.randn(2,3,dtype=torch.bfloat16)
    row,observation=sample(embeds=original)
    bank=FrozenVisionBank()
    bank.capture([row],[observation])
    path=tmp_path/'vision.pt'
    bank.save(path)
    # Simulate another encoding batch producing slightly different BF16 values.
    observation[1][0].image_embeds=original+1
    bank.replay=True
    bank.restore([row],[observation],'cpu')
    assert torch.equal(observation[1][0].image_embeds,original)
    fresh=FrozenVisionBank()
    fresh.load(path,'cpu')
    new_row,new_sample=sample(embeds=torch.zeros_like(original))
    fresh.restore([new_row],[new_sample],'cpu')
    assert torch.equal(new_sample[1][0].image_embeds,original)
    assert new_sample[1][0].image_embeds.dtype==torch.bfloat16
    assert not new_sample[1][0].image_embeds.requires_grad


def test_replay_cannot_silently_reencode_missing_or_changed_vision():
    row,observation=sample(embeds=torch.ones(2,3))
    bank=FrozenVisionBank();bank.capture([row],[observation]);bank.replay=True
    different_row,different=sample(path='other-video.mp4',embeds=torch.ones(2,3))
    with pytest.raises(ValueError,match='missing'):
        bank.restore([different_row],[different],'cpu')
    observation[1][0].grid=torch.tensor([[1,4,2]])
    with pytest.raises(ValueError,match='grid'):
        bank.restore([row],[observation],'cpu')


def test_behavior_bank_is_independent_of_mutated_encoder_buffers():
    original=torch.randn(2,3)
    row,observation=sample(embeds=original)
    bank=FrozenVisionBank();bank.capture([row],[observation])
    expected=original.clone()
    original.add_(10)
    bank.restore([row],[observation],'cpu')
    assert torch.equal(observation[1][0].image_embeds,expected)
