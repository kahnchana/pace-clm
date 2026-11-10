"""Training-only supervised anchor: globally normalized loss and fixed calibration."""
import math
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def anchor_loss(logits, labels, global_examples):
    if global_examples<=0 or len(labels)>global_examples:
        raise ValueError('invalid global supervised batch size')
    return F.cross_entropy(logits.float(),labels,reduction='sum')/global_examples


def calibrate_anchor(rl_norm, ce_norm, target_fraction=.15):
    if not all(math.isfinite(x) and x>0 for x in (rl_norm,ce_norm,target_fraction)):
        raise ValueError('anchor calibration requires finite positive gradient norms')
    return target_fraction*rl_norm/ce_norm


def anchor_partition(rows, collection_round, rank, world, per_round=126):
    if not 0<=rank<world or per_round%world or collection_round<0:
        raise ValueError('invalid supervised replay partition')
    selected=rows[collection_round*per_round:(collection_round+1)*per_round]
    if len(selected)!=per_round:raise ValueError('supervised replay schedule exhausted')
    return selected[rank::world]


def verify_committed_checkpoint(path, step):
    path=Path(path)
    commit=json.loads((path/'commit.json').read_text())
    required={'pointer_head.pt','training_state.pt','rl_state.json',
              'controller_adapter/adapter_model.safetensors'}
    if commit['step']!=step or not required<=commit['files'].keys():
        raise ValueError('checkpoint step or required files invalid')
    for name,expected in commit['files'].items():
        target=path/name
        if not target.resolve().is_relative_to(path.resolve()):raise ValueError('invalid checkpoint file path')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'committed checkpoint file changed: {name}')
    return commit
