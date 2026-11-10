"""Categorical clipped GRPO contracts, schedules and committed-update recovery."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F


def group_advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError('need at least two finite rewards')
    if max(rewards) == min(rewards):
        return [0.] * len(rewards)
    mean = math.fsum(rewards)/len(rewards)
    std = math.sqrt(math.fsum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+epsilon) for r in rewards]


def clipped_loss(logits, reference_log_probs, actions, advantages, behavior_log_probs,
                 *, total_trajectories, horizon=32, epsilon=.2, kl_beta=.02, entropy_beta=.001):
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.gather(-1, actions[:, None]).squeeze(-1)
    behavior = behavior_log_probs.detach().float()
    advantage = advantages.detach().float()
    ratio = (selected-behavior).exp()
    if not all(torch.isfinite(t).all() for t in (logp, reference_log_probs, ratio, advantage)):
        raise ValueError('nonfinite GRPO likelihood, ratio, reference or advantage')
    unclipped = ratio*advantage
    bounded = ratio.clamp(1-epsilon, 1+epsilon)*advantage
    probability = logp.exp()
    kl = (probability*(logp-reference_log_probs.detach().float())).sum(-1)
    entropy = -(probability*logp).sum(-1)
    loss = (-torch.minimum(unclipped, bounded)+kl_beta*kl-entropy_beta*entropy).sum()
    loss = loss/(total_trajectories*horizon)
    return loss, kl.detach(), entropy.detach(), {
        'ratios': ratio.detach(),
        'outside_clip': ((ratio < 1-epsilon) | (ratio > 1+epsilon)).detach(),
        'surrogate_clipped': (bounded < unclipped).detach(),
    }


def update_schedule(trajectory_ids, seed, collection_round):
    ids = list(trajectory_ids)
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError('GRPO requires 128 unique trajectories')
    schedule = []
    for actor_pass in range(2):
        order = list(ids)
        random.Random(seed+collection_round*1000003+actor_pass*10007).shuffle(order)
        for minibatch in range(2):
            schedule.append({'update_index': actor_pass*2+minibatch, 'pass': actor_pass,
                             'minibatch': minibatch, 'trajectory_ids': order[minibatch*64:(minibatch+1)*64]})
    return schedule


def partition_trajectories(trajectories, ids, world=7):
    lookup = {t['trajectory_id']: t for t in trajectories}
    if len(ids) != 64 or len(set(ids)) != 64 or not set(ids) <= lookup.keys():
        raise ValueError('invalid global 64-trajectory minibatch')
    ranks = [[] for _ in range(world)]
    for identity in ids:
        # Keep each exact behavior observation on its original collection rank.
        # The global 64-trajectory shuffle is unchanged; only physical placement
        # is stable, preserving rank-local frozen vision features.
        ranks[identity % world].extend(lookup[identity]['steps'])
    return ranks


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


BATCH_FILES = (('rollouts_sha256','rollouts.json'),('scored_sha256','scored.json'),
               ('behavior_sha256','behavior.json'),('schedule_sha256','schedule.json'),
               ('parity_sha256','parity.json'),('vision_sha256','vision.json'))


def batch_identity(directory, expected=None):
    identity={key:sha(Path(directory)/name) for key,name in BATCH_FILES}
    if expected is not None and any(expected[key]!=value for key,value in identity.items()):
        raise ValueError('immutable behavior, observations, scores, shuffle or parity changed')
    return identity


def validate_progress(progress):
    rounds, steps, cursor = progress['step'], progress['optimizer_steps'], progress['cursor']
    if cursor is None:
        if steps != 4*rounds:
            raise ValueError('completed-round optimizer counter mismatch')
    elif (cursor['round'] != rounds or not 1 <= cursor['next_update'] <= 3 or
          steps != 4*rounds+cursor['next_update']):
        raise ValueError('in-flight pass/minibatch optimizer cursor mismatch')
    return progress


def recover_progress(run):
    """Adopt a fully committed checkpoint if its atomic latest-pointer write was interrupted.

    A renamed checkpoint plus commit manifest is durable. An incomplete staging
    directory is never adopted. Therefore a saved optimizer update is not repeated.
    """
    run = Path(run)
    latest = run/'latest.json'
    progress = json.loads(latest.read_text()) if latest.exists() else {
        'step': 0, 'optimizer_steps': 0, 'cursor': None, 'checkpoint': None}
    validate_progress(progress)
    if progress['checkpoint'] is not None:
        current = Path(progress['checkpoint'])
        commit = json.loads((current/'commit.json').read_text())
        if commit['progress'] != progress:
            raise ValueError('latest pointer differs from committed optimizer cursor')
        for name, expected in commit['files'].items():
            if sha(current/name) != expected:
                raise ValueError(f'corrupt current checkpoint: {name}')
    while True:
        candidate = run/'optimizer_checkpoints'/f"update-{progress['optimizer_steps']+1:06d}"
        manifest_path = candidate/'commit.json'
        if not manifest_path.exists():
            break
        manifest = json.loads(manifest_path.read_text())
        if manifest['parent_checkpoint'] != progress['checkpoint']:
            raise ValueError('checkpoint commit ancestry differs')
        next_progress = validate_progress(manifest['progress'])
        if (next_progress['optimizer_steps'] != progress['optimizer_steps']+1 or
                next_progress['checkpoint'] != str(candidate.resolve())):
            raise ValueError('checkpoint commit counter/path mismatch')
        for name, expected in manifest['files'].items():
            if sha(candidate/name) != expected:
                raise ValueError(f'incomplete/corrupt committed checkpoint: {name}')
        progress = next_progress
        atomic_json(latest, progress)
    return progress
