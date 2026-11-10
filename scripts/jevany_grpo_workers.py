"""GRPO learner with the unchanged v10 controller collection and replay kernels."""
from __future__ import annotations

import copy
import faulthandler
import signal
import json
import os
import queue
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from stream_gemma.v24_memory import normalize_memory_content

from benchmark_qwen35_2b_training import (
    batched_logits, chunk_conv1d_batches, collate, encode_batch_vision, packed_nccl_sync,
    predecode_batch,
)
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import ACTIONS, PointerHead, decision_messages, option_text
from pace_clm.qwen_controller_runtime import load_2b_controller, MODEL_NAME, MODEL_REVISION
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.rl_reference import load_exact_reference
from pace_clm.rl import categorical_loss, observation, pad_inputs, parity_metrics, ready_collection


from jevany_rl_workers import stamp, memory
from pace_clm.grpo import clipped_loss, sha
from pace_clm.grpo_vision import FrozenVisionBank

def prepare_chunks(rows, frames, runtime, size, pool):
    """One CPU batch ahead; no CUDA work runs in the prefetch thread."""
    if not hasattr(frames, 'rl_stats'):
        frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
    def prepare(chunk):
        started = time.monotonic()
        prefetched = None
        if hasattr(frames, 'decode_pool'):
            images = {key: image for key,(image,_) in frames.frames.items()}
            prefetched = predecode_batch(runtime, chunk, frames.decode_pool, images)
        samples = [frames.prepare(r, prefetched=prefetched, cpu_only=True) for r in chunk]
        inputs, visions, lengths = collate(samples, runtime)
        specified = {r['padded_length'] for r in chunk if r.get('padded_length')}
        if len(specified) > 1:
            raise ValueError('replay microbatch mixes collection padding lengths')
        bucket = getattr(frames, 'token_bucket', 512)
        target = next(iter(specified)) if specified else (max(lengths)//bucket+1)*bucket
        physical = getattr(frames, 'physical_batch', 0)
        if physical:
            cap = min(physical, max(1,getattr(frames,'token_budget',262144)//target))
            saved = {r['physical_batch'] for r in chunk if 'physical_batch' in r}
            if len(saved)>1:raise ValueError('replay mixes physical batch shapes')
            if saved:
                physical=next(iter(saved))
                if physical>cap:raise ValueError('saved collection shape exceeds replay token budget')
            else:
                shapes=sorted({b for b in (4,8,16,20,24,cap) if b<=cap})
                physical=next((b for b in shapes if b>=len(samples)),0)
            if len(samples)>physical:
                raise ValueError('real rows exceed the recorded physical batch shape')
            # Match both dimensions per observation. Filler rows get no readout/loss.
            filled=samples+[samples[0]]*(physical-len(samples))
            inputs,visions,_=collate(filled,runtime)
        packed = pad_inputs(inputs,target,runtime.tokenizer.pad_token_id), visions, lengths
        frames.rl_stats['cpu_prepare_seconds'] += time.monotonic()-started
        return samples, packed
    chunks = []
    start = 0
    while start < len(rows):
        padded = rows[start].get('padded_length')
        stop = start+1
        limit = min(size, max(1,getattr(frames,'token_budget',262144)//padded),
                    rows[start].get('physical_batch',size)) if padded else size
        while (stop < len(rows) and stop-start < limit and rows[stop].get('padded_length') == padded
               and rows[stop].get('physical_batch') == rows[start].get('physical_batch')):
            stop += 1
        chunks.append((start, rows[start:stop]))
        start = stop
    pending = pool.submit(prepare,chunks[0][1]) if chunks else None
    for index,(start,chunk) in enumerate(chunks):
        started = time.monotonic()
        samples, packed = pending.result()
        frames.rl_stats['prepare_wait_seconds'] += time.monotonic()-started
        pending = pool.submit(prepare,chunks[index+1][1]) if index+1 < len(chunks) else None
        started = stamp()
        frames.frozen_vision_bank.restore(chunk, samples, runtime.device)
        encode_batch_vision(runtime, samples, 8)
        frames.frozen_vision_bank.capture(chunk, samples)
        frames.rl_stats['vision_seconds'] += stamp()-started
        inputs, visions, lengths = packed
        packed = ({k:v.to(runtime.device) for k,v in inputs.items()}, visions, lengths)
        frames.rl_stats.setdefault('microbatch_shapes',[]).append(list(packed[0]['input_ids'].shape))
        frames.rl_stats.setdefault('real_batch_sizes',[]).append(len(samples))
        frames.rl_stats['real_tokens'] = frames.rl_stats.get('real_tokens',0)+sum(lengths)
        frames.rl_stats['physical_tokens'] = frames.rl_stats.get('physical_tokens',0)+packed[0]['input_ids'].numel()
        yield start, samples, packed


def controller(rank, cfg, commands, events, requests, answers):
    try:
        trace_file=(Path(cfg['output'])/f'controller_stack.rank{rank}.log').open('a')
        faulthandler.register(signal.SIGUSR1,file=trace_file,all_threads=True)
        os.environ['LOCAL_RANK'] = '0'
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        random.seed(cfg['seed']+rank)
        torch.manual_seed(cfg['seed'])
        dist.init_process_group('nccl', init_method=cfg['init_method'], rank=rank,
                                world_size=cfg['learners'], timeout=timedelta(minutes=20))
        collective_started=time.monotonic()
        probe=torch.ones(63570432,device='cuda',dtype=torch.float32)
        dist.all_reduce(probe)
        if not bool((probe==cfg['learners']).all()):
            raise RuntimeError('learner gradient-sized NCCL preflight failed')
        torch.cuda.synchronize()
        collective_seconds=time.monotonic()-collective_started
        del probe
        initial = Path(cfg['checkpoint'])
        restore = Path(cfg['resume']) if cfg.get('resume') else initial
        runtime, _ = load_2b_controller(checkpoint=str(restore/'controller_adapter'))
        install_cached_vision_inputs(runtime)
        reference_tensors = load_exact_reference(runtime.model, initial/'controller_adapter')
        head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
        head.load_state_dict(torch.load(restore/'pointer_head.pt', map_location='cpu', weights_only=True))
        reference_head = copy.deepcopy(head).requires_grad_(False).eval()
        reference_head.load_state_dict(torch.load(initial/'pointer_head.pt', map_location='cpu', weights_only=True))
        runtime.model.set_adapter('default')
        for name, p in runtime.model.named_parameters():
            p.requires_grad_('.default.' in name and ('lora_A' in name or 'lora_B' in name))
        for module in runtime.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        chunk_conv1d_batches(runtime.model, 128)
        runtime.model.gradient_checkpointing_enable()
        runtime.model.enable_input_require_grads()
        lora = [p for p in runtime.model.parameters() if p.requires_grad]
        parameters = list(head.parameters()) + lora
        policy_version = 0
        optimizer_steps = 0
        cursor = None
        optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': cfg['head_lr']},
                                       {'params': lora, 'lr': cfg['lora_lr']}], weight_decay=.01)
        if cfg.get('resume'):
            saved_state = torch.load(restore/'training_state.pt', map_location=runtime.device,
                                     weights_only=False)
            optimizer.load_state_dict(saved_state['optimizer'])
            policy_version = saved_state['step']
            optimizer_steps = saved_state['optimizer_steps']
            cursor = saved_state['cursor']
            if optimizer_steps != cfg['resume_progress']['optimizer_steps'] or cursor != cfg['resume_progress']['cursor']:
                raise RuntimeError('optimizer/pass cursor restore mismatch')
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual = get_peft_model_state_dict(runtime.model, adapter_name='default')
            saved = load_file(str(restore/'controller_adapter/adapter_model.safetensors'))
            if set(actual) != set(saved) or any(not torch.equal(actual[k].cpu(), saved[k].to(actual[k].dtype)) for k in actual):
                raise RuntimeError('adapter resume mismatch')
            saved_head = torch.load(restore/'pointer_head.pt',map_location='cpu',weights_only=True)
            if set(head.state_dict()) != set(saved_head) or any(not torch.equal(value.cpu(),saved_head[key]) for key,value in head.state_dict().items()):
                raise RuntimeError('pointer resume mismatch')
            rng = torch.load(restore/f'rng.rank{rank}.pt', map_location='cpu', weights_only=False)
            random.setstate(rng['python'])
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'])
        frames = EpisodeFrames(runtime, response_blind=False, cache_frames=cfg['frame_cache'], defer_vision=True)
        frames.decode_pool = ThreadPoolExecutor(max_workers=4)
        frames.token_bucket = cfg['token_bucket']
        frames.physical_batch = cfg['physical_batch']
        frames.token_budget = cfg['replay_token_budget']
        frames.frozen_vision_bank = FrozenVisionBank()
        vision_round = None

        def restore_vision(round_index):
            nonlocal vision_round
            batch = Path(cfg['output'])/f'batch-{round_index:06d}'
            manifest = json.loads((batch/'vision.json').read_text())
            record = manifest[str(rank)]
            path = batch/record['file']
            if sha(path) != record['sha256']:
                raise RuntimeError('saved behavior vision cache changed')
            if vision_round != round_index or not frames.frozen_vision_bank.replay:
                frames.frozen_vision_bank.load(path, runtime.device)
                vision_round = round_index
        pool = ThreadPoolExecutor(max_workers=1)
        events.put(('ready', rank, {'lora_parameters': sum(p.numel() for p in lora),
                                    'pointer_parameters': sum(p.numel() for p in head.parameters()),
                                    'exact_reference_tensors':reference_tensors,
                                    'nccl_preflight_seconds':collective_seconds,
                                    'restored_step': policy_version, 'restored_optimizer_steps': optimizer_steps, 'restored_cursor': cursor}))
        while True:
            command, payload = commands.get()
            if command == 'stop':
                break
            if command == 'rollout':
                frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
                torch.cuda.reset_peak_memory_stats()
                runtime.model.eval()
                version, assignments = payload
                frames.frozen_vision_bank = FrozenVisionBank()
                vision_round = version
                if version != policy_version:
                    raise RuntimeError('rollout policy version is stale')
                states = []
                for trajectory_id, spec in assignments:
                    rng = torch.Generator(device=runtime.device)
                    rng.manual_seed(cfg['seed'] + version*1000003 + trajectory_id*101)
                    states.append({'id': trajectory_id, 'spec': spec, 'history': list(spec['prefix_turns']),
                                   'frame': spec['rollout_start'], 'trace': [], 'pending': False, 'rng': rng})
                pending = {}
                started = stamp()
                actor_seconds = answer_wait = 0.

                def accept(answer):
                    key, text, stats = answer
                    state, row, action, logp = pending.pop(key)
                    if not text.strip():
                        raise RuntimeError('empty frozen-generator answer')
                    finish(state, row, action, logp, text, stats)

                def finish(state, row, action, logp, text='', stats=None):
                    tag = ('</Silence>', '</Standby>', '</Response>')[action]
                    turn = {'frame_index': state['frame'], 'ts': row['turns'][-1]['ts'],
                            'emit': tag+(' '+text if text else ''),
                            'state': ('Silence', 'Standby', 'Response')[action]}
                    if action == 2:
                        turn['memory_content'] = normalize_memory_content(text)
                    state['trace'].append({'row': row, 'action': action, 'old_logp': logp,
                                           'answer': text, 'generator_stats': stats,
                                           'policy_version': version})
                    state['history'].append(turn)
                    state['frame'] += 1
                    state['pending'] = False

                while any(s['frame'] < s['spec']['rollout_stop'] for s in states):
                    while True:
                        try:
                            accept(answers.get_nowait())
                        except queue.Empty:
                            break
                    ready,wait,finished=ready_collection(states,
                        max_lead=cfg['rollout_max_lead'],min_ready_fraction=cfg['collection_min_ready'],
                        batch_size=cfg['collection_batch'])
                    if finished:break
                    if wait or not ready:
                        waiting = time.monotonic()
                        accept(answers.get(timeout=1200))
                        answer_wait += time.monotonic()-waiting
                        continue
                    rows = [observation(s['spec'], s['history'], s['frame']) for s in ready]
                    tick = stamp()
                    for offset, samples, packed in prepare_chunks(rows, frames, runtime, cfg['collection_batch'], pool):
                        with torch.no_grad():
                            logits, _ = batched_logits(runtime, head, samples, packed)
                            logps = F.log_softmax(logits.float(), -1)
                            for state, row, logp in zip(ready[offset:offset+len(samples)],
                                                       rows[offset:offset+len(samples)], logps, strict=True):
                                action = int(torch.multinomial(logp.exp(), 1, generator=state['rng']))
                                values = logp.cpu().tolist()
                                row['padded_length'] = packed[0]['input_ids'].shape[1]
                                row['physical_batch'] = packed[0]['input_ids'].shape[0]
                                if action == 2:
                                    key = f'{version}:{state["id"]}:{state["frame"]}'
                                    pending[key] = (state, row, action, values)
                                    state['pending'] = True
                                    requests.put((rank, key, row, time.monotonic()))
                                else:
                                    finish(state, row, action, values)
                    actor_seconds += stamp()-tick
                traces = [{'trajectory_id': s['id'], 'spec': s['spec'], 'steps': s['trace']} for s in states]
                vision_path = Path(cfg['output'])/f'batch-{version:06d}'/f'vision.rank{rank}.pt'
                frames.frozen_vision_bank.save(vision_path)
                events.put(('rollout', rank, {'traces': traces, 'vision_cache': {'file':vision_path.name,'sha256':sha(vision_path),'frames':len(frames.frozen_vision_bank.entries)}, 'seconds': stamp()-started,
                                              'actor_seconds': actor_seconds,
                                              'preparation': dict(frames.rl_stats),
                                              'answer_wait_seconds': answer_wait, **memory()}))
            elif command == 'verify_behavior':
                if optimizer_steps != 4*policy_version or cursor is not None:
                    raise RuntimeError('behavior parity must precede every update in a round')
                steps, total_trajectories, microbatch = payload
                restore_vision(policy_version)
                if any(s['policy_version'] != policy_version or
                       s['behavior_selected_logp'] != s['old_logp'][s['action']] for s in steps):
                    raise RuntimeError('behavior policy identity/log-probability mismatch')
                runtime.model.set_adapter('default')
                runtime.model.eval()
                started = stamp()
                frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
                steps = sorted(steps, key=lambda s: (s['row']['padded_length'], s['row']['physical_batch'], len(s['row']['turns'])))
                maxima = {'all_logp_max': 0., 'probability_max': 0., 'kl_max': 0., 'selected_logp_max': 0.}
                kl_sum = 0.
                for start, samples, packed in prepare_chunks([s['row'] for s in steps], frames, runtime, microbatch, pool):
                    chunk = steps[start:start+len(samples)]
                    with torch.no_grad():
                        logits, _ = batched_logits(runtime, head, samples, packed)
                    old = torch.tensor([s['old_logp'] for s in chunk], device=runtime.device)
                    actions = torch.tensor([s['action'] for s in chunk], device=runtime.device)
                    metrics = parity_metrics(F.log_softmax(logits.float(), -1), old, actions)
                    for key in maxima:
                        maxima[key] = max(maxima[key], metrics[key])
                    kl_sum += metrics['kl_sum']
                    if (metrics['probability_max'] > cfg['parity_probability_tolerance'] or
                            metrics['kl_max'] > cfg['parity_kl_tolerance'] or
                            metrics['selected_logp_max'] > cfg['selected_logp_tolerance']):
                        raise RuntimeError(f'behavior likelihood replay mismatch: {metrics}')
                if steps and kl_sum/len(steps) > cfg['parity_mean_kl_tolerance']:
                    raise RuntimeError('behavior replay mean KL too high')
                events.put(('verify_behavior', rank, {'seconds': stamp()-started, 'decisions': len(steps),
                                                     **maxima, 'kl_mean': kl_sum/max(1,len(steps))}))
            elif command == 'learn_grpo':
                steps, total_trajectories, microbatch = payload['steps'], payload['total_trajectories'], payload['microbatch']
                version, update = payload['round'], payload['update_index']
                restore_vision(version)
                if (version != policy_version or optimizer_steps != 4*version+update or
                        any(s['policy_version'] != version or s['behavior_selected_logp'] != s['old_logp'][s['action']] for s in steps)):
                    raise RuntimeError('stale or repeated GRPO minibatch/behavior policy')
                if update and (cursor is None or cursor['next_update'] != update):
                    raise RuntimeError('GRPO pass/minibatch cursor mismatch')
                frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                started = stamp()
                steps = sorted(steps, key=lambda s: (s['row']['padded_length'], s['row']['physical_batch'], len(s['row']['turns'])))
                reference_seconds = actor_seconds = kl_sum = entropy_sum = loss_sum = 0.
                ratio_sum = ratio_square_sum = outside = effective = 0.
                ratio_min, ratio_max = float('inf'), 0.
                for start, samples, packed in prepare_chunks([s['row'] for s in steps], frames, runtime, microbatch, pool):
                    chunk = steps[start:start+len(samples)]
                    tick = stamp()
                    runtime.model.set_adapter('reference')
                    runtime.model.eval()
                    with torch.no_grad():
                        ref_logits, _ = batched_logits(runtime, reference_head, samples, packed)
                        ref_logp = F.log_softmax(ref_logits.float(), -1)
                    runtime.model.set_adapter('default')
                    runtime.model.train()
                    runtime._conditional_model().model.visual.eval()
                    reference_seconds += stamp()-tick
                    tick = stamp()
                    logits, _ = batched_logits(runtime, head, samples, packed)
                    actions = torch.tensor([s['action'] for s in chunk], device=runtime.device)
                    advantage = torch.tensor([s['advantage'] for s in chunk], device=runtime.device)
                    behavior = torch.tensor([s['behavior_selected_logp'] for s in chunk], device=runtime.device)
                    loss, kl, entropy, metrics = clipped_loss(logits, ref_logp, actions, advantage, behavior,
                                                            total_trajectories=total_trajectories)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite clipped GRPO loss')
                    if update == 0 and bool(((metrics['ratios']-1).abs()>.05).any()):
                        raise RuntimeError(f'initial behavior ratios differ before optimizer step: '
                                           f"min={float(metrics['ratios'].min())}, max={float(metrics['ratios'].max())}")
                    with runtime._cached_image_features(packed[1]):
                        (loss*cfg['learners']).backward()
                    ratios = metrics['ratios']
                    ratio_sum += float(ratios.sum())
                    ratio_square_sum += float(ratios.square().sum())
                    ratio_min, ratio_max = min(ratio_min,float(ratios.min())), max(ratio_max,float(ratios.max()))
                    outside += float(metrics['outside_clip'].sum())
                    effective += float(metrics['surrogate_clipped'].sum())
                    kl_sum += float(kl.sum())
                    entropy_sum += float(entropy.sum())
                    loss_sum += float(loss.detach())
                    actor_seconds += stamp()-tick
                head_norm = sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
                lora_norm = sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None)**.5
                if any(p.grad is not None for n,p in runtime.model.named_parameters() if '.default.' not in n):
                    raise RuntimeError('gradient on frozen model/reference parameter')
                tick = stamp()
                packed_nccl_sync(parameters, cfg['learners'])
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
                if not torch.isfinite(torch.tensor(grad_norm)):
                    raise RuntimeError('nonfinite gradients')
                optimizer.step()
                optimizer_steps += 1
                if update == 3:
                    policy_version += 1
                    cursor = None
                else:
                    cursor = {**payload['batch_identity'], 'round': version, 'next_update': update+1}
                optimizer.zero_grad(set_to_none=True)
                events.put(('learn_grpo', rank, {'seconds': stamp()-started, 'reference_seconds': reference_seconds,
                    'actor_seconds': actor_seconds, 'sync_optimizer_seconds': stamp()-tick,
                    'decisions': len(steps), 'kl_sum': kl_sum, 'entropy_sum': entropy_sum, 'loss': loss_sum,
                    'head_grad_norm': head_norm, 'lora_grad_norm': lora_norm, 'grad_norm': grad_norm,
                    'ratio_sum': ratio_sum, 'ratio_square_sum': ratio_square_sum, 'ratio_min': ratio_min if steps else 1.,
                    'ratio_max': ratio_max if steps else 1., 'outside_clip_count': outside, 'surrogate_clip_count': effective,
                    'optimizer_steps': optimizer_steps, 'completed_rounds': policy_version,
                    'pass': update//2, 'minibatch': update%2, **memory()}))
            elif command == 'save':
                path, progress = Path(payload['path']), payload['progress']
                if (progress['step'] != policy_version or progress['optimizer_steps'] != optimizer_steps or
                        progress['cursor'] != cursor):
                    raise RuntimeError('checkpoint optimizer/collection/pass cursor mismatch')
                path.mkdir(parents=True, exist_ok=True)
                torch.save({'python': random.getstate(), 'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state()}, path/f'rng.rank{rank}.pt')
                if rank == 0:
                    runtime.model.save_pretrained(path/'controller_adapter', selected_adapters=['default'])
                    torch.save(head.state_dict(), path/'pointer_head.pt')
                    torch.save({'optimizer': optimizer.state_dict(), 'step': policy_version,
                                'optimizer_steps': optimizer_steps, 'cursor': cursor}, path/'training_state.pt')
                    (path/'model_identity.json').write_text(json.dumps({
                        'model_name':MODEL_NAME, 'model_revision':MODEL_REVISION,
                        'controller_adapter':'controller_adapter', 'pointer_head':'pointer_head.pt'},indent=2)+'\n')
                    (path/'rl_state.json').write_text(json.dumps({
                        'policy_version':policy_version, 'completed_batch_id':policy_version-1,
                        'sampler_cursor':policy_version*cfg['starts'], 'seed':cfg['seed'],
                        'optimizer_steps': optimizer_steps, 'cursor': cursor},indent=2)+'\n')
                dist.barrier()
                events.put(('save', rank, {'step': policy_version, 'optimizer_steps': optimizer_steps}))
        pool.shutdown()
        frames.decode_pool.shutdown()
        dist.destroy_process_group()
    except BaseException:
        events.put(('error', rank, traceback.format_exc()))
        raise
