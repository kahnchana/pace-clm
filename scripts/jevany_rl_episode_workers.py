"""Resident controller/learner and dynamically batched frozen-generator workers."""
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
from pace_clm.rl_anchor import anchor_loss, calibrate_anchor
from pace_clm.rl_episode import ANCHOR_COEFFICIENT, replay_partition, ordered_replay
from pace_clm.rl import categorical_loss, observation, pad_inputs, parity_metrics, ready_collection


def stamp():
    torch.cuda.synchronize()
    return time.monotonic()


def memory():
    return {'allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
            'reserved_gib': torch.cuda.max_memory_reserved()/1024**3}


def fp32_depthwise_convolutions(model):
    """Keep BF16 inputs/outputs while accumulating the short convolution in FP32."""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            def forward(inputs, module=module):
                bias = module.bias.float() if module.bias is not None else None
                return module._conv_forward(inputs.float(),module.weight.float(),bias).to(inputs.dtype)
            module.forward = forward


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
               and rows[stop].get('collection_batch') == rows[start].get('collection_batch')
               and rows[stop].get('physical_batch') == rows[start].get('physical_batch')):
            stop += 1
        chunks.append((start, rows[start:stop]))
        if rows[start].get('collection_size',stop-start)!=stop-start:
            raise ValueError('replay must preserve the full collected batch layout')
        start = stop
    pending = pool.submit(prepare,chunks[0][1]) if chunks else None
    for index,(start,chunk) in enumerate(chunks):
        started = time.monotonic()
        samples, packed = pending.result()
        frames.rl_stats['prepare_wait_seconds'] += time.monotonic()-started
        pending = pool.submit(prepare,chunks[index+1][1]) if index+1 < len(chunks) else None
        started = stamp()
        encode_batch_vision(runtime, samples, 8)
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
        anchor_calibration = {'coefficient':ANCHOR_COEFFICIENT, 'source':'frozen original anchored clip RLOO coefficient; no recalibration'}
        optimizer = torch.optim.AdamW([{'params': head.parameters(), 'lr': cfg['head_lr']},
                                       {'params': lora, 'lr': cfg['lora_lr']}], weight_decay=.01)
        if cfg.get('resume'):
            saved_state = torch.load(restore/'training_state.pt', map_location=runtime.device,
                                     weights_only=False)
            optimizer.load_state_dict(saved_state['optimizer'])
            policy_version = saved_state['step']
            anchor_calibration=saved_state['anchor_calibration']
            if anchor_calibration!=json.loads((Path(cfg['output'])/'anchor-calibration.json').read_text()):
                raise RuntimeError('anchor coefficient restore mismatch')
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual = get_peft_model_state_dict(runtime.model, adapter_name='default')
            saved = load_file(str(restore/'controller_adapter/adapter_model.safetensors'))
            if set(actual) != set(saved) or any(not torch.equal(actual[k].cpu(), saved[k].to(actual[k].dtype)) for k in actual):
                raise RuntimeError('adapter resume mismatch')
            saved_head=torch.load(restore/'pointer_head.pt',map_location='cpu',weights_only=True)
            if any(not torch.equal(value.cpu(),saved_head[key]) for key,value in head.state_dict().items()):
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
        anchor_rows=[json.loads(line) for line in Path(cfg['anchor_manifest']).read_text().splitlines()]
        anchor_frames=EpisodeFrames(runtime,response_blind=False,cache_frames=cfg['frame_cache'],defer_vision=True)
        anchor_frames.decode_pool=ThreadPoolExecutor(max_workers=4)
        anchor_frames.token_bucket=cfg['token_bucket']
        anchor_frames.physical_batch=cfg['physical_batch']
        anchor_frames.token_budget=cfg['replay_token_budget']
        pool = ThreadPoolExecutor(max_workers=1)
        events.put(('ready', rank, {'lora_parameters': sum(p.numel() for p in lora),
                                    'pointer_parameters': sum(p.numel() for p in head.parameters()),
                                    'exact_reference_tensors':reference_tensors,
                                    'nccl_preflight_seconds':collective_seconds,
                                    'anchor_calibration':anchor_calibration,
                                    'restored_step': saved_state['step'] if cfg.get('resume') else 0}))
        while True:
            command, payload = commands.get()
            if command == 'stop':
                break
            if command == 'rollout':
                frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
                torch.cuda.reset_peak_memory_stats()
                runtime.model.eval()
                version, assignments = payload
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
                collection_batch = 0

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
                            for position,(state, row, logp) in enumerate(zip(ready[offset:offset+len(samples)],
                                                       rows[offset:offset+len(samples)], logps, strict=True)):
                                action = int(torch.multinomial(logp.exp(), 1, generator=state['rng']))
                                values = logp.cpu().tolist()
                                row['padded_length'] = packed[0]['input_ids'].shape[1]
                                row['physical_batch'] = packed[0]['input_ids'].shape[0]
                                row['collection_batch'] = collection_batch
                                row['collection_position'] = position
                                row['collection_size'] = len(samples)
                                if action == 2:
                                    key = f'{version}:{state["id"]}:{state["frame"]}'
                                    pending[key] = (state, row, action, values)
                                    state['pending'] = True
                                    requests.put((rank, key, row, time.monotonic()))
                                else:
                                    finish(state, row, action, values)
                            collection_batch += 1
                    actor_seconds += stamp()-tick
                traces = [{'trajectory_id': s['id'], 'spec': s['spec'], 'steps': s['trace']} for s in states]
                events.put(('rollout', rank, {'traces': traces, 'seconds': stamp()-started,
                                              'actor_seconds': actor_seconds,
                                              'preparation': dict(frames.rl_stats),
                                              'answer_wait_seconds': answer_wait, **memory()}))
            elif command in ('learn', 'benchmark'):
                frames.rl_stats = {'cpu_prepare_seconds': 0., 'prepare_wait_seconds': 0., 'vision_seconds': 0.}
                steps, total_trajectories, microbatch = payload
                if any(s['policy_version'] != policy_version for s in steps):
                    raise RuntimeError('attempted replay of a stale/completed policy batch')
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                started = stamp()
                steps = ordered_replay(steps)
                rows = [s['row'] for s in steps]
                parity_max = kl_sum = entropy_sum = loss_sum = 0.
                parity_probability_max = parity_kl_max = parity_kl_sum = parity_selected_max = 0.
                reference_seconds = actor_seconds = 0.
                for start, samples, packed in prepare_chunks(rows, frames, runtime, microbatch, pool):
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
                    old = torch.tensor([s['old_logp'] for s in chunk], device=runtime.device)
                    actions = torch.tensor([s['action'] for s in chunk], device=runtime.device)
                    parity = parity_metrics(F.log_softmax(logits.float(),-1),old,actions)
                    parity_max = max(parity_max,parity['all_logp_max'])
                    parity_probability_max = max(parity_probability_max,parity['probability_max'])
                    parity_kl_max = max(parity_kl_max,parity['kl_max'])
                    parity_kl_sum += parity['kl_sum']
                    parity_selected_max = max(parity_selected_max,parity['selected_logp_max'])
                    if (parity['probability_max'] > cfg['parity_probability_tolerance'] or
                        parity['kl_max'] > cfg['parity_kl_tolerance'] or
                        parity['selected_logp_max'] > cfg['selected_logp_tolerance']):
                        raise RuntimeError(f'rollout/replay distribution mismatch: {parity}')
                    adv = torch.tensor([s['advantage'] for s in chunk], device=runtime.device)
                    loss, kl, entropy = categorical_loss(logits, ref_logp, actions, adv,
                                                         total_trajectories=total_trajectories)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite RL loss')
                    # The reducer below averages ranks; loss is globally normalized.
                    with runtime._cached_image_features(packed[1]):
                        (loss*cfg['learners']).backward()
                    kl_sum += float(kl.sum())
                    entropy_sum += float(entropy.sum())
                    loss_sum += float(loss.detach())
                    actor_seconds += stamp()-tick
                tick = stamp()
                if steps and parity_kl_sum/len(steps) > cfg['parity_mean_kl_tolerance']:
                    raise RuntimeError(f'rollout/replay mean KL too high: {parity_kl_sum/len(steps)}')
                head_norm = sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
                lora_norm = sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None)**.5
                if any(p.grad is not None for n,p in runtime.model.named_parameters()
                       if '.default.' not in n):
                    raise RuntimeError('gradient on a frozen parameter')
                gradient_norm_seconds=stamp()-tick
                sync_started=stamp()
                # Calibrate once using synchronized, globally normalized gradients.
                # Later rounds fuse RL + weighted CE locally and use one all-reduce.
                calibrating=anchor_calibration is None
                if calibrating:
                    packed_nccl_sync(parameters,cfg['learners'])
                    rl_gradients=[p.grad.detach().clone() for p in parameters]
                    rl_norm=sum(float(g.float().square().sum()) for g in rl_gradients)**.5
                    optimizer.zero_grad(set_to_none=True)
                anchor_started=stamp()
                anchor_ce=0.;anchor_counts=[0,0,0];anchor_correct=[0,0,0]
                episode_batch=cfg['episode_batches'][policy_version]
                supervised=replay_partition(anchor_rows,episode_batch,rank,cfg['learners'])
                supervised=sorted(supervised,key=lambda r:len(r['turns']))
                runtime.model.set_adapter('default');runtime.model.train()
                runtime._conditional_model().model.visual.eval()
                for start,samples,packed in prepare_chunks(supervised,anchor_frames,runtime,microbatch,pool):
                    chunk=supervised[start:start+len(samples)]
                    logits,_=batched_logits(runtime,head,samples,packed)
                    labels=torch.tensor([ACTIONS.index(r['label']) for r in chunk],device=runtime.device)
                    ce=anchor_loss(logits,labels,episode_batch['anchor_count'])
                    if not torch.isfinite(ce):raise RuntimeError('nonfinite supervised anchor loss')
                    weight=1. if calibrating else anchor_calibration['coefficient']
                    with runtime._cached_image_features(packed[1]):
                        (ce*weight*cfg['learners']).backward()
                    anchor_ce+=float(ce.detach())
                    for label in range(3):
                        anchor_counts[label]+=int((labels==label).sum())
                        anchor_correct[label]+=int(((labels==label)&(logits.argmax(-1)==labels)).sum())
                packed_nccl_sync(parameters, cfg['learners'])
                if calibrating:
                    ce_norm=sum(float(p.grad.float().square().sum()) for p in parameters)**.5
                    coefficient=calibrate_anchor(rl_norm,ce_norm,cfg['anchor_target_fraction'])
                    dot=sum(float((p.grad.float()*g.float()).sum()) for p,g in zip(parameters,rl_gradients,strict=True))
                    anchor_calibration={'coefficient':coefficient,'rl_objective_grad_norm':rl_norm,
                        'unweighted_ce_grad_norm':ce_norm,'target_fraction':cfg['anchor_target_fraction'],
                        'gradient_cosine':dot/(rl_norm*ce_norm),'collection_round':0,
                        'source':'first training round only; frozen before first optimizer update'}
                    for p,g in zip(parameters,rl_gradients,strict=True):p.grad.mul_(coefficient).add_(g)
                    del rl_gradients
                anchor_seconds=stamp()-anchor_started
                gradient_sync_seconds=stamp()-sync_started
                optimizer_started=stamp()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
                if not torch.isfinite(torch.tensor(grad_norm)):
                    raise RuntimeError('nonfinite gradients')
                if command == 'learn':
                    optimizer.step()
                    policy_version += 1
                optimizer.zero_grad(set_to_none=True)
                events.put((command, rank, {'seconds': stamp()-started,
                                           'reference_seconds': reference_seconds,
                                           'actor_seconds': actor_seconds, 'sync_optimizer_seconds': stamp()-tick,
                                           'gradient_norm_seconds':gradient_norm_seconds,
                                           'gradient_sync_seconds':gradient_sync_seconds,
                                           'optimizer_seconds':stamp()-optimizer_started,
                                           'anchor_lambda':anchor_calibration['coefficient'],
                                           'anchor_calibration':anchor_calibration,
                                           'anchor_examples':len(supervised),'anchor_seconds':anchor_seconds,
                                           'anchor_ce_global_contribution':anchor_ce,
                                           'anchor_class_counts':anchor_counts,'anchor_class_correct':anchor_correct,
                                           'decisions': len(steps), 'kl_sum': kl_sum, 'entropy_sum': entropy_sum,
                                           'loss': loss_sum, 'parity_max': parity_max,
                                           'parity_probability_max':parity_probability_max,
                                           'parity_kl_max':parity_kl_max,
                                           'parity_kl_mean':parity_kl_sum/max(1,len(steps)),
                                           'parity_selected_logp_max':parity_selected_max,
                                           'preparation': dict(frames.rl_stats),
                                           'head_grad_norm': head_norm, 'lora_grad_norm': lora_norm,
                                           'grad_norm': grad_norm, **memory()}))
            elif command == 'save':
                path, step = payload
                if step != policy_version:
                    raise RuntimeError('checkpoint step differs from policy version')
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                torch.save({'python': random.getstate(), 'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state()}, path/f'rng.rank{rank}.pt')
                if rank == 0:
                    runtime.model.save_pretrained(path/'controller_adapter', selected_adapters=['default'])
                    torch.save(head.state_dict(), path/'pointer_head.pt')
                    torch.save({'optimizer': optimizer.state_dict(), 'step': step,'anchor_calibration':anchor_calibration}, path/'training_state.pt')
                    (path/'model_identity.json').write_text(json.dumps({
                        'model_name':MODEL_NAME, 'model_revision':MODEL_REVISION,
                        'controller_adapter':'controller_adapter', 'pointer_head':'pointer_head.pt'},indent=2)+'\n')
                    (path/'rl_state.json').write_text(json.dumps({
                        'policy_version':step, 'completed_batch_id':step-1,
                        'sampler_cursor':cfg['episode_batches'][step-1]['groups_after'], 'seed':cfg['seed']},indent=2)+'\n')
                dist.barrier()
                events.put(('save', rank, {'step': step}))
        pool.shutdown()
        frames.decode_pool.shutdown()
        anchor_frames.decode_pool.shutdown()
        dist.destroy_process_group()
    except BaseException:
        events.put(('error', rank, traceback.format_exc()))
        raise


def generator(gpu, cfg, events, requests, answer_queues):
    try:
        trace_file=(Path(cfg['output'])/f'generator_stack.gpu{gpu}.log').open('a')
        faulthandler.register(signal.SIGUSR1,file=trace_file,all_threads=True)
        from stream_gemma.inference.qwen_stream import QwenRuntime
        from pace_clm.experiment import direct_inputs
        os.environ['LOCAL_RANK'] = '0'
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        runtime = QwenRuntime.load(checkpoint=cfg['generator'])
        install_cached_vision_inputs(runtime)
        frames = EpisodeFrames(runtime, response_blind=False, cache_frames=cfg['frame_cache'], defer_vision=True)
        from jevany_rl_judge import TransformersLocalJudge
        judge = TransformersLocalJudge(runtime.device, runtime=runtime)
        judge.calibrate(Path(cfg['output'])/f'judge_calibration.gpu{gpu}.json')
        events.put(('ready', gpu, {'local_judge_calibrated': True}))
        carry = None
        checked_comparison = False
        while True:
            task = carry if carry is not None else requests.get()
            carry = None
            if task is None:
                break
            if task[0] == 'judge':
                _, part, payload, batch_dir = task
                started = stamp()
                results = judge.score_cached(payload, Path(cfg['output'])/f'local-judge.gpu{gpu}.sqlite',
                                              Path(batch_dir)/f'judge.part{part}.jsonl')
                events.put(('judge', part, {'gpu': gpu, 'results': results, 'seconds': stamp()-started}))
                continue
            tasks = [task]
            deadline = time.monotonic()+.015
            while len(tasks) < cfg['generator_batch']:
                try:
                    next_task = requests.get(timeout=max(.0001, deadline-time.monotonic()))
                    if next_task is None:
                        carry = None
                        break
                    tasks.append(next_task)
                except queue.Empty:
                    break
                if time.monotonic() >= deadline:
                    break
            torch.cuda.reset_peak_memory_stats()
            started = stamp()
            inputs_list, visions = [], []
            prefix = runtime.tokenizer('</Response>', add_special_tokens=False, return_tensors='pt')['input_ids']
            for rank, key, row, submitted in tasks:
                sample = frames.prepare(row, cpu_only=True)
                vision = sample[1]
                runtime.encode_frames_features(vision)
                images = [frames.frames[(row['video_path'], t['frame_index'])][0] for t in row['turns']]
                messages = decision_messages(row, images)
                messages[-1]['content'] = [part for part in messages[-1]['content']
                                           if part.get('text') != option_text()]
                inputs = direct_inputs(runtime, messages, vision)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], prefix.to(runtime.device)], -1)
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],
                                                         torch.zeros_like(prefix, device=runtime.device)], -1)
                inputs_list.append(inputs)
                visions.extend(vision)
            longest = (max(v['input_ids'].shape[1] for v in inputs_list)//cfg['token_bucket']+1)*cfg['token_bucket']
            packed = {}
            for name, fill in (('input_ids', runtime.tokenizer.pad_token_id),
                               ('attention_mask', 0), ('mm_token_type_ids', 0)):
                packed[name] = torch.cat([F.pad(v[name], (longest-v[name].shape[1], 0), value=fill)
                                           for v in inputs_list])
            for name in ('pixel_values', 'image_grid_thw'):
                packed[name] = torch.cat([v[name] for v in inputs_list])
            with runtime._cached_image_features(visions), torch.no_grad():
                output = runtime.model.generate(**packed, max_new_tokens=96, do_sample=False)
            texts = runtime.tokenizer.batch_decode(output[:, longest:], skip_special_tokens=True)
            if not checked_comparison:
                comparison = []
                with torch.no_grad():
                    for index,(inputs, task, batched_text) in enumerate(zip(inputs_list, tasks, texts, strict=True)):
                        row = task[2]
                        real_length=inputs['input_ids'].shape[1]
                        for name in ('input_ids','attention_mask','mm_token_type_ids'):
                            if not torch.equal(packed[name][index:index+1,-real_length:],inputs[name]):
                                raise RuntimeError('frozen-generator packed input changed')
                        if packed['attention_mask'][index,:longest-real_length].any():
                            raise RuntimeError('frozen-generator left padding is unmasked')
                        vision = [frames.frames[(row['video_path'], t['frame_index'])][1]
                                  for t in row['turns']]
                        with runtime._cached_image_features(vision):
                            single_inputs = dict(inputs)
                            for name,fill in (('input_ids',runtime.tokenizer.pad_token_id),
                                               ('attention_mask',0),('mm_token_type_ids',0)):
                                single_inputs[name] = F.pad(inputs[name],
                                    (longest-inputs[name].shape[1],0),value=fill)
                            single = runtime.model.generate(**single_inputs, max_new_tokens=96, do_sample=False)
                        text = runtime.tokenizer.decode(single[0, longest:],
                                                        skip_special_tokens=True)
                        comparison.append({'frame_index':row['frame_index'], 'video_id':row['video_id'],
                                           'batched_text':batched_text.strip(),'singleton_text':text.strip(),
                                           'text_match':text.strip()==batched_text.strip()})
                # BF16 MoE expert GEMMs can change a greedy tie into a paraphrase.
                # Inputs are checked above; actual answers are saved in actor history.
                (Path(cfg['output'])/f'generator_batch_comparison.gpu{gpu}.json').write_text(
                    json.dumps({'input_contract_verified':True,'batch':len(tasks),'cases':comparison},indent=2)+'\n')
                checked_comparison = True
            seconds = stamp()-started
            stats = {'gpu': gpu, 'batch_id': tasks[0][1], 'batch': len(tasks), 'seconds': seconds,
                     'singleton_comparison_checked': checked_comparison, **memory()}
            for (rank, key, row, submitted), text in zip(tasks, texts, strict=True):
                answer_queues[rank].put((key, text.strip(), {**stats, 'request_seconds': time.monotonic()-submitted}))
    except BaseException:
        events.put(('error', gpu, traceback.format_exc()))
        raise
