#!/usr/bin/env python3
"""Clipped GRPO pipeline with four durable optimizer commits per collection round."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import importlib.metadata
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent/'stream-gemma'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(SG/'src'), str(SG)]

import torch.multiprocessing as mp

from jevany_grpo_workers import controller
from jevany_rl_rpc import serve, Sidecar
from pace_clm.rl import reward_ledger
from pace_clm.grpo import group_advantages, update_schedule, partition_trajectories, recover_progress, validate_progress, batch_identity
from stream_gemma.grpo.judge import JudgeRequest


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def judge_requests(traces):
    """Construct local-only semantic requests; never send video material off-node."""
    requests, indexes = {}, []
    for trace in traces:
        spec = trace['spec']
        fires = [s for s in trace['steps'] if s['action'] == 2]
        semantic, repetition = [], []
        for step in fires:
            ids = []
            for gold in spec['gold_points']:
                request = JudgeRequest('semantic', spec['question'], step['answer'], gold['content'])
                requests[request.request_id] = request
                ids.append(request.request_id)
            semantic.append(ids)
            row = step['row']
            prior = ([row['memory']['content']] if row['memory'] else [])
            prior.extend(t['emit'][len('</Response>'):].strip() for t in row['turns'][:-1]
                         if t['emit'].startswith('</Response>'))
            request = JudgeRequest('repetition', spec['question'], step['answer'], '\n'.join(prior) or '- none')
            requests[request.request_id] = request
            repetition.append(request.request_id)
        indexes.append((fires, semantic, repetition))
    return [asdict(r) for r in requests.values()], indexes


def apply_scores(traces, indexes, results):
    by_id = {r['request_id']: r['score'] for r in results}
    for trace, (fires, semantic, repetition) in zip(traces, indexes, strict=True):
        ledger = reward_ledger(trace['spec']['gold_points'],
                               [s['row'] for s in fires],
                               [[by_id[i] for i in ids] for ids in semantic],
                               [by_id[i] for i in repetition])
        trace['ledger'] = ledger
        trace['reward'] = sum(r['value'] for r in ledger)
    return {'requests': len(results), 'cached': sum(r['cached'] for r in results)}


def finalize_round(output, version, learners):
    batch=output/f'batch-{version:06d}'
    checkpoint=output/'optimizer_checkpoints'/f'update-{4*(version+1):06d}'
    if not (checkpoint/'commit.json').exists():raise RuntimeError('round lacks its fourth committed optimizer update')
    alias=output/f'step-{version+1:06d}'
    if alias.is_symlink():
        if alias.resolve()!=checkpoint.resolve():raise RuntimeError('completed round checkpoint alias changed')
    elif alias.exists():raise FileExistsError(alias)
    else:alias.symlink_to(checkpoint.resolve(),target_is_directory=True)
    if (batch/'report.json').exists():return
    collection=json.loads((batch/'collection.json').read_text())
    judge=json.loads((batch/'judge.json').read_text())
    parity=json.loads((batch/'parity.json').read_text())
    updates=[json.loads((batch/f'update-{i:02d}.json').read_text()) for i in range(4)]
    saves=[json.loads((batch/f'save-{i:02d}.json').read_text()) if (batch/f'save-{i:02d}.json').exists() else {'seconds':0.} for i in range(4)]
    traces=json.loads((batch/'scored.json').read_text())
    core_seconds=(collection['ended_epoch']-collection['started_epoch']+judge['seconds']+
                  max(v['seconds'] for v in parity.values())+sum(u['seconds'] for u in updates)+sum(s['seconds'] for s in saves))
    report={'step':version+1,'collection_round':version+1,'optimizer_steps':4*(version+1),
            'decisions':4096,'actor_replayed_decisions':8192,'parity_verified_decisions':4096,
            'seconds':core_seconds,'end_to_end_seconds':time.time()-collection['started_epoch'],
            'gpu_hours':8*core_seconds/3600,'judge':judge,'collection':collection['collection'],
            'behavior_parity':parity,'updates':updates,'saves':saves,'completed_epoch':time.time(),
            'mean_reward':sum(t['reward'] for t in traces)/128,
            'fires':sum(s['action']==2 for t in traces for s in t['steps'])}
    write_json(batch/'report.json',report)
    print(json.dumps({'complete_round':version+1,'optimizer_steps':4*(version+1),'seconds':core_seconds}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spans', type=Path, default=ROOT/'outputs/jevany-2b-rl/data/spans.jsonl')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243')
    p.add_argument('--generator-python',type=Path,default=ROOT.parent/'lmms-eval/.venv/bin/python')
    p.add_argument('--learners', type=int, choices=(5, 6, 7), default=7)
    p.add_argument('--starts', type=int, default=32)
    p.add_argument('--rollouts', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=24)
    p.add_argument('--physical-batch', type=int, default=24)
    p.add_argument('--collection-batch', type=int, default=24)
    p.add_argument('--rollout-max-lead',type=int,default=2)
    p.add_argument('--collection-min-ready',type=float,default=.75)
    p.add_argument('--generator-batch', type=int, default=8)
    p.add_argument('--frame-cache', type=int, default=2048)
    p.add_argument('--token-bucket', type=int, default=512)
    p.add_argument('--replay-token-budget', type=int, default=262144)
    p.add_argument('--updates', type=int, default=4, help='Default bounded gate; includes warmup')
    p.add_argument('--full-run',action='store_true',help='Explicitly allow more than 16 updates; never enabled by profiling commands')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--lora-lr', type=float, default=1e-6)
    p.add_argument('--parity-probability-tolerance',type=float,default=.005)
    p.add_argument('--parity-kl-tolerance',type=float,default=1e-4)
    p.add_argument('--parity-mean-kl-tolerance',type=float,default=1e-5)
    p.add_argument('--selected-logp-tolerance',type=float,default=.05)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reward-version', choices=('proxy-v1', 'pauc-f-aligned-gpt-v1'), default='pauc-f-aligned-gpt-v1')
    p.add_argument('--reward-audit', type=Path, default=ROOT/'outputs/jevany-2b-rl/reward-audit-v1')
    p.add_argument('--tune-replay', action='store_true',
                   help='Benchmark real-row chunk sizes within the fixed physical shape, without optimizer steps')
    p.add_argument('--stop-after-optimizer-steps', type=int)
    a = p.parse_args()
    if (a.learners, a.starts, a.rollouts, a.physical_batch, a.microbatch, a.collection_batch, a.frame_cache) != (7,32,4,24,24,24,2048):
        p.error('GRPO requires the frozen 7-learner, 32x4, batch24, frame-cache2048 recipe')
    if a.stop_after_optimizer_steps is not None and not 1 <= a.stop_after_optimizer_steps <= 4*a.updates:
        p.error('invalid absolute optimizer-step stop target')
    if a.reward_version != 'pauc-f-aligned-gpt-v1' or a.tune_replay or a.updates > 32:
        p.error('GRPO requires audited GPT reward, no replay tuning, maximum 32 collection rounds')
    if a.reward_version != 'proxy-v1':
        from jevany_rl_reward import contract
        if json.loads((a.reward_audit/'reward-spec.json').read_text()) != contract():
            raise ValueError('reward contract differs from audited specification')
        if not json.loads((a.reward_audit/'summary.json').read_text())['gate_passed']:
            raise ValueError('reward audit must pass before training')
    if not 1 <= a.updates <= (4096 if a.full_run else 16) or a.rollouts < 2 or a.starts < 1:
        p.error('invalid bounded profiling budget')
    if a.starts*a.rollouts < a.learners:
        p.error('each learner must have a trajectory')
    if a.rollout_max_lead<0 or not 0<a.collection_min_ready<=1:
        p.error('invalid rollout batching limits')
    if max(a.microbatch,a.collection_batch) > a.physical_batch:
        p.error('logical chunks cannot exceed the fixed physical batch cap')
    if min(a.physical_batch,a.microbatch, a.collection_batch, a.generator_batch, a.frame_cache,
           a.token_bucket, a.replay_token_budget) < 1:
        p.error('batch and cache sizes must be positive')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'outputs/jevany-2b-rl/gpu.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    app = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if app.strip():
        raise RuntimeError('profiling requires eight unused GPUs')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    if len(devices) != 8 or any('H200' not in s for s in devices):
        raise RuntimeError('requires eight H200 GPUs')
    write_json(a.output/'runtime.json', {'hostname':socket.gethostname(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'devices':devices,
        'run_kind':'full_training' if a.full_run else 'bounded_profile',
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
    spans = [json.loads(line) for line in a.spans.read_text().splitlines() if line]
    if any(s['rollout_stop']-s['rollout_start'] != 32 or
           any(not s['rollout_start'] <= g['time_seconds'] or
               g['time_seconds']+4 > s['rollout_stop'] for g in s['gold_points']) for s in spans):
        raise ValueError('spans must have 32 decisions and complete reward windows')
    if a.starts*a.updates > len(spans):
        raise ValueError('insufficient unique span starts')
    freeze = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()
              if k not in ('resume', 'updates','full_run','stop_after_optimizer_steps')}
    files = [a.spans, a.checkpoint/'pointer_head.pt',
             a.checkpoint/'controller_adapter/adapter_model.safetensors',
             a.checkpoint/'controller_adapter/adapter_config.json',
             SG/'outputs/lora-qwen-w1-long-neg/adapter_model.safetensors',
             SG/'outputs/lora-qwen-w1-long-neg/adapter_config.json',
             ROOT/'src/pace_clm/rl.py', Path(__file__), ROOT/'scripts/jevany_rl_workers.py']
    files += [a.spans.parent/'audit.json', ROOT/'src/pace_clm/qwen_controller_runtime.py',
              ROOT/'src/pace_clm/jevany_streaming.py', ROOT/'configs/actions_v1.yaml',
              ROOT/'scripts/benchmark_qwen35_2b_training.py', ROOT/'scripts/train_jevany_streaming.py',
              SG/'src/stream_gemma/grpo/judge.py', SG/'src/stream_gemma/inference/qwen_stream.py']
    files += [ROOT/'scripts/jevany_rl_judge.py', SG/'src/stream_gemma/grpo/local_judge.py']
    files += [ROOT/'scripts/run_jevany_rl_profile.sh',ROOT/'scripts/verify_jevany_rl_resume.py',ROOT/'src/pace_clm/rl_inputs.py',ROOT/'src/pace_clm/rl_broker.py',ROOT/'scripts/jevany_rl_rpc.py',
              ROOT/'scripts/jevany_rl_generator_sidecar.py']
    files += [ROOT/'scripts/run_jevany_rl_gate.sh', SG/'src/stream_gemma/config.py',
              SG/'src/stream_gemma/v24_memory.py', ROOT/'src/pace_clm/rl_reference.py']
    freeze['algorithm'] = {'version':'categorical-clipped-grpo-v2-exact-vision','passes':2,'global_minibatch':64,
        'steps_per_round':4,'clip_epsilon':.2,'group_std_correction':0,'advantage_epsilon':1e-6,
        'kl_beta':.02,'entropy_beta':.001,'weight_decay':.01,'gradient_clip':1.,'horizon':32,
        'behavior':'fixed selected log-probabilities for the full collected round'}
    files += [ROOT/'src/pace_clm/grpo.py', ROOT/'src/pace_clm/grpo_vision.py', ROOT/'scripts/jevany_grpo_workers.py',
              ROOT/'scripts/verify_jevany_grpo_resume.py']
    freeze['judge'] = 'offline Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754 base / jevany-offline-judge-v8-native-grouped-mm'
    if a.reward_version != 'proxy-v1':
        freeze['judge'] = 'gpt-4.1 / official cumulative-answer rubric / authorized gateway'
        freeze['reward_contract'] = contract()
        files += [ROOT/'scripts/jevany_rl_reward.py', ROOT/'scripts/audit_jevany_rl_reward.py',
                  a.reward_audit/'reward-spec.json', a.reward_audit/'summary.json',
                  SG/'zs_infer/pauc_f.py', SG/'zs_infer/scoring.py',
                  SG/'third_party/proactive-videoqa/pauc/judge_llm_scores_openai.py',
                  ROOT/'scripts/judge_exp10_utility.py']
    freeze['hashes'] = {str(f.resolve()): digest(f) for f in files}
    freeze['versions'] = {name:importlib.metadata.version(name) for name in ('torch','transformers','peft')}
    version_code="import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('torch','transformers','peft','triton')}))"
    freeze['generator_versions']=json.loads(subprocess.check_output(
        [str(a.generator_python),'-c',version_code],text=True))
    freeze_path = a.output/'freeze.json'
    if a.resume:
        if json.loads(freeze_path.read_text()) != freeze:
            raise ValueError('resume contract differs from frozen configuration or code')
        latest = recover_progress(a.output)
        completed = latest['step']
    else:
        if freeze_path.exists():
            raise FileExistsError('output exists; use a new output or --resume')
        write_json(freeze_path, freeze)
        latest, completed = {'step':0,'optimizer_steps':0,'cursor':None,'checkpoint':None}, 0
    if completed >= a.updates:
        # Still restore and compare checkpoint state in the workers below.
        print(json.dumps({'resume_verification': completed}), flush=True)
    cfg = {**freeze, 'resume': latest['checkpoint'], 'resume_progress': latest,
           'generator': str(SG/'outputs/lora-qwen-w1-long-neg')}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        cfg['init_method'] = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    ctx = mp.get_context('spawn')
    events, requests = ctx.Queue(), ctx.Queue()
    commands = [ctx.Queue() for _ in range(a.learners)]
    answers = [ctx.Queue() for _ in range(a.learners)]
    workers = []
    server,key=serve(events,requests,answers)
    worker_config=a.output/'worker_runtime.json'
    write_json(worker_config,cfg)
    stop = threading.Event()

    def monitor():
        with (a.output/'gpu_samples.csv').open('a') as handle:
            while not stop.is_set():
                subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                 '--format=csv,noheader,nounits'], stdout=handle, stderr=handle)
                handle.flush()
                stop.wait(1)

    def receive(kind, count):
        received = {}
        deadline = time.monotonic()+1800
        while len(received) < count:
            try:
                event, rank, value = events.get(timeout=5)
            except queue.Empty:
                dead = [(w.pid,w.exitcode) for w in workers if w.exitcode is not None]
                if dead or time.monotonic() > deadline:
                    raise RuntimeError(f'workers stalled or exited: {dead}')
                continue
            if event == 'error':
                raise RuntimeError(f'worker {rank}: {value}')
            if event != kind or rank in received:
                raise RuntimeError(f'unexpected worker event {event}/{rank}, expected {kind}')
            received[rank] = value
            print(json.dumps({'event': event, 'rank': rank,
                              **({k:v for k,v in value.items() if k not in ('traces', 'results')})}), flush=True)
        return received

    try:
        prior_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        visible = prior_visible.split(',') if prior_visible else list(map(str, range(8)))
        if len(visible) != 8:
            raise RuntimeError('expected eight visible GPUs')
        for rank in range(a.learners):
            os.environ['CUDA_VISIBLE_DEVICES'] = visible[rank]
            worker = ctx.Process(target=controller, args=(rank,cfg,commands[rank],events,requests,answers[rank]))
            worker.start()
            workers.append(worker)
        for gpu in range(a.learners, 8):
            os.environ['CUDA_VISIBLE_DEVICES'] = visible[gpu]
            env=os.environ.copy()
            env['PACE_RL_RPC_KEY']=key.hex()
            worker=Sidecar([str(a.generator_python),'-u',str(ROOT/'scripts/jevany_rl_generator_sidecar.py'),
                            '--port',str(server.address[1]),'--gpu',str(gpu),
                            '--config',str(worker_config)],env)
            workers.append(worker)
        if prior_visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = prior_visible
        threading.Thread(target=monitor, daemon=True).start()
        ready = receive('ready', 8)
        if any(ready[r].get('lora_parameters') != 62521344 or
               ready[r].get('pointer_parameters') != 1049088 or
               ready[r].get('exact_reference_tensors') != 300 for r in range(a.learners)):
            raise RuntimeError('trainable parameter contract differs from SFT')
        write_json(a.output/'readiness.json', ready)
        chosen_microbatch = a.microbatch
        tuning_path = a.output/'replay_tuning.json'
        if a.resume and tuning_path.exists():
            chosen_microbatch = json.loads(tuning_path.read_text())['selected_microbatch']
        for prior in range(completed):
            finalize_round(a.output, prior, a.learners)
        for version in range(completed, a.updates):
            if a.stop_after_optimizer_steps is not None and latest['optimizer_steps'] >= a.stop_after_optimizer_steps:
                break
            batch = a.output/f'batch-{version:06d}'
            batch.mkdir(exist_ok=True)
            if not (batch/'rollouts.json').exists():
                started_epoch = time.time()
                broker_before = server.generator_broker.snapshot()
                assignments = [[] for _ in range(a.learners)]
                for group, spec in enumerate(spans[version*a.starts:(version+1)*a.starts]):
                    for rollout in range(a.rollouts):
                        identity = group*a.rollouts+rollout
                        assignments[identity%a.learners].append((identity,spec))
                for rank in range(a.learners):
                    commands[rank].put(('rollout',(version,assignments[rank])))
                collected = receive('rollout',a.learners)
                traces = sorted((t for value in collected.values() for t in value.pop('traces')),
                                key=lambda t:t['trajectory_id'])
                broker_after = server.generator_broker.snapshot()
                logical = broker_after['logical_requests']-broker_before['logical_requests']
                if logical != sum(s['action']==2 for t in traces for s in t['steps']) or broker_after['pending_contexts']:
                    raise RuntimeError('generator/trajectory accounting mismatch')
                write_json(batch/'vision.json',{str(rank):value['vision_cache'] for rank,value in collected.items()})
                write_json(batch/'collection.json', {'started_epoch':started_epoch, 'ended_epoch':time.time(),
                    'collection':collected,'generator_broker':{key:broker_after[key]-broker_before[key]
                        for key in ('logical_requests','unique_computations','cache_hits','inflight_duplicates')}})
                write_json(batch/'rollouts.json',traces)
            if not (batch/'scored.json').exists():
                from jevany_rl_reward import score_traces
                traces = json.loads((batch/'rollouts.json').read_text())
                tick = time.monotonic()
                judge = score_traces(traces,batch/'reward',a.reward_audit/'gpt-cache.sqlite')
                judge['seconds'] = time.monotonic()-tick
                write_json(batch/'judge.json',judge)
                for group in range(a.starts):
                    members = traces[group*a.rollouts:(group+1)*a.rollouts]
                    advantages = group_advantages([t['reward'] for t in members])
                    for trace, advantage in zip(members,advantages,strict=True):
                        for step in trace['steps']:
                            step['advantage'] = advantage
                            step['behavior_selected_logp'] = step['old_logp'][step['action']]
                write_json(batch/'scored.json',traces)
            traces = json.loads((batch/'scored.json').read_text())
            if len(traces)!=128 or [t['trajectory_id'] for t in traces]!=list(range(128)):
                raise RuntimeError('incomplete collected trajectory identities')
            for trace in traces:
                if (trace['spec'] != spans[version*32+trace['trajectory_id']//4] or len(trace['steps'])!=32 or
                        any(s['policy_version']!=version for s in trace['steps'])):
                    raise RuntimeError('span draw, horizon or behavior version changed')
            if not any(s['action']==2 for t in traces if t['spec']['gold_points'] for s in t['steps']):
                raise RuntimeError('near-silence: zero fires in event-containing training trajectories')
            behavior = [{'trajectory_id':t['trajectory_id'], 'span_id':t['spec']['span_id'],
                'steps':[{'frame':s['row']['frame_index'],'action':s['action'],'old_logp':s['old_logp'],
                          'selected':s['behavior_selected_logp'],'version':s['policy_version']} for s in t['steps']]}
                         for t in traces]
            schedule = update_schedule([t['trajectory_id'] for t in traces],a.seed,version)
            for name,value in (('behavior.json',behavior),('schedule.json',schedule)):
                path=batch/name
                if path.exists() and json.loads(path.read_text())!=value:
                    raise RuntimeError(f'immutable collected batch changed: {name}')
                if not path.exists():write_json(path,value)
            if latest['cursor'] is None:
                per_rank = [[s for t in traces if t['trajectory_id']%a.learners==rank for s in t['steps']]
                            for rank in range(a.learners)]
                for rank in range(a.learners):
                    commands[rank].put(('verify_behavior',(per_rank[rank],128,a.microbatch)))
                verified = receive('verify_behavior',a.learners)
                write_json(batch/'parity.json',verified)
            identity = batch_identity(batch, latest['cursor'])
            first_update = latest['cursor']['next_update'] if latest['cursor'] else 0
            for update in range(first_update,4):
                batch_identity(batch,identity)
                per_rank = partition_trajectories(traces,schedule[update]['trajectory_ids'],a.learners)
                tick = time.monotonic()
                for rank in range(a.learners):
                    commands[rank].put(('learn_grpo',{'steps':per_rank[rank],'total_trajectories':64,
                        'microbatch':a.microbatch,'round':version,'update_index':update,'batch_identity':identity}))
                learned = receive('learn_grpo',a.learners)
                optimizer_steps = version*4+update+1
                rounds = version+1 if update==3 else version
                cursor = None if update==3 else {**identity,'round':version,'next_update':update+1}
                if any(v['optimizer_steps']!=optimizer_steps or v['completed_rounds']!=rounds for v in learned.values()):
                    raise RuntimeError('learner optimizer counters disagree')
                write_json(batch/f'update-{update:02d}.json', {'update_index':update,'pass':update//2,
                    'minibatch':update%2,'seconds':time.monotonic()-tick,'learner':learned})
                checkpoint = a.output/'optimizer_checkpoints'/f'update-{optimizer_steps:06d}'
                checkpoint.parent.mkdir(exist_ok=True)
                staging = checkpoint.with_name(checkpoint.name+f'.{os.getpid()}.{time.time_ns()}.incomplete')
                staging.mkdir()
                progress = validate_progress({'step':rounds,'optimizer_steps':optimizer_steps,'cursor':cursor,
                                              'checkpoint':str(checkpoint.resolve())})
                tick = time.monotonic()
                for q in commands:q.put(('save',{'path':str(staging),'progress':progress}))
                receive('save',a.learners)
                manifest = {'parent_checkpoint':latest['checkpoint'],'progress':progress,
                    'files':{str(p.relative_to(staging)):digest(p) for p in staging.rglob('*') if p.is_file()}}
                write_json(staging/'commit.json',manifest)
                if checkpoint.exists():raise FileExistsError('refuse to overwrite an already committed optimizer update')
                staging.rename(checkpoint)
                write_json(a.output/'latest.json',progress)
                latest=progress
                write_json(batch/f'save-{update:02d}.json',{'seconds':time.monotonic()-tick,'checkpoint':str(checkpoint)})
                if update==3:finalize_round(a.output,version,a.learners)
                if a.stop_after_optimizer_steps is not None and optimizer_steps>=a.stop_after_optimizer_steps:
                    break
            if latest['cursor'] is not None:
                break
        write_json(a.output/'status.json',{'status':'target_reached','completed_rounds':latest['step'],
            'optimizer_steps':latest['optimizer_steps'],'cursor':latest['cursor']})
    except BaseException as exc:
        write_json(a.output/'status.json', {'status':'failed', 'error':str(exc),
                                           'full_training_launched':a.full_run})
        raise
    finally:
        for q in commands:
            q.put(('stop',None))
        for _ in range(8-a.learners):
            requests.put(None)
        for worker in workers:
            worker.join(timeout=5)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
        stop.set()
        server.generator_broker.close()
        server.stop_event.set()


if __name__ == '__main__':
    main()
