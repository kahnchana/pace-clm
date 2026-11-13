#!/usr/bin/env python3
"""Eight-H200 RL pipeline; bounded profiling unless full training is explicit."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import signal
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

from jevany_rl_mixed_workers import controller
from pace_clm.rl_episode import ANCHOR_COEFFICIENT
from pace_clm.rl_mixed import training_schedule, validate_spec, attach_loss_scales
from jevany_rl_rpc import serve, Sidecar
from pace_clm.rl import leave_one_out, reward_ledger
from pace_clm.rl_anchor import verify_committed_checkpoint
from stream_gemma.grpo.judge import JudgeRequest
from jevany_mixed_diagnostics import training_diagnostics


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--anchor-manifest',type=Path,required=True)
    p.add_argument('--anchor-target-fraction',type=float,default=.15)
    p.add_argument('--anchor-per-round',type=int,default=126)
    p.add_argument('--spans', type=Path, default=ROOT/'outputs/jevany-2b-rl/data/spans.jsonl')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243')
    p.add_argument('--generator-python',type=Path,default=ROOT.parent/'lmms-eval/.venv/bin/python')
    p.add_argument('--learners', type=int, choices=(5, 6, 7), default=6)
    p.add_argument('--starts', type=int, default=8)
    p.add_argument('--rollouts', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--physical-batch', type=int, default=32)
    p.add_argument('--collection-batch', type=int, default=8)
    p.add_argument('--rollout-max-lead',type=int,default=2)
    p.add_argument('--collection-min-ready',type=float,default=.75)
    p.add_argument('--generator-batch', type=int, default=8)
    p.add_argument('--frame-cache', type=int, default=512)
    p.add_argument('--token-bucket', type=int, default=512)
    p.add_argument('--replay-token-budget', type=int, default=262144)
    p.add_argument('--updates', type=int, default=4, help='Default bounded gate; includes warmup')
    p.add_argument('--decision-budget',type=int,default=131072)
    p.add_argument('--sampling-unit',choices=('decision','episode'),default='decision')
    p.add_argument('--full-run',action='store_true',help='Explicitly allow more than 16 updates; never enabled by profiling commands')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--lora-lr', type=float, default=1e-6)
    p.add_argument('--parity-probability-tolerance',type=float,default=.005)
    p.add_argument('--parity-kl-tolerance',type=float,default=1e-4)
    p.add_argument('--parity-mean-kl-tolerance',type=float,default=1e-5)
    p.add_argument('--selected-logp-tolerance',type=float,default=.05)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reward-version', choices=('proxy-v1', 'pauc-f-aligned-gpt-v1'), default='proxy-v1')
    p.add_argument('--reward-audit', type=Path, default=ROOT/'outputs/jevany-2b-rl/reward-audit-v1')
    p.add_argument('--tune-replay', action='store_true',
                   help='Benchmark real-row chunk sizes within the fixed physical shape, without optimizer steps')
    a = p.parse_args()
    if not 1<=a.decision_budget<=131072:p.error('invalid frozen decision exposure budget')
    if a.anchor_per_round!=126 or a.anchor_target_fraction!=.15:
        p.error('frozen pilot requires 126 supervised examples and target fraction .15')
    if a.reward_version != 'proxy-v1':
        from jevany_rl_reward import contract
        if json.loads((a.reward_audit/'reward-spec.json').read_text()) != contract():
            raise ValueError('reward contract differs from audited specification')
        if not json.loads((a.reward_audit/'summary.json').read_text())['gate_passed']:
            raise ValueError('reward audit must pass before training')
    if not 1 <= a.updates <= (4096 if a.full_run else 16) or a.rollouts < 2 or a.starts < 1:
        p.error('invalid bounded profiling budget')
    if a.rollouts != 4 or a.learners != 7:
        p.error('each learner must have a trajectory')
    if a.rollout_max_lead<0 or not 0<a.collection_min_ready<=1:
        p.error('invalid rollout batching limits')
    if max(a.microbatch,a.collection_batch) > a.physical_batch:
        p.error('logical chunks cannot exceed the fixed physical batch cap')
    if min(a.physical_batch,a.microbatch, a.collection_batch, a.generator_batch, a.frame_cache,
           a.token_bucket, a.replay_token_budget) < 1:
        p.error('batch and cache sizes must be positive')
    if socket.gethostname()!='ip-10-3-120-167':raise RuntimeError('authorized node only')
    allocation=subprocess.check_output(['squeue','-h','-j','38177','-o','%T %N'],text=True).strip()
    if allocation!='RUNNING ip-10-3-120-167':raise RuntimeError('authorized allocation unavailable')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'outputs/jevany-2b-rl/gpu.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    app = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if app.strip():
        raise RuntimeError('episode training requires eight unused GPUs')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    if len(devices) != 8 or any('H200' not in s for s in devices):
        raise RuntimeError('requires eight H200 GPUs')
    write_json(a.output/'runtime.json', {'hostname':socket.gethostname(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'devices':devices,
        'run_kind':'full_training' if a.full_run else 'bounded_profile',
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
    spans = [json.loads(line) for line in a.spans.read_text().splitlines() if line]
    for spec in spans:validate_spec(spec)
    episode_batches=training_schedule(spans,sampling_unit=a.sampling_unit,budget=a.decision_budget,seed=a.seed)
    prepared=json.loads((a.spans.parent/'schedule.json').read_text())
    if prepared!=episode_batches:raise ValueError('trainer schedule differs from prepared schedule')
    if a.updates>len(episode_batches):raise ValueError('pilot budget exhausted')
    freeze = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()
              if k not in ('resume', 'updates','full_run')}
    files = [a.spans, a.checkpoint/'pointer_head.pt',
             a.checkpoint/'controller_adapter/adapter_model.safetensors',
             a.checkpoint/'controller_adapter/adapter_config.json',
             SG/'outputs/lora-qwen-w1-long-neg/adapter_model.safetensors',
             SG/'outputs/lora-qwen-w1-long-neg/adapter_config.json',
             ROOT/'src/pace_clm/rl.py', Path(__file__), ROOT/'scripts/jevany_rl_mixed_workers.py',ROOT/'scripts/jevany_rl_workers.py',
             ROOT/'src/pace_clm/rl_anchor.py',a.anchor_manifest,a.anchor_manifest.parent/'freeze.json',
             ROOT/'scripts/prepare_jevany_rl_anchor.py',ROOT/'scripts/verify_jevany_mixed_resume.py']
    files += [a.spans.parent/'audit.json', ROOT/'src/pace_clm/qwen_controller_runtime.py',
              ROOT/'src/pace_clm/jevany_streaming.py', ROOT/'configs/actions_v1.yaml',
              ROOT/'scripts/benchmark_qwen35_2b_training.py', ROOT/'scripts/train_jevany_streaming.py',
              SG/'src/stream_gemma/grpo/judge.py', SG/'src/stream_gemma/inference/qwen_stream.py']
    files += [ROOT/'scripts/jevany_rl_judge.py', SG/'src/stream_gemma/grpo/local_judge.py']
    files += [ROOT/'scripts/run_jevany_rl_profile.sh',ROOT/'scripts/verify_jevany_mixed_resume.py',ROOT/'src/pace_clm/rl_inputs.py',ROOT/'src/pace_clm/rl_broker.py',ROOT/'scripts/jevany_rl_rpc.py',
              ROOT/'scripts/jevany_rl_mixed_sidecar.py']
    files += [ROOT/'scripts/run_jevany_rl_gate.sh', SG/'src/stream_gemma/config.py',
              SG/'src/stream_gemma/v24_memory.py', ROOT/'src/pace_clm/rl_reference.py']
    freeze['algorithm']='mixed-native-episode-rloo-v1' if a.sampling_unit=='decision' else 'mixed-episode-balanced-rloo-v2'
    freeze['domain_objective']='0.5 mean ours + 0.25 mean WEB + 0.25 mean EGO; constant 32 loss scale'
    files += [ROOT/'scripts/jevany_mixed_control.py',ROOT/'src/pace_clm/rl_mixed.py',ROOT/'scripts/jevany_mixed_inputs.py',ROOT/'scripts/jevany_mixed_reward.py',ROOT/'scripts/jevany_rl_mixed_generator_workers.py',ROOT/'scripts/jevany_mixed_diagnostics.py']
    freeze['episode_batches']=episode_batches
    freeze['loss_normalization']='sum decision losses / (global trajectories * constant 32); never per-trajectory length'
    freeze['anchor_coefficient']=ANCHOR_COEFFICIENT
    freeze['replay_layout']='original collected peers, row positions and padding shapes; no regrouping'
    freeze['episode_reward_contract']={'version':'mixed-native-episode-pauc-f-v1',
        'ours_windows':'original four-second trusted windows',
        'web_ego_windows':'unchanged native answer objects with original reply/related spans',
        'reward':'official per-episode PAUC-F / 100', 'sampling':'four complete causal trajectories per episode'}
    files += [ROOT/'src/pace_clm/rl_episode.py',ROOT/'scripts/prepare_jevany_episode_rl.py',ROOT/'scripts/jevany_episode_reward.py']
    if a.sampling_unit=='episode':files += [ROOT/'scripts/prepare_jevany_episode_balanced_mixed.py',a.spans.parent/'schedule.json']
    freeze['inference']='raw learned argmax; no heuristic score modifications'
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
        latest_path=a.output/'latest.json'
        latest=json.loads(latest_path.read_text()) if latest_path.exists() else None
        completed=latest['step'] if latest else 0
        if latest:verify_committed_checkpoint(latest['checkpoint'],completed)
        elif (a.output/'anchor-calibration.json').exists():
            # No published optimizer commit: restart calibration and collection from SFT.
            (a.output/'anchor-calibration.json').rename(a.output/f'uncommitted-calibration-{time.time_ns()}.json')
    else:
        if freeze_path.exists():
            raise FileExistsError('output exists; use a new output or --resume')
        write_json(freeze_path, freeze)
        latest, completed = None, 0
    if completed >= a.updates:
        # Still restore and compare checkpoint state in the workers below.
        print(json.dumps({'resume_verification': completed}), flush=True)
    cfg = {**freeze, 'resume': latest['checkpoint'] if latest else None,
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
        # Whole-episode final batches can be much larger than ordinary updates.
        deadline = time.monotonic()+(7200 if kind == 'learn' else 1800)
        last_notice=time.monotonic()
        while len(received) < count:
            try:
                event, rank, value = events.get(timeout=5)
            except queue.Empty:
                dead = [(w.pid,w.exitcode) for w in workers if w.exitcode is not None]
                if time.monotonic()-last_notice>=60:
                    print(json.dumps({'waiting_for':kind,'received_ranks':sorted(received),
                        'pending_ranks':sorted(set(range(count))-set(received)),
                        'seconds_remaining':max(0,deadline-time.monotonic())}),flush=True)
                    last_notice=time.monotonic()
                if dead or time.monotonic() > deadline:
                    for w in workers:
                        if w.is_alive():
                            try:os.kill(w.pid,signal.SIGUSR1)
                            except ProcessLookupError:pass
                    write_json(a.output/'worker-failure.json',{'phase':kind,'received_ranks':sorted(received),
                        'dead_workers':dead,'utc_epoch':time.time(),'worker_pids':[w.pid for w in workers]})
                    time.sleep(2)
                    raise RuntimeError(f'workers stalled or exited in {kind}; received={sorted(received)}; dead={dead}; see worker-failure.json and stack logs')
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
            worker=Sidecar([str(a.generator_python),'-u',str(ROOT/'scripts/jevany_rl_mixed_sidecar.py'),
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
        for version in range(completed, a.updates):
            from jevany_mixed_control import before_batch
            if not before_batch(a.output,version):break
            started = time.monotonic()
            started_epoch = time.time()
            broker_before=server.generator_broker.snapshot()
            batch = a.output/f'batch-{version:06d}'
            batch.mkdir(exist_ok=True)
            assignments = [[] for _ in range(a.learners)]
            batch_specs=[spans[i] for i in episode_batches[version]['episode_indices']]
            for group, spec in enumerate(batch_specs):
                for rollout in range(a.rollouts):
                    i = group*a.rollouts+rollout
                    assignments[i%a.learners].append((i,spec))
            for rank in range(a.learners):
                commands[rank].put(('rollout',(version,assignments[rank])))
            collected = receive('rollout', a.learners)
            collection_ended_epoch = time.time()
            traces = sorted((t for value in collected.values() for t in value.pop('traces')),
                            key=lambda t: t['trajectory_id'])
            broker_after=server.generator_broker.snapshot()
            logical=broker_after['logical_requests']-broker_before['logical_requests']
            if logical!=sum(s['action']==2 for t in traces for s in t['steps']) or broker_after['pending_contexts']:
                raise RuntimeError('generator broker/trajectory accounting mismatch')
            write_json(batch/'rollouts.json', traces)
            tick = time.monotonic()
            if a.reward_version == 'proxy-v1':
                requests_payload, indexes = judge_requests(traces)
                parts = 8-a.learners
                for part in range(parts):
                    requests.put(('judge', part, requests_payload[part::parts], str(batch)))
                judged = receive('judge', parts)
                judge = apply_scores(traces, indexes, [r for value in judged.values() for r in value['results']])
            else:
                from jevany_mixed_reward import score_traces
                judge = score_traces(traces, batch/'reward', a.output/'cache/training.sqlite')
            judge['seconds'] = time.monotonic()-tick
            judge_ended_epoch = time.time()
            for group in range(len(batch_specs)):
                members = traces[group*a.rollouts:(group+1)*a.rollouts]
                advantages = leave_one_out([t['reward'] for t in members])
                for t, advantage in zip(members, advantages, strict=True):
                    for step in t['steps']:
                        step['advantage'] = advantage
            source_counts=attach_loss_scales(traces)
            write_json(batch/'scored.json', traces)
            if a.reward_version != 'proxy-v1' and not any(
                    s['action'] == 2 for t in traces if t['spec']['gold_points'] for s in t['steps']):
                raise RuntimeError('near-silence: zero fires across all event-containing training trajectories')
            per_rank = [[s for t in traces if t['trajectory_id']%a.learners == rank for s in t['steps']]
                        for rank in range(a.learners)]
            if a.tune_replay and version == 0:
                tuning = []
                probe_rows = [sorted(rows, key=lambda s: len(s['row']['turns']))[-64:]
                              for rows in per_rank]
                candidates = sorted({min(8,a.physical_batch),min(16,a.physical_batch),a.physical_batch})
                for size in candidates:
                    for repeat in range(3):
                        for rank in range(a.learners):
                            commands[rank].put(('benchmark',(probe_rows[rank],len(traces),size)))
                        results = receive('benchmark', a.learners)
                        tuning.append({'microbatch': size, 'repeat': repeat, 'ranks': results})
                safe = [size for size in candidates
                        if max(r['reserved_gib'] for t in tuning if t['microbatch'] == size
                               for r in t['ranks'].values()) < 115]
                if not safe:
                    raise RuntimeError('no replay candidate has sufficient memory headroom')
                chosen_microbatch = min(safe, key=lambda size:
                    sum(max(r['seconds'] for r in t['ranks'].values())
                        for t in tuning if t['microbatch'] == size and t['repeat'] > 0)/2)
                write_json(tuning_path, {'selected_microbatch': chosen_microbatch, 'trials': tuning})
            for rank in range(a.learners):
                commands[rank].put(('learn',(per_rank[rank],len(traces),chosen_microbatch)))
            learned = receive('learn', a.learners)
            learner_ended_epoch = time.time()
            calibration=learned[0]['anchor_calibration']
            if any(v['anchor_lambda']!=calibration['coefficient'] for v in learned.values()):
                raise RuntimeError('anchor coefficient differs across ranks')
            calibration_path=a.output/'anchor-calibration.json'
            if calibration_path.exists() and json.loads(calibration_path.read_text())!=calibration:
                raise RuntimeError('frozen anchor calibration changed')
            if not calibration_path.exists():write_json(calibration_path,calibration)
            checkpoint = a.output/f'step-{version+1:06d}'
            tick = time.monotonic()
            for q in commands:
                q.put(('save',(str(checkpoint),version+1)))
            receive('save', a.learners)
            commit={'step':version+1,'files':{str(path.relative_to(checkpoint)):digest(path) for path in sorted(checkpoint.rglob('*')) if path.is_file() and path.name!='commit.json'}}
            write_json(checkpoint/'commit.json',commit)
            save_seconds = time.monotonic()-tick
            seconds = time.monotonic()-started
            broker_after=server.generator_broker.snapshot()
            broker_stats={k:broker_after[k]-broker_before[k] for k in ('logical_requests','unique_computations','cache_hits','inflight_duplicates')}
            if broker_after['pending_contexts']:
                raise RuntimeError('generator contexts still pending after a complete update')
            report = {'step': version+1, 'collection_round': version+1, 'optimizer_steps': version+1,
                      'reward_version': a.reward_version,
                      'generator_broker':broker_stats, 'started_epoch': started_epoch, 'ended_epoch': time.time(),
                      'collection_ended_epoch': collection_ended_epoch,
                      'judge_ended_epoch': judge_ended_epoch, 'learner_ended_epoch': learner_ended_epoch,
                      'seconds': seconds, 'decisions': sum(len(t['steps']) for t in traces),
                      'episode_schedule':episode_batches[version],
                      'microbatch': chosen_microbatch,
                      'collection': collected, 'judge': judge, 'learner': learned,
                      'save_seconds': save_seconds, 'gpu_hours': 8*seconds/3600,
                      'mean_reward': sum(t['reward'] for t in traces)/len(traces),
                      'domain_trajectory_counts':source_counts,
                      'rollout_diagnostics':training_diagnostics(traces),
                      'fires': sum(s['action'] == 2 for t in traces for s in t['steps']),
                      'nonzero_advantage_trajectories': sum(t['steps'][0]['advantage'] != 0 for t in traces)}
            write_json(batch/'report.json',report)
            write_json(a.output/'latest.json', {'step': version+1, 'checkpoint': str(checkpoint)})
            from jevany_mixed_control import publish_checkpoint
            publish_checkpoint(a.output,version+1,checkpoint,report)
            print(json.dumps({'complete_update': version+1, 'seconds': seconds,
                              'decisions_per_second': report['decisions']/seconds,
                              'mean_reward':report['mean_reward']}), flush=True)
        write_json(a.output/'status.json', {'status':'stopped_by_evaluation' if (a.output/'stop-request.json').exists() else 'full_training_complete', 'steps':json.loads((a.output/'latest.json').read_text())['step'], 'full_training_launched':a.full_run})
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
