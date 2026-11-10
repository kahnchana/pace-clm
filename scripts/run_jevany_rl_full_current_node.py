#!/usr/bin/env python3
"""Continue frozen RL to 16/32, with full GPT-scored WEB checks on this node."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent/'stream-gemma'
PY = SG/'.venv/bin/python'
NODE = 'ip-10-3-120-167'
DEFAULT_RUN = ROOT/'outputs/jevany-2b-rl/profile-7p1-v10'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def pending_actions(step, evaluated):
    if not 1 <= step <= 32:
        raise ValueError('expected corrected checkpoint step 1 through 32')
    actions = []
    for target in (16, 32):
        if step < target:
            actions.append(('train', target))
            step = target
        if target not in evaluated:
            actions.append(('web', target))
    return actions


def checkpoint_view(run, checkpoint):
    """Supply SFT evaluator metadata without altering RL optimizer checkpoints."""
    parent = run/'full_run/checkpoints'
    write(parent/'summary.rank0.json', {'mode':'J1', 'purpose':'RL evaluation metadata only'})
    view = parent/checkpoint.name
    view.mkdir(parents=True, exist_ok=True)
    for name in ('controller_adapter', 'pointer_head.pt', 'model_identity.json'):
        source = checkpoint/name
        if not source.exists():
            raise FileNotFoundError(source)
        target = view/name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f'checkpoint view target changed: {target}')
        elif target.exists():
            raise ValueError(f'unexpected checkpoint view file: {target}')
        else:
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    full = run/'full_run'
    full.mkdir(exist_ok=True)
    freeze = read(run/'freeze.json')
    for name, expected in freeze['hashes'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'frozen training artifact changed: {name}')
    latest = read(run/'latest.json')['step']
    evaluated = set()
    for step in (16,32):
        path = run/f'evaluation/web_step{step}'
        if (path/'pauc_f_summary-gpt-4.1-v1.json').exists():
            from run_qwen35_control import verify_web
            verify_web(path)
            evaluated.add(step)
    actions = pending_actions(latest, evaluated)
    plan = {'node':NODE, 'gpus':list(range(8)), 'total_steps':32,
            'initial_completed_step':latest, 'actions':actions,
            'web_videos':500, 'judge':'gpt-4.1', 'scorer':'pauc-f-v1', 'omega':.5,
            'response_margin':0., 'max_new_tokens':96,
            'training_freeze_sha256':digest(run/'freeze.json'),
            'evaluation_runtime':'original SFT WEB evaluator / stream-gemma environment',
            'evaluation_gpu_layout':'8 local ranks, separate frozen promoted generator per rank',
            'gpt_authorization':'User explicitly requested full WEB PAUC-F GPT scoring at steps 16 and 32'}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if socket.gethostname() != NODE:
        raise RuntimeError(f'requires {NODE}, got {socket.gethostname()}')
    if len(os.sched_getaffinity(0)) < 8:
        raise RuntimeError('insufficient CPU affinity; launch with --cpu-bind=none')
    lock = (full/'orchestrator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('node GPUs are already in use')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(devices)!=8 or any('H200' not in name for name in devices):
        raise RuntimeError('requires all eight local H200s')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               TORCHINDUCTOR_COMPILE_THREADS='4', PACE_DISTRIBUTED_BACKEND='gloo',
               TRITON_CACHE_DIR='/tmp/pace-rl-triton',
               PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    logs = full/'logs'
    logs.mkdir(exist_ok=True)
    workflow_files = [Path(__file__), ROOT/'scripts/eval_jevany_streaming.py',
                      ROOT/'scripts/merge_jevany_benchmark.py', ROOT/'scripts/score_exp10_web_ego.py',
                      ROOT/'scripts/judge_exp10_utility.py', ROOT/'scripts/run_qwen35_control.py',
                      ROOT/'scripts/generate_selected_responses.py', SG/'zs_infer/scoring.py',
                      SG/'zs_infer/pauc_f.py', ROOT/'scripts/run_jevany_rl_cached.sh',
                      ROOT/'scripts/jevany_rl_kernel_cache.py']
    hashes = {str(p):digest(p) for p in workflow_files}
    previous = full/'launch_plan.json'
    if previous.exists() and read(previous)['workflow_hashes'] != hashes:
        raise ValueError('workflow source differs from previous launch')
    plan.update(workflow_hashes=hashes, pid=os.getpid(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                started_utc=datetime.now(timezone.utc).isoformat(), cpu_affinity=len(os.sched_getaffinity(0)))
    write(previous, plan)
    snapshots = full/'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name, sha in hashes.items():
        (snapshots/(sha+Path(name).suffix)).write_bytes(Path(name).read_bytes())

    def status(state, **fields):
        record = {'state':state, 'utc':datetime.now(timezone.utc).isoformat(),
                  'node':NODE, 'pid':os.getpid(), **fields}
        write(full/'status.json', record)
        print(json.dumps(record), flush=True)

    def command(argv, log):
        for name, expected in hashes.items():
            if digest(Path(name)) != expected:
                raise ValueError(f'workflow artifact changed: {name}')
        print(json.dumps({'command':list(map(str,argv)), 'log':str(log)}), flush=True)
        with log.open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)

    def evaluate(checkpoint, output, *, canary=False):
        view = checkpoint_view(run, checkpoint)
        output.mkdir(parents=True, exist_ok=True)
        ranks = 1 if canary else 8
        rollout = [PY,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}',
                   ROOT/'scripts/eval_jevany_streaming.py','--controller',view,'--benchmark','web',
                   '--output-dir',output,'--response-margin','0.0','--max-new-tokens','96','--answer-model','promoted']
        if canary:
            rollout += ['--max-videos','1']
        command(rollout,logs/f'{output.name}_rollout.log')
        command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',output,'--world-size',str(ranks)],logs/f'{output.name}_score.log')
        command([PY,ROOT/'scripts/score_exp10_web_ego.py','web','--pred-dir',output,
                 '--expected-videos','1' if canary else '500'],logs/f'{output.name}_score.log')
        if not canary:
            from run_qwen35_control import verify_web
            score, _, _ = verify_web(output)
            write(full/f'web_step{read(checkpoint/"rl_state.json")["policy_version"]}_verified.json',score)

    try:
        status('gpt_preflight')
        sys.path[:0] = [str(ROOT/'scripts'),str(SG)]
        from judge_exp10_utility import load_gateway_key
        load_gateway_key()
        env.pop('OPENAI_API_KEY', None)
        env.pop('OAI_API_KEY', None)
        env.update({k:os.environ[k] for k in ('X_API_KEY','OPENAI_BASE_URL')})
        # Exercise one real, previously recorded public-WEB scoring request.
        probe = full/'gpt_preflight'
        probe.mkdir(exist_ok=True)
        baseline = ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web/pauc_batch_input-gpt-4.1.jsonl'
        with baseline.open() as handle:
            first = next(line for line in handle if line.strip())
        (probe/'input.jsonl').write_text(first)
        if not (probe/'output.jsonl').exists():
            from zs_infer.scoring import web_sync
            web_sync({'batch_input':probe/'input.jsonl','batch_output':probe/'output.jsonl'},
                     concurrency=1,max_attempts=2,timeout=60.)
        response = json.loads((probe/'output.jsonl').read_text())
        content = response['response']['body']['choices'][0]['message']['content']
        if response['custom_id']!=json.loads(first)['custom_id'] or not content or content[0] not in '123':
            raise ValueError('GPT preflight response invalid')
        status('web_canary')
        evaluate(run/'step-000001',run/'evaluation/web_canary_step1',canary=True)
        write(full/'canary_passed.json',{'verified':True,'checkpoint':str(run/'step-000001'),
                                       'gpt_gateway_verified':True})
        for action, step in actions:
            if action=='train':
                status('training',target_step=step,completed_step=read(run/'latest.json')['step'])
                command(['bash',ROOT/'scripts/run_jevany_rl_cached.sh',run,'--learners','7',
                         '--starts','32','--rollouts','4','--physical-batch','24','--microbatch','24',
                         '--collection-batch','24','--frame-cache','2048','--full-run','--resume','--updates',str(step)],
                        logs/f'train_to_step{step}.log')
                if read(run/'latest.json')['step'] != step:
                    raise ValueError('training stopped at unexpected step')
            else:
                status('web_evaluation',step=step,videos=500,judge='gpt-4.1')
                evaluate(run/f'step-{step:06d}',run/f'evaluation/web_step{step}')
        status('complete',completed_step=32,evaluated_steps=[16,32])
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    main()
