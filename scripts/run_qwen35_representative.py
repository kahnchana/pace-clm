#!/usr/bin/env python3
"""Local-only representative-mixture preflight and frozen four-phase experiment."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/jevany-qwen35-2b-mixture-12h'
OUT = BASE/'representative-seed42'
PLANS = BASE/'plans'
SOURCE = ROOT/'outputs/jevany-qwen35-2b-full/manifests/j1s.jsonl'
INITIAL = BASE/'shared-initial-seed42/step-000000'
PY = str(ROOT.parent/'stream-gemma/.venv/bin/python')
HOST = 'ip-10-3-70-177'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temp.replace(path)


def status(phase, **kw):
    value = dict(phase=phase, host=socket.gethostname(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    write(OUT/'status.json',value)
    print(json.dumps(value,default=str),flush=True)


def command(args, log):
    print(json.dumps({'command':args,'log':str(log)}),flush=True)
    with log.open('a') as f:
        subprocess.run(args,cwd=ROOT,env=os.environ,stdout=f,stderr=subprocess.STDOUT,check=True)


def distributed(script, args, log, world=8):
    command([PY,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(ROOT/'scripts'/script),*map(str,args)],log)


def train(output, order, stop, log, *, resume=False, initialize=False):
    args = ['--manifest',SOURCE,'--row-order',order,'--output',output,
        '--microbatch','16','--require-global-batch','128','--backend','gloo',
        '--frame-cache','1024','--conv-batch-chunk','128','--predecode-workers','4',
        '--prefetch-mode','prepare','--microbatch-ratio','1.0','--log-every','1',
        '--save-every','32' if initialize else '250','--stop-after-steps',stop]
    args += ['--resume'] if resume else ['--initialize-from',INITIAL]
    distributed('train_jevany_2b_batched.py',args,log)


def compare_states(left, right):
    import torch
    from safetensors.torch import load_file
    def compare(a,b):
        if isinstance(a,torch.Tensor):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        elif isinstance(a,dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):
                compare(x,y)
        else:
            assert a == b
    compare(load_file(left/'controller_adapter/adapter_model.safetensors'),load_file(right/'controller_adapter/adapter_model.safetensors'))
    compare(torch.load(left/'pointer_head.pt',weights_only=True),torch.load(right/'pointer_head.pt',weights_only=True))
    a,b = [torch.load(p/'training_state.pt',weights_only=False,map_location='cpu') for p in (left,right)]
    for key in ('optimizer','step','seen_per_rank','global_labels','world','seed','model_revision'):
        compare(a[key],b[key])
    for rank in range(8):
        compare(torch.load(left/f'rng.rank{rank}.pt',weights_only=False),torch.load(right/f'rng.rank{rank}.pt',weights_only=False))
    return {'adapter_head_optimizer_rng_match':True,'rtol':1e-5,'atol':1e-6}


def memory_check(output):
    values = [json.loads((output/f'summary.rank{rank}.json').read_text()) for rank in range(8)]
    for value in values:
        if value['peak_reserved_gib'] > .9*value['device_memory_gib']:
            raise ValueError('preflight has less than 10 percent reserved-memory headroom')
    return values


def report(schedule):
    import torch
    from run_qwen35_control import verify_web
    results = []
    common = None
    for phase, step in enumerate(schedule['checkpoint_steps'],1):
        web = OUT/f'evaluation/web_phase{phase}'
        score_path = web/'pauc_f_summary-gpt-4.1-v1.json'
        if not score_path.exists():
            continue
        score, _, fingerprint = verify_web(web)
        protocol = json.loads((web/'manifest.json').read_text())
        predictions = [json.loads(line) for line in (web/'predictions.jsonl').read_text().splitlines()]
        if len(predictions)!=500 or len({r['question_id'] for r in predictions})!=500 or set(r['question_id'] for r in predictions)!=set(protocol['question_ids']):
            raise ValueError('WEB coverage mismatch')
        if score.get('num_videos')!=500 or not score.get('headline_eligible'):
            raise ValueError('incomplete WEB judging')
        stable = {k:v for k,v in protocol.items() if k not in ('controller_adapter_sha256','pointer_head_sha256','controller_model')}
        if common is not None and common != stable:
            raise ValueError('WEB protocol changed between phases')
        common = stable
        state = torch.load(OUT/f'train/step-{step:06d}/training_state.pt',weights_only=False,map_location='cpu')
        actions=Counter()
        repeated_answers=0
        for prediction in predictions:
            actions.update(prediction.get('jevany_actions',{}))
            answers=[str(turn.get('content','')).strip().lower() for turn in prediction.get('model_response_list',[])]
            repeated_answers += len(answers)-len(set(answers))
        results.append(dict(phase=phase,step=step,presentations=state['seen_per_rank']*8,
            actual_training_hours=state['training_seconds']/3600,training_labels=state['global_labels'],web_score=score,
            web_protocol_fingerprint=fingerprint,web_action_counts=dict(actions),
            web_action_rates={k:v/sum(actions.values()) for k,v in actions.items()},
            repeated_identical_answers=repeated_answers))
    write(OUT/'reports/results.json',dict(schedule=schedule,checkpoints=results,
        comparison='Control A is not launched by this runner; paired A/B analysis requires its results.'))
    lines=['# Representative-mixture experiment B','',
        'Local node: '+HOST+'. Step count estimated from a training preflight, as requested.',
        'The run uses the existing Gloo gradient reduction because the NCCL preflight failed with CUDA error 802.',
        '', '| Phase | Step | Presentations | Actual training hours | WEB PAUC-F |',
        '| --- | ---: | ---: | ---: | ---: |']
    for result in results:
        score=result['web_score']
        lines.append(f"| {result['phase']} | {result['step']} | {result['presentations']} | {result['actual_training_hours']:.3f} | {score.get('pauc_f',score.get('pauc_f_score','see JSON'))} |")
    lines += ['', 'Complete WEB score components, fire precision/recall, useful/wasted fires and integrity metadata are in results.json.',
        'A paired A/B conclusion remains pending the control results.']
    (OUT/'reports/results.md').write_text('\n'.join(lines)+'\n')


def main():
    if socket.gethostname().split('.')[0] != HOST:
        raise RuntimeError('runner is restricted to the user-selected current node')
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PACE_DISTRIBUTED_BACKEND='gloo',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
        TRITON_CACHE_DIR='/dev/shm/pace-representative-triton',TMPDIR='/dev/shm/pace-representative-tmp')
    for key in ('TRITON_CACHE_DIR','TMPDIR'):
        Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    logs=OUT/'logs'; logs.mkdir(exist_ok=True)
    status('waiting_for_data_audit_and_initialization')
    deadline=time.monotonic()+1800
    while not (PLANS/'audit.json').exists() or not (INITIAL.parent/'latest.json').exists():
        if time.monotonic()>deadline:
            raise TimeoutError('data preparation or initialization did not complete; inspect prepare.log and initialize.log')
        time.sleep(10)
    audit=json.loads((PLANS/'audit.json').read_text())
    if not audit['full_a_b_row_multiset_equal'] or not audit['causal_history_checked']:
        raise ValueError('failed data integrity audit')
    from judge_exp10_utility import load_gateway_key
    load_gateway_key()
    # Capture the exact code used for review and later diagnosis.
    for name in ('train_jevany_2b_batched.py','train_jevany_streaming.py','plan_qwen35_representative.py','prepare_qwen35_control.py','run_qwen35_representative.py','eval_jevany_streaming.py'):
        target=OUT/'code'/name; target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT/'scripts'/name,target)
    schedule_path=OUT/'schedule.json'
    order=PLANS/'representative.npy'
    if not schedule_path.exists():
        stress = OUT/'stress_check'
        if not (stress/'summary.rank0.json').exists():
            status('maximum_context_memory_preflight')
            offsets=np.load(order,mmap_mode='r')
            lengths=np.load(PLANS/'lengths.npy',mmap_mode='r')
            longest=np.lexsort((offsets[:,1],lengths))[-128:]
            stress_order=OUT/'stress_order.npy'
            np.save(stress_order,offsets[longest])
            train(stress,stress_order,1,logs/'stress_check.log')
        stress_memories=memory_check(stress)
        preflight=OUT/'preflight'
        status('training_preflight',steps=33)
        if not (preflight/'summary.rank0.json').exists():
            train(preflight,order,33,logs/'preflight.log',initialize=True,
                resume=(preflight/'latest.json').exists())
        memories=memory_check(preflight)
        # A saved step 32 compared against the uninterrupted step 33 tests optimizer/RNG/input resume.
        resumed=OUT/'resume_check'; resumed.mkdir(exist_ok=True)
        if not (resumed/'summary.rank0.json').exists():
            shutil.copy2(preflight/'recipe.json',resumed/'recipe.json')
            write(resumed/'latest.json',{'step':32,'checkpoint':str(preflight/'step-000032')})
            train(resumed,order,33,logs/'resume_check.log',resume=True)
        equivalence=compare_states(preflight/'step-000033',resumed/'step-000033')
        status('web_pipeline_smoke_check')
        smoke=OUT/'preflight_web'; smoke.mkdir(exist_ok=True)
        if not (smoke/'pauc_f_summary-gpt-4.1-v1.json').exists():
            annotations=json.loads((ROOT.parent/'stream-gemma/datasets/proactive-videoqa-web/anno.json').read_text())
            shortest=min(annotations,key=lambda row:row['duration'])
            selection=OUT/'preflight_web_ids.json'
            write(selection,[shortest['question_id']])
            distributed('eval_jevany_streaming.py',['--controller',preflight/'step-000033','--benchmark','web',
                '--output-dir',smoke,'--selection-ids',selection,'--response-margin','0.0','--max-new-tokens','96',
                '--answer-model','promoted','--record-action-logits'],logs/'preflight_web.log',world=1)
            command([PY,'scripts/merge_jevany_benchmark.py','--input-dir',str(smoke),'--world-size','1'],logs/'preflight_web.log')
            command([PY,'scripts/score_exp10_web_ego.py','web','--pred-dir',str(smoke),'--expected-videos','1'],logs/'preflight_web_score.log')
        import torch
        state=torch.load(preflight/'step-000032/training_state.pt',weights_only=False,map_location='cpu')
        seconds_per_step=state['training_seconds']/32
        segment_steps=max(1,math.ceil(10800/seconds_per_step))
        if segment_steps*4*128 > audit['eligible_rows']//128*128:
            raise ValueError('estimated run exhausts pool')
        schedule=dict(seed=42,global_batch=128,backend='gloo',host=HOST,
            estimated_seconds_per_step=seconds_per_step,checkpoint_steps=[segment_steps*i for i in (1,2,3,4)],
            estimated_training_hours=segment_steps*4*seconds_per_step/3600,
            preflight_memory=memories,stress_memory=stress_memories,resume_equivalence=equivalence,
            step_budget_requested_by_user=True)
        write(schedule_path,schedule)
    else:
        schedule=json.loads(schedule_path.read_text())
    for phase, step in enumerate(schedule['checkpoint_steps'],1):
        train_out=OUT/'train'
        latest=json.loads((train_out/'latest.json').read_text()) if (train_out/'latest.json').exists() else None
        if latest is None or latest['step']<step:
            status('training',phase_number=phase,target_step=step,schedule=schedule['checkpoint_steps'])
            train(train_out,order,step,logs/f'train_phase{phase}.log',resume=latest is not None)
        elif latest['step']>step and not (OUT/f'evaluation/web_phase{phase}/pauc_f_summary-gpt-4.1-v1.json').exists():
            raise ValueError('training advanced before scheduled evaluation completed')
        checkpoint=train_out/f'step-{step:06d}'
        phase_stats=OUT/f'reports/train_phase{phase}'
        phase_stats.mkdir(parents=True,exist_ok=True)
        for rank in range(8):
            destination=phase_stats/f'summary.rank{rank}.json'
            if not destination.exists():
                shutil.copy2(train_out/f'summary.rank{rank}.json',destination)
        web=OUT/f'evaluation/web_phase{phase}'; web.mkdir(parents=True,exist_ok=True)
        if not (web/'pauc_f_summary-gpt-4.1-v1.json').exists():
            status('web_rollout',phase_number=phase,checkpoint=checkpoint)
            distributed('eval_jevany_streaming.py',['--controller',checkpoint,'--benchmark','web','--output-dir',web,
                '--response-margin','0.0','--max-new-tokens','96','--answer-model','promoted','--record-action-logits'],logs/f'web_phase{phase}.log')
            command([PY,'scripts/merge_jevany_benchmark.py','--input-dir',str(web),'--world-size','8'],logs/f'web_phase{phase}.log')
            status('web_scoring',phase_number=phase)
            command([PY,'scripts/score_exp10_web_ego.py','web','--pred-dir',str(web),'--expected-videos','500'],logs/f'score_phase{phase}.log')
        report(schedule)
    status('complete',checkpoint_steps=schedule['checkpoint_steps'],report=OUT/'reports/results.md')

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc())
        raise
