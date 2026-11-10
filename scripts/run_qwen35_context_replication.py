#!/usr/bin/env python3
"""One-node replication followed by preregistered, quality-gated full-source doses."""
import fcntl,json,os,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import run_qwen35_context_order_pilot as training
import evaluate_qwen35_context_order_pilot as evaluation
from qwen35_context_order import sha256
from analyze_qwen35_scaling_diagnostic import bootstrap_delta
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'stream-gemma/src'),str(ROOT.parent/'stream-gemma')]
OUT=ROOT/'outputs/jevany-qwen35-2b-context-replication-seed43'
PILOT=ROOT/'outputs/jevany-qwen35-2b-context-order-pilot'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def status(state,**kw):
    v=dict(state=state,host=socket.gethostname(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(OUT/'status.json',v);print(json.dumps(v),flush=True)
def replicate_passes(delta):
    return delta['delta_shuffled_minus_reference']>=1. and delta['ci95'][0]>0.
def million_passes(vs_replay,vs_anchor):
    return vs_replay['ci95'][0]>0. and vs_anchor['ci95'][0]>-1.
def clearly_worse(delta):return delta['ci95'][1]<-1.
def checkpoint(path,step):return path/f'step-{step:06d}'
def load_state(path):
    import torch
    return torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
def train(plan,name,output,order,target,parent,recipe,verify=False):
    if not verify and checkpoint(output,target).exists() and (output/'summary.rank0.json').exists():
        v=load_state(checkpoint(output,target))
        if v['step']==target and v.get('continuation_seed')==43:return v
    status('training' if not verify else 'verifying_exact_parent_restore',stage=name,target_step=target,global_batch=128,local_gpus=8)
    training.run_training(plan,name,output,order,target,parent=parent,parent_recipe=recipe,verify=verify)
    if verify:return
    v=load_state(checkpoint(output,target))
    if v['step']!=target or v['seen_per_rank']*8!=target*128 or v.get('continuation_seed')!=43:raise ValueError('checkpoint counter/seed mismatch')
    return v

def condition(name,arm,step,controller,cohort):
    return dict(name=f'{name}_step{step}_margin+1',arm=arm,step=step,margin=1.,controller=str(controller),
        manifest=str(OUT/f'evaluation/data/{cohort}.jsonl'),output=str(OUT/f'evaluation/{cohort}/{name}_step{step}_margin+1'))
def warm_generator():
    from huggingface_hub import snapshot_download
    from stream_gemma.config import QWEN_MODEL_NAME,QWEN_MODEL_REVISION
    directory=Path(snapshot_download(QWEN_MODEL_NAME,revision=QWEN_MODEL_REVISION,local_files_only=True))
    def read(path):
        n=0
        with path.open('rb',buffering=0) as f:
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_SEQUENTIAL)
            while block:=f.read(16<<20):n+=len(block)
        return n
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:n=sum(pool.map(read,sorted(directory.glob('*.safetensors'))))
    print(json.dumps(dict(weight_read_ahead_bytes=n,seconds=time.monotonic()-start)),flush=True)
def gpu_evaluate(config,log,teacher=None,teacher_output=None):
    previous=os.environ.get('TRITON_CACHE_DIR')
    os.environ['TRITON_CACHE_DIR']='/tmp/pace-context-eval-triton36'
    try:evaluation.distributed(config,log,teacher,teacher_output)
    finally:
        if previous is None:os.environ.pop('TRITON_CACHE_DIR',None)
        else:os.environ['TRITON_CACHE_DIR']=previous
def evaluate(conditions,cohort,label):
    config=OUT/f'evaluation/configs/{label}.json';evaluation.write(config,conditions,True)
    if not all((Path(c['output'])/'rollout_complete.json').exists() for c in conditions):
        status('closed_loop_evaluation',stage=label,cohort=cohort,conditions=len(conditions),margin=1.)
        warm_generator();gpu_evaluate(config,OUT/f'logs/eval_{label}.log')
    scores=[]
    for c in conditions:
        scores.append(evaluation.score(c,cohort));write(OUT/f'reports/{label}_scores.json',scores)
    return scores
def paired(left,right):
    result=bootstrap_delta(evaluation.per_video(left),evaluation.per_video(right))
    result['delta_shuffled_minus_reference']=result.pop('delta_final_minus_earlier')
    result['delta_definition']=f'{right["name"]} minus {left["name"]}'
    return result

def phase_probes(plan):
    source=PILOT/'evaluation/data/phase_probe.jsonl';dest=OUT/'evaluation/data/phase_probe.jsonl'
    if dest.exists() and sha256(dest)!=sha256(source):raise ValueError('phase-probe source changed')
    if not dest.exists():shutil.copyfile(source,dest)
    frozen=dict(primary_margin=1.,phase_steps=[1744,1760,1776,1792],phase_probe_sha256=sha256(dest),
        generator_kernels='Transformers PyTorch fallbacks; native MoE grouped GEMMs',periodic_stack_dumps=False,
        training_python=training.PY,evaluation_python=evaluation.GPU_PY)
    evaluation.write(OUT/'evaluation_protocol.json',frozen,True)
    phases={}
    for arm in ['reference','shuffled']:
        root=evaluation.PHASE_ROOT/arm;root.mkdir(parents=True,exist_ok=True)
        config=root/'conditions.json'
        cs=[condition('phase_'+arm,arm,step,checkpoint(OUT/f'replication/{arm}/train',step),'replication') for step in frozen['phase_steps']]
        evaluation.write(config,cs,True)
        gold=root/'gold'
        if not (gold/'complete.json').exists():
            if gold.exists():gold.rename(root/f'gold_incomplete_{time.time_ns()}')
            status('matched_gold_history_probes',arm=arm)
            gpu_evaluate(config,OUT/f'logs/phase_{arm}.log',dest,gold)
        phases[arm]=evaluation.phase_summary(arm,frozen['phase_steps'])
        write(OUT/'reports/phase_diagnostics.json',phases)
    return phases

def report_replication(scores,delta,phases,passed):
    write(OUT/'reports/replication_results.json',dict(scores=scores,paired_delta=delta,phase_diagnostics=phases,
        gate_passed=passed,primary_margin=1.,continuation_seed=43,independent_initialization_seed=False,
        gate='Point delta >= 1 and paired 95% CI lower > 0; conditional on this continuation state and fresh cohort.'))
    lines=['# Context-order continuation replication: seed 43','',
        'Two matched 448-update continuations from common step 1344 to 1792. Base checkpoint seed remains 42; rank-specific continuation RNG seeds are 43 + rank. This is not an independent from-scratch seed.',
        'Primary response margin +1 was fixed before scoring; the 64-video replication cohort is disjoint from the previous diagnostic and pilot cohorts.',
        '', '| Arm | PAUC-F@4s | Precision | Recall | Useful | Wasted | Identical repeats |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in scores:
        s=c['score'];lines.append(f"| {c['arm']} | {s['pauc_f']:.2f} | {s['fire_precision']:.2f} | {s['pauc_recall_component']:.2f} | {s['value_adding_fires']} | {s['wasted_fires']} | {c['identical_repeats']} |")
    lines+=['',f"Shuffled minus reference: {delta['delta_shuffled_minus_reference']:+.2f}, paired 95% interval [{delta['ci95'][0]:+.2f}, {delta['ci95'][1]:+.2f}].",'',f'Preregistered replication gate passed: {passed}.',
        'Video-bootstrap uncertainty excludes training-seed and judge repeatability. The availability-constrained heldout length mix differs from the first pilot; compare arms within the new cohort.',
        'Full-source scaling starts only after a positive replication. Its 1M-presentation fresh-versus-replay gate must pass before larger doses.']
    (OUT/'reports/replication_results.md').write_text('\n'.join(lines)+'\n')

def report_scaling(stages,decision,final=None):
    lines=['# Gated full-source scaling','',decision,'',
        'All runs use this node and its eight GPUs, fixed global batch 128, unchanged learning rates, and response margin +1.',
        '', '| Step | Presentations | Fresh development PAUC-F | Replay development PAUC-F |',
        '| ---: | ---: | ---: | ---: |']
    for stage in stages.values():
        scores={s['arm']:s['score']['pauc_f'] for s in stage['scores']}
        replay=f"{scores['replay']:.2f}" if 'replay' in scores else 'not extended'
        lines.append(f"| {stage['step']} | {stage['presentations']:,} | {scores['fresh']:.2f} | {replay} |")
    if final:
        lines+=['','## Frozen final confirmation','', '| Controller | PAUC-F@4s | Precision | Recall |','| --- | ---: | ---: | ---: |']
        for c in final['confirmation']:
            s=c['score'];lines.append(f"| {c['arm']} | {s['pauc_f']:.2f} | {s['fire_precision']:.2f} | {s['pauc_recall_component']:.2f} |")
        for key in ['full_minus_anchor','full_minus_million_fresh','full_minus_million_replay']:
            d=final[key];lines+=['',f"{key}: {d['delta_shuffled_minus_reference']:+.2f}, paired 95% interval [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]."]
        lines+=['',*final['limitations']]
    (OUT/'reports/scaling_results.md').write_text('\n'.join(lines)+'\n')

def main():
    if socket.gethostname().split('.')[0]!='ip-10-3-70-177':raise RuntimeError('wrong node: all work must remain on ip-10-3-70-177')
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'launch_plan.json').read_text())
    from pace_clm.evaluation_policy import require_raw_action_scores
    require_raw_action_scores(plan['primary_margin'])
    if plan['host']!=socket.gethostname().split('.')[0]:raise ValueError('frozen host mismatch')
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines()
    if len(gpus)!=8:raise ValueError('requires this node\'s eight GPUs')
    training.OUT=OUT;evaluation.OUT=OUT;evaluation.PHASE_ROOT=OUT/'evaluation/phase/torch210'
    os.environ.update(PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',PACE_DISTRIBUTED_BACKEND='gloo',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',TRITON_CACHE_DIR='/dev/shm/pace-context-rep43-training',TMPDIR='/dev/shm/pace-context-rep43-tmp',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    for key in ['TRITON_CACHE_DIR','TMPDIR']:Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    for name in ['replication','fresh','replay']:
        if sha256(OUT/f'plans/{name}.npy')!=plan['plan_hashes'][name]:raise ValueError('frozen order changed')
    if sha256(plan['reference_order'])!=plan['reference_order_sha256']:raise ValueError('reference order changed')
    for cohort,meta in plan['cohorts'].items():
        for suffix,key in [('.jsonl','manifest_sha256'),('.gold.json','gold_sha256')]:
            if sha256(OUT/f'evaluation/data/{cohort}{suffix}')!=meta[key]:raise ValueError('frozen cohort changed')
    for name in ['run_qwen35_context_replication.py','prepare_qwen35_context_replication.py','train_jevany_2b_batched.py',
                 'run_qwen35_context_order_pilot.py','eval_qwen35_development_sweep.py','evaluate_qwen35_context_order_pilot.py','qwen35_context_order.py','qwen35_diagnostic_kernels.py']:
        dest=OUT/'code'/name
        if not dest.exists():shutil.copyfile(ROOT/'scripts'/name,dest)
    initial=load_state(Path(plan['parent_checkpoint']))
    if initial['step']!=1344 or initial['seen_per_rank']*8!=172032:raise ValueError('parent cursor mismatch')
    verify=OUT/'preflight/exact_restore'
    if not all((verify/f'verified.rank{rank}.json').exists() for rank in range(8)):
        train(plan,'exact_restore',verify,plan['reference_order'],1792,Path(plan['parent_checkpoint']),Path(plan['parent_recipe']),True)
    states={}
    for arm in plan['training_arm_order']:
        path=OUT/f'replication/{arm}/train';order=plan['replication_order'] if arm=='shuffled' else plan['reference_order']
        states[arm]=train(plan,'replication_'+arm,path,order,1792,Path(plan['parent_checkpoint']),Path(plan['parent_recipe']))
    for rank in range(8):
        a=json.loads((OUT/f'replication/reference/train/continuation_rng.rank{rank}.json').read_text())
        b=json.loads((OUT/f'replication/shuffled/train/continuation_rng.rank{rank}.json').read_text())
        if a!=b or not a['reseeded'] or a['parent_rng_sha256']==a['continuation_rng_sha256']:raise ValueError('continuation RNG streams are not correctly matched/new')
    if states['reference']['global_labels']!=states['shuffled']['global_labels']:raise ValueError('replication label exposures differ')
    write(OUT/'replication_training_complete.json',dict(step=1792,continuation_seed=43,presentations_per_arm=57344,matched_rank_rng_streams=True))
    phases=phase_probes(plan)
    cs=[condition('replication_'+arm,arm,1792,checkpoint(OUT/f'replication/{arm}/train',1792),'replication') for arm in ['reference','shuffled']]
    scores=evaluate(cs,'replication','replication');delta=paired(scores[0],scores[1]);passed=replicate_passes(delta)
    report_replication(scores,delta,phases,passed)
    write(OUT/'replication_gate.json',dict(passed=passed,paired_delta=delta))
    if not passed:
        status('complete_replication_not_confirmed',report=str(OUT/'reports/replication_results.md'),scaling_launched=False);return
    scale_plan=dict(plan,save_every=256)
    anchor=checkpoint(OUT/'replication/shuffled/train',1792);anchor_recipe=OUT/'replication/shuffled/train/recipe.json'
    anchor_dev=evaluate([condition('replication_anchor','anchor',1792,anchor,'scaling_development')],'scaling_development','scaling_anchor')[0]
    previous={'fresh':(anchor,anchor_recipe),'replay':(anchor,anchor_recipe)};stages={}
    million_fresh=million_replay=None
    for step in plan['scaling']['milestones']:
        arms=['fresh','replay'] if step<=8192 else ['fresh']
        for arm in arms:
            path=OUT/f'scaling/{arm}/train';parent,recipe=previous[arm]
            train(scale_plan,f'scaling_{arm}_{step}',path,plan[f'{arm}_order'],step,parent,recipe)
            previous[arm]=(checkpoint(path,step),path/'recipe.json')
        cs=[condition('scaling_'+arm,arm,step,previous[arm][0],'scaling_development') for arm in arms]
        scored=evaluate(cs,'scaling_development',f'scaling_step{step}');by_arm={s['arm']:s for s in scored}
        stage=dict(step=step,presentations=step*128,scores=scored,vs_anchor=paired(anchor_dev,by_arm['fresh']))
        if step<=8192:stage['vs_replay']=paired(by_arm['replay'],by_arm['fresh'])
        else:stage['vs_million_fresh']=paired(million_fresh,by_arm['fresh'])
        stages[str(step)]=stage;write(OUT/'reports/scaling_stages.json',stages)
        report_scaling(stages,'Development dose evaluation is in progress; these are not final confirmation results.')
        if clearly_worse(stage['vs_anchor']) or (step<=8192 and clearly_worse(stage['vs_replay'])) or (step>8192 and clearly_worse(stage['vs_million_fresh'])):
            report_scaling(stages,'Stopped at a preregistered clear-regression boundary; the larger dose was not launched.')
            status('complete_scaling_regression',step=step,report=str(OUT/'reports/scaling_stages.json'));return
        if step==8192:
            passed=million_passes(stage['vs_replay'],stage['vs_anchor'])
            write(OUT/'million_presentation_gate.json',dict(passed=passed,stage=stage))
            if not passed:
                report_scaling(stages,'The 1M-presentation quality gate was inconclusive or failed; the full-source epoch was not launched.')
                status('complete_scaling_inconclusive',step=step,report=str(OUT/'reports/scaling_stages.json'),full_epoch_launched=False);return
            million_fresh=by_arm['fresh'];million_replay=by_arm['replay']
    end=plan['scaling']['milestones'][-1]
    final_conditions=[condition('full_fresh','full_fresh',end,previous['fresh'][0],'scaling_confirmation'),
        condition('million_fresh','million_fresh',8192,Path(million_fresh['controller']),'scaling_confirmation'),
        condition('million_replay','million_replay',8192,Path(million_replay['controller']),'scaling_confirmation'),
        condition('replication_anchor','anchor',1792,anchor,'scaling_confirmation')]
    final=evaluate(final_conditions,'scaling_confirmation','final_scaling_confirmation');by_arm={s['arm']:s for s in final}
    result=dict(plan=plan,replication_delta=delta,development_stages=stages,confirmation=final,
        full_minus_anchor=paired(by_arm['anchor'],by_arm['full_fresh']),full_minus_million_fresh=paired(by_arm['million_fresh'],by_arm['full_fresh']),
        full_minus_million_replay=paired(by_arm['million_replay'],by_arm['full_fresh']),
        limitations=['Conditional continuation seeds, not independently initialized full training runs.',
            'Development gates are repeated looks; final confirmation was held aside until recipe/dose completion.',
            'Replay control is update-matched through 1M presentations; larger-dose comparisons have unequal training budgets.',
            'Original full-source exclusions retained; 86 terminal rows reserved for full-batch alignment.'])
    write(OUT/'reports/full_scaling_results.json',result)
    report_scaling(stages,'The gated full-source dose and frozen confirmation evaluations completed.',result)
    status('complete_full_source_scaling',report=str(OUT/'reports/full_scaling_results.json'),step=end,presentations=end*128)
if __name__=='__main__':
    try:main()
    except BlockingIOError:
        raise SystemExit('This experiment already has an active runner; see status.json.')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise
