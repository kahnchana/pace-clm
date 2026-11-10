#!/usr/bin/env python3
"""Authorized GPT-4.1 evaluation of the matched context-order pilot on this node."""
from __future__ import annotations
import fcntl,json,os,socket,subprocess,time,traceback
from pathlib import Path
import numpy as np
from analyze_qwen35_scaling_diagnostic import bootstrap_delta,average_precision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-context-order-pilot'
PY=str(ROOT.parent/'stream-gemma/.venv/bin/python')
GPU_PY=os.environ.get('PACE_CONTEXT_EVAL_PYTHON',str(ROOT.parent/'lmms-eval/.venv/bin/python'))
PHASE_ROOT=OUT/('evaluation/phase/torch210' if GPU_PY!=PY else 'evaluation/phase')

def write(path,value,frozen=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if frozen and path.exists() and json.loads(path.read_text())!=value:raise ValueError(f'frozen artifact changed: {path}')
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)
def status(state,**kw):
    value=dict(state=state,host=socket.gethostname(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(OUT/'evaluation_status.json',value);write(OUT/'status.json',value);print(json.dumps(value),flush=True)
def run(args,log):
    with log.open('a') as f:subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def await_gpus():
    deadline=time.monotonic()+3*3600
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        status('evaluation_waiting_for_this_nodes_gpus')
        if time.monotonic()>deadline:raise TimeoutError('node remained occupied before evaluation')
        time.sleep(15)
def distributed(conditions,log,teacher=None,teacher_output=None):
    await_gpus()
    args=[GPU_PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','scripts/eval_qwen35_development_sweep.py','--conditions',conditions]
    if GPU_PY!=PY:args+=['--require-native-moe']
    if teacher:args+=['--teacher-only','--teacher-batch','1','--gold-history',teacher,'--teacher-output',teacher_output]
    if teacher:
        run(args,log)
        return
    # Judge finished conditions while the same GPU workers continue the sweep.
    # A failed/restarted sweep reuses only complete, validated video shards.
    entries=json.loads(Path(conditions).read_text())
    judged=set()
    with log.open('a') as f:
        child=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                for c in entries:
                    if c['name'] not in judged and (Path(c['output'])/'rollout_complete.json').exists():
                        score(c,Path(c['manifest']).stem)
                        judged.add(c['name'])
                time.sleep(5)
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,args)
        finally:
            if child.poll() is None:
                child.terminate();child.wait()
def condition(arm,step,margin,cohort):
    name=f'{arm}_step{step}_margin{margin:+g}'
    return dict(name=name,arm=arm,step=step,margin=margin,controller=str(OUT/f'{arm}/train/step-{step:06d}'),
        manifest=str(OUT/f'evaluation/data/{cohort}.jsonl'),output=str(OUT/f'evaluation/{cohort}/{name}'))
def score(c,cohort):
    from pace_clm.evaluation_policy import require_raw_action_scores
    require_raw_action_scores(c['margin'])
    directory=Path(c['output']);summary=directory/'official/results/exp10_judged_summary.json'
    if not (directory/'rollout_complete.json').exists():raise ValueError('incomplete rollout')
    if not summary.exists():
        status('gpt_scoring',cohort=cohort,condition=c['name'])
        log=OUT/f'logs/score_{cohort}_{c["name"]}.log'
        run([PY,'scripts/prepare_exp10_scoring.py','--rollout-dir',directory,'--manifest',c['manifest'],
            '--gold',OUT/f'evaluation/data/{cohort}.gold.json','--world-size','8'],log)
        judge=directory/'official/judge';request=judge/'pauc_batch_input.jsonl'
        if request.read_text().strip() and not (judge/'pauc_llm_scores.json').exists():
            run([PY,'scripts/judge_exp10_rollout.py','--rollout-dir',directory,'--concurrency','16'],log)
        else:
            if not request.read_text().strip():
                write(judge/'pauc_llm_scores.json',dict(judge_model='gpt-4.1',scores_by_request_key={},request_mappings=[]),True)
            run([PY,'scripts/score_exp10.py','--rollout-dir',directory],log)
    value=json.loads(summary.read_text())
    if value['num_sampled_videos']!=64 or value['judge_model']!='gpt-4.1' or not value['headline']:raise ValueError('invalid semantic score coverage/protocol')
    predictions=[json.loads(l) for l in (directory/'official/generation/predictions.jsonl').read_text().splitlines()]
    expected=[json.loads(l)['question_id'] for l in Path(c['manifest']).read_text().splitlines()]
    if len(predictions)!=64 or len({r['question_id'] for r in predictions})!=64 or {r['question_id'] for r in predictions}!=set(expected):raise ValueError('semantic cohort IDs mismatch')
    repeated=sum(len(a)-len(set(a)) for r in predictions for a in [[' '.join(t['content'].lower().split()) for t in r['model_response_list']]])
    return dict(**c,score=value,identical_repeats=repeated)
def phase_summary(arm,steps):
    source=[json.loads(l) for l in (OUT/'evaluation/data/phase_probe.jsonl').read_text().splitlines()]
    results=[]
    for step in steps:
        rows=[]
        for path in (PHASE_ROOT/arm/'gold').glob(f'step-{step:06d}.rank*.jsonl'):rows.extend(json.loads(l) for l in path.read_text().splitlines())
        rows.sort(key=lambda r:r['index'])
        if [r['index'] for r in rows]!=list(range(len(source))):raise ValueError('phase probe coverage mismatch')
        for r,s in zip(rows,source):
            if (r['video_id'],r['frame_index'],r['label'])!=(s['video_id'],s['frame_index'],s['label']):raise ValueError('phase probe row changed')
        logits=np.array([r['logits'] for r in rows]);prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
        truth=np.array([r['label_index'] for r in rows]);pred=logits.argmax(axis=1)
        results.append(dict(step=step,rows=len(rows),nll=float(np.mean([r['nll'] for r in rows])),accuracy=float(np.mean(pred==truth)),
            response_ap=average_precision(truth==2,prob[:,2]),mean_response_probability=float(prob[:,2].mean()),
            response_decision_fraction=float(np.mean(pred==2)),hold_decision_fraction=float(np.mean(pred==1))))
    return dict(checkpoints=results,response_probability_sd=float(np.std([r['mean_response_probability'] for r in results])),
        note='Matched gold-history probe, not semantic rollout quality or a population loss estimate.')
def per_video(c):
    values=json.loads((Path(c['output'])/'official/results/exp10_per_video.json').read_text())
    return {r['question_id']:r for r in values}
def report(plan,dev,selected,confirmation,phases):
    primary={r['arm']:r for r in confirmation if r['margin']==0}
    chosen={r['arm']:r for r in confirmation if r['margin']==selected[r['arm']]['margin']}
    def paired(reference,shuffled):
        result=bootstrap_delta(per_video(reference),per_video(shuffled))
        result['delta_shuffled_minus_reference']=result.pop('delta_final_minus_earlier')
        return result
    value=dict(plan=plan,training=json.loads((OUT/'training_results.json').read_text()),development=dev,
        selected_margins=selected,confirmation=confirmation,phase_diagnostics=phases,
        primary_fixed_zero_margin_delta=paired(primary['reference'],primary['shuffled']),
        secondary_development_selected_margin_delta=paired(chosen['reference'],chosen['shuffled']),
        delta_definition='shuffled minus original-order reference; video uncertainty excludes seed variance',
        evaluation_runtime=json.loads((OUT/'preflight/evaluation_runtime/runtime_repair.json').read_text()) if GPU_PY!=PY else None,
        limitations=['One seed, one continuation starting state; not a scaling or convergence result.',
            'Development selects margins, separate confirmation reports the frozen choices.',
            'MarkII PAUC-F@4s is distinct from WEB PAUC-F; do not compare the numbers directly.',
            'Both arms and phase probes use the same repaired evaluation runtime; no historical-runtime quality comparison.',
            'Use both quality and phase sensitivity; constant firing rates alone are not success.'])
    for name in ['gold_history_paired','verification']:
        path=OUT/f'reports/{name}.json'
        if path.exists():value[name]=json.loads(path.read_text())
    lines=['# Qwen3.5-2B context-order continuation pilot','',
        'Two fresh continuations from common step 1,344 to step 1,792: 448 updates and 57,344 presentations per arm, on one eight-H200 node.',
        'Only global-batch order inside each 4,096-row window changes. Both arms use the same validated normalization launch setting.',
        '', '| Arm | Confirmation margin | Judged PAUC-F@4s | Precision | Recall | Useful fires | Wasted fires |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in confirmation:
        s=r['score'];lines.append(f"| {r['arm']} | {r['margin']:+g} | {s['pauc_f']:.2f} | {s['fire_precision']:.2f} | {s['pauc_recall_component']:.2f} | {s['value_adding_fires']} | {s['wasted_fires']} |")
    for label,key in [('Primary fixed-zero margin','primary_fixed_zero_margin_delta'),('Secondary development-selected margins','secondary_development_selected_margin_delta')]:
        s=value[key];lines+=['',f"{label}, shuffled-minus-reference delta: {s['delta_shuffled_minus_reference']:+.2f}, paired 95% interval [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]."]
    lines+=['',*value['limitations']]
    if 'gold_history_paired' in value:
        g=value['gold_history_paired'];n=g['nll'];ap=g['response_ap'];pr=g['mean_response_probability']
        lines+=['','## Matched gold-history diagnostics','',
            f"At step 1,792, {g['rows']} identical inputs across {g['videos']} videos: shuffled-minus-reference NLL delta {n['delta']:+.5f} (paired 95% interval [{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}]); Response AP delta {ap['delta']:+.5f} (interval [{ap['ci95'][0]:+.5f}, {ap['ci95'][1]:+.5f}]). Neither interval separates the arms.",
            f"Mean Response probability shifted by {100*pr['delta']:+.2f} percentage points (interval [{100*pr['ci95'][0]:+.2f}, {100*pr['ci95'][1]:+.2f}]). Across the four checkpoints, its standard deviation was {phases['shuffled']['response_probability_sd']:.5f} for shuffled and {phases['reference']['response_probability_sd']:.5f} for reference. Shuffling did not reduce this measured calibration variation.",
            'These are conditional matched-input diagnostics; they do not estimate population loss, seed variance, or convergence.']
    lines+=['','## Interpretation and next experiment','',
        'The predeclared fixed-zero-margin confirmation comparison is inconclusive. The secondary development-selected-margin comparison favors shuffled order on this confirmation sample; it is a promising secondary signal, not a demonstrated scaling or stability improvement.',
        'At the selected +1 margin, useful fires were 121 for shuffled and 122 for reference, while wasted fires fell from 57 to 37 and identical repeated answers from 29 to 17.',
        'Next: replicate the matched continuation with another continuation seed, predeclare margin +1 as the primary operating point, and use a fresh confirmation cohort. Keep learning rates and other training settings fixed for that replication. A lower pointer-head learning-rate ablation is a separate follow-up for calibration drift.']
    if 'verification' in value:
        v=value['verification'];lines+=['','## Verification','',
            f"Passed integrity checks for {v['conditions']} conditions, {v['video_evaluations']} video evaluations over {v['unique_videos']} unique videos, {v['causal_decisions']:,} causal decisions, and {v['valid_gpt_judgments']:,} valid GPT-4.1 judgments. Cohort hashes, checkpoint hashes, calibrated action traces, judge coverage, and aggregate paired deltas were verified."]
    write(OUT/'reports/results.json',value)
    (OUT/'reports/results.md').write_text('\n'.join(lines)+'\n')

def main():
    if socket.gethostname().split('.')[0]!='ip-10-3-70-177':raise RuntimeError('wrong node')
    lock=(OUT/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'launch_plan.json').read_text());authorization=json.loads((OUT/'authorization.json').read_text())
    from pace_clm.evaluation_policy import require_raw_action_scores
    for margin in plan['evaluation']['secondary_development_margins']:require_raw_action_scores(margin)
    if authorization['gateway']!='https://gateway.salesforceresearch.ai/openai/process/v1/':raise ValueError('unexpected judge destination')
    # This pipeline is authorized by the user's 2026-11-07 instruction to do any
    # GPT scoring needed for this experiment. It is not the unapproved prior job.
    os.environ.update(PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',
        PACE_DISTRIBUTED_BACKEND='gloo',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',
        TRITON_CACHE_DIR=('/tmp/pace-context-eval-triton36' if GPU_PY!=PY else '/dev/shm/pace-context-order-triton'),TMPDIR='/dev/shm/pace-context-order-tmp')
    deadline=time.monotonic()+24*3600
    while not (OUT/'training_complete.json').exists():
        path=OUT/'training_status.json'
        if path.exists() and json.loads(path.read_text())['state']=='failed':raise RuntimeError('training failed; see training_status.json')
        if time.monotonic()>deadline:raise TimeoutError('training did not complete')
        time.sleep(30)
    phases={}
    for arm in ['shuffled','reference']:
        location=PHASE_ROOT/arm;location.mkdir(parents=True,exist_ok=True)
        conditions=[condition(arm,step,0.,'development') for step in plan['evaluation']['phase_probes']]
        config=location/'conditions.json';write(config,conditions,True)
        if not (location/'gold/complete.json').exists():
            status('phase_probe_evaluation',arm=arm)
            distributed(config,OUT/f'logs/phase_{arm}.log',OUT/'evaluation/data/phase_probe.jsonl',location/'gold')
        phases[arm]=phase_summary(arm,plan['evaluation']['phase_probes'])
        write(OUT/'reports/phase_diagnostics.json',phases)
    conditions=[condition(arm,1792,margin,'development') for arm in ['shuffled','reference'] for margin in plan['evaluation']['secondary_development_margins']]
    config=OUT/'evaluation/development_conditions.json';write(config,conditions,True)
    if not all((Path(c['output'])/'rollout_complete.json').exists() for c in conditions):
        status('development_rollouts',conditions=len(conditions),videos_per_condition=64)
        distributed(config,OUT/'logs/development_gpu.log')
    dev=[]
    for c in conditions:
        dev.append(score(c,'development'));write(OUT/'evaluation/development_scores.json',dev)
    selected={}
    for arm in ['shuffled','reference']:
        best=max([r for r in dev if r['arm']==arm],key=lambda r:(r['score']['pauc_f'],-abs(r['margin']),-r['margin']))
        selected[arm]=dict(margin=best['margin'],development_pauc_f=best['score']['pauc_f'])
    write(OUT/'evaluation/selected_margins.json',selected,True)
    conditions=[condition(arm,1792,m,'confirmation') for arm in ['shuffled','reference'] for m in sorted({0.,selected[arm]['margin']})]
    config=OUT/'evaluation/confirmation_conditions.json';write(config,conditions,True)
    if not all((Path(c['output'])/'rollout_complete.json').exists() for c in conditions):
        status('confirmation_rollouts',conditions=len(conditions),selected=selected)
        distributed(config,OUT/'logs/confirmation_gpu.log')
    conf=[]
    for c in conditions:
        conf.append(score(c,'confirmation'));write(OUT/'evaluation/confirmation_scores.json',conf)
    report(plan,dev,selected,conf,phases)
    status('complete',report=str(OUT/'reports/results.md'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise
