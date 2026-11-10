#!/usr/bin/env python3
"""Retired margin-sweep orchestration; historical implementation retained for audit."""
from __future__ import annotations

RETIREMENT_MESSAGE = 'Retired: nonzero response-margin sweeps are disabled by the raw learned-score policy. Historical code/artifacts are retained for audit only. Use scripts/eval_qwen35_development_sweep.py with zero-margin conditions for current evaluation.'

def main():
    """Reject the retired launcher before imports, file writes, GPU work or judging."""
    raise SystemExit(RETIREMENT_MESSAGE)

if __name__ == "__main__":
    main()

import argparse,fcntl,json,os,socket,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-mixture-12h/scaling-diagnostic'
PY=str(ROOT.parent/'stream-gemma/.venv/bin/python')
LOCAL_ONLY=False

def write(path,v):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def status(state,**kw):
    v=dict(state=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),**kw)
    write(OUT/'status.json',v);print(json.dumps(v),flush=True)
def run(argv,log):
    with log.open('a') as f:subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def score(c,cohort):
    directory=Path(c['output']);summary=directory/'official/results/exp10_judged_summary.json'
    if summary.exists():return json.loads(summary.read_text())
    status('scoring',condition=c['name'],cohort=cohort)
    log=OUT/f'logs/score_{cohort}_{c["name"]}.log'
    run([PY,'scripts/prepare_exp10_scoring.py','--rollout-dir',directory,'--manifest',OUT/f'data/{cohort}.jsonl',
        '--gold',OUT/f'data/{cohort}.gold.json','--world-size','8'],log)
    if LOCAL_ONLY:
        value=json.loads((directory/'official/results/timing_prejudge.json').read_text())
        value.pop('per_video',None)
        value.update(metric='timing-only PAUC-F@4s; assumes in-window answers are correct',headline=False)
        return value
    judge=directory/'official/judge'
    requests=[json.loads(l) for l in (judge/'pauc_batch_input.jsonl').read_text().splitlines() if l.strip()]
    if requests:
        run([PY,'scripts/judge_exp10_rollout.py','--rollout-dir',directory,'--concurrency','16'],log)
    else:
        write(judge/'pauc_llm_scores.json',dict(judge_model='gpt-4.1',scores_by_request_key={},request_mappings=[]))
        run([PY,'scripts/score_exp10.py','--rollout-dir',directory],log)
    value=json.loads(summary.read_text())
    if value['num_sampled_videos']!=32 or not value['headline']:raise ValueError('incomplete diagnostic scoring')
    return value

def await_conditions(conditions,cohort):
    result=[];deadline=time.monotonic()+8*3600
    for c in conditions:
        while not (Path(c['output'])/'rollout_complete.json').exists():
            if (OUT/'gpu_failure.json').exists():raise RuntimeError((OUT/'gpu_failure.json').read_text())
            if time.monotonic()>deadline:raise TimeoutError('GPU rollout did not complete')
            status('waiting_for_rollout',cohort=cohort,condition=c['name'])
            time.sleep(30)
        result.append(dict(**c,score=score(c,cohort)))
        write(OUT/f'{cohort}_scores.json',result)
    return result

def _historical_main():
    global LOCAL_ONLY
    parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true')
    parser.add_argument('--semantic-approved',action='store_true',help='Use only after explicit user approval for the recorded text export')
    args=parser.parse_args()
    LOCAL_ONLY=args.local_only
    if not LOCAL_ONLY and not args.semantic_approved:
        raise PermissionError('Semantic judging was rejected by automatic approval review. Obtain explicit user approval before using --semantic-approved.')
    if socket.gethostname().split('.')[0]!='ip-10-3-70-177':raise RuntimeError('wrong node')
    lock=(OUT/'orchestrator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',
        PACE_DISTRIBUTED_BACKEND='gloo',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',TORCHINDUCTOR_COMPILE_THREADS='1',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
        TRITON_CACHE_DIR='/dev/shm/pace-qwen35-diagnostic-triton',TMPDIR='/dev/shm/pace-qwen35-diagnostic-tmp')
    for key in ['TRITON_CACHE_DIR','TMPDIR']:Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    conditions=json.loads((OUT/'development_conditions.json').read_text())
    dev=await_conditions(conditions,'development')
    if LOCAL_ONLY:
        while not (OUT/'gold_history/complete.json').exists():time.sleep(20)
        status('local_analysis_semantic_judging_not_authorized')
        run([PY,'scripts/analyze_qwen35_scaling_diagnostic.py','--local-only'],OUT/'logs/analysis.log')
        status('awaiting_semantic_approval',report=str(OUT/'reports/diagnostic.json'),
            request_files=[str(Path(c['output'])/'official/judge/pauc_batch_input.jsonl') for c in conditions])
        return
    selected=[];confirmation=[]
    for step in [1332,1776]:
        candidates=[r for r in dev if r['step']==step]
        best=max(candidates,key=lambda r:(r['score']['pauc_f'],-abs(r['margin']),-r['margin']))
        selected.append(dict(step=step,margin=best['margin'],development_pauc_f=best['score']['pauc_f']))
        for margin in sorted({0.,best['margin']}):
            name=f'step{step}_margin{margin:+g}'
            confirmation.append(dict(name=name,step=step,margin=margin,controller=best['controller'],
                manifest=str(OUT/'data/confirmation.jsonl'),output=str(OUT/f'rollouts/confirmation/{name}')))
    # No confirmation output is inspected before the selected settings are persisted.
    frozen=OUT/'selected_margins.json'
    if frozen.exists() and json.loads(frozen.read_text())!=selected:raise ValueError('frozen margin selection changed')
    write(frozen,selected);write(OUT/'confirmation_conditions.json',confirmation)
    status('waiting_for_gold_history_and_gpu_release',selected=selected)
    while not (OUT/'gold_history/complete.json').exists():time.sleep(20)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
    status('confirmation_rollouts',selected=selected)
    if not all((Path(c['output'])/'rollout_complete.json').exists() for c in confirmation):
        run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            'scripts/eval_qwen35_development_sweep.py','--conditions',OUT/'confirmation_conditions.json'],OUT/'logs/confirmation_gpu.log')
    conf=await_conditions(confirmation,'confirmation')
    status('local_analysis')
    run([PY,'scripts/analyze_qwen35_scaling_diagnostic.py'],OUT/'logs/analysis.log')
    status('complete',selected=selected,report=str(OUT/'reports/diagnostic.json'))
