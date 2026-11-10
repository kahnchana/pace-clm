#!/usr/bin/env python3
"""Full 500-video WEB comparison of replication endpoints and the 54.67 baseline."""
from __future__ import annotations
import fcntl,hashlib,json,math,os,shutil,socket,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SG=ROOT.parent/'stream-gemma'
sys.path[:0]=[str(ROOT/'src'),str(SG/'src'),str(SG)]
OUT=ROOT/'outputs/jevany-qwen35-2b-raw-full-web'
PY=str(SG/'.venv/bin/python');GPU_PY=str(ROOT.parent/'lmms-eval/.venv/bin/python')
HOST='ip-10-3-70-177'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def write(path,value,frozen=False):
    path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if frozen and path.exists() and path.read_text()!=text:raise ValueError(f'frozen file differs: {path}')
    tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)
def status(state,**kw):
    v=dict(state=state,host=socket.gethostname(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(OUT/'status.json',v);print(json.dumps(v),flush=True)
def run(args,log):
    with log.open('a') as f:
        f.write(json.dumps({'command':list(map(str,args))})+'\n');f.flush()
        subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def freeze_plan():
    old=ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web'
    baseline=json.loads((old/'manifest.json').read_text())
    gold=SG/'datasets/proactive-videoqa-web/anno.json'
    if sha(gold)!=baseline['source_sha256']:raise ValueError('WEB gold changed from the 54.67 baseline')
    expected=baseline['question_ids']
    if len(expected)!=500 or len(set(expected))!=500:raise ValueError('historical WEB cohort incomplete')
    rep=ROOT/'outputs/jevany-qwen35-2b-context-replication-seed43/replication'
    best=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
    specs=[('baseline_margin0',best,0.),
        ('reference_margin0',rep/'reference/train/step-001792',0.),('shuffled_margin0',rep/'shuffled/train/step-001792',0.)]
    conditions=[]
    for name,path,margin in specs:
        for suffix in ['controller_adapter/adapter_model.safetensors','pointer_head.pt','model_identity.json']:
            if not (path/suffix).is_file():raise FileNotFoundError(path/suffix)
        conditions.append(dict(name=name,controller=str(path),benchmark='web',margin=margin,output=str(OUT/name)))
    hashes={c['name']:dict(adapter=sha(Path(c['controller'])/'controller_adapter/adapter_model.safetensors'),
        pointer=sha(Path(c['controller'])/'pointer_head.pt')) for c in conditions}
    plan=dict(host=HOST,local_gpus=8,benchmark='web',videos=500,question_ids=expected,gold=str(gold),gold_sha256=sha(gold),
        conditions=conditions,checkpoint_hashes=hashes,quality_floor=54.67,historical_2b_exact=54.665128505970486,
        historical_pace_qwen=57.47,primary_candidate_margin=0.,margin_selection='None. Raw learned scores only, per user instruction; no manual offsets or calibration.',
        scorer_version='pauc-f-v1',judge_model='gpt-4.1',omega=.5,max_new_tokens=96,answer_model='promoted',
        generator_adapter_sha256=baseline['original_generator_adapter_sha256'],historical_baseline_directory=str(old),
        runtime=dict(training_not_modified=True,evaluation_python=GPU_PY,torch='2.10.0+cu128',transformers='5.5.3',
            generator_kernels='Transformers PyTorch recurrence/convolution/gated norm; native MoE grouped GEMMs',periodic_stack_dump_timer=False),
        comparison_scope='Same 500 WEB IDs, raw learned action argmax, and official WEB scorer for all three models. Baseline 0 reproduces the historical setting. Historical PACE result is contextual, without a fresh paired PACE comparison.')
    write(OUT/'launch_plan.json',plan,True);write(OUT/'conditions.json',conditions,True)
    write(OUT/'authorization.json',dict(date='2026-11-08',source='User requested full WEB PAUC-F and comparison against 54.67, then explicitly prohibited heuristic runtime score changes. This replacement evaluates raw learned scores only; previous margin-adjusted results are retired. Prior approval covers needed GPT scoring.',
        node_scope='Only this node and its eight GPUs for inference.',gateway='https://gateway.salesforceresearch.ai/openai/process/v1/'),True)
    return plan

def score(condition,plan):
    from pace_clm.evaluation_policy import require_raw_action_scores
    require_raw_action_scores(condition['margin'])
    d=Path(condition['output']);summary=d/'pauc_f_summary-gpt-4.1-v1.json'
    if not (d/'rollout_complete.json').exists():raise ValueError('WEB rollout not complete')
    log=OUT/f'logs/score_{condition["name"]}.log'
    run([PY,'scripts/merge_jevany_benchmark.py','--input-dir',d,'--world-size','8'],log)
    if not summary.exists():
        status('gpt_scoring',condition=condition['name'],benchmark='web',videos=500)
        run([PY,'scripts/score_exp10_web_ego.py','web','--pred-dir',d,'--expected-videos','500'],log)
    from run_qwen35_control import verify_web
    value,rows,_=verify_web(d)
    manifest=json.loads((d/'manifest.json').read_text())
    if manifest['split']!='web' or manifest['rows']!=500 or manifest['question_ids']!=plan['question_ids'] or manifest['source_sha256']!=plan['gold_sha256']:raise ValueError('not the frozen full WEB cohort')
    if manifest['response_margin']!=condition['margin'] or manifest['max_new_tokens']!=96 or manifest['history']!='self-generated':raise ValueError('WEB inference protocol mismatch')
    expected=plan['checkpoint_hashes'][condition['name']]
    if manifest['controller_adapter_sha256']!=expected['adapter'] or manifest['pointer_head_sha256']!=expected['pointer']:raise ValueError('checkpoint changed')
    if manifest['original_generator_adapter_sha256']!=plan['generator_adapter_sha256']:raise ValueError('generator weights changed')
    decisions=0
    for r in rows:
        trace=r['jevany_action_trace'];decisions+=len(trace)
        if len(trace)!=sum(r['jevany_actions'].values()):raise ValueError('incomplete action trace')
        if any(not math.isfinite(v) for t in trace for v in t['logits']):raise ValueError('nonfinite action logits')
        if r['jevany_actions']['Response']!=len(r['model_response_list']):raise ValueError('fire-count mismatch')
    if decisions!=8803:raise ValueError(f'WEB decision grid differs from historical protocol: {decisions}')
    result=dict(**condition,score=value,web_decisions=decisions,meets_54_67=value['pauc_f']>=54.67,
        delta_vs_54_67=value['pauc_f']-54.67,delta_vs_historical_pace=value['pauc_f']-57.47)
    write(OUT/f'reports/{condition["name"]}.json',result);return result

def compare(left,right):
    from report_qwen35_representative import bootstrap
    result=bootstrap(Path(left['output']),Path(right['output']))
    expected=right['score']['pauc_f']-left['score']['pauc_f']
    if abs(result['delta_b_minus_a']-expected)>1e-8:raise ValueError('paired bootstrap does not reproduce official score difference')
    result['definition']=right['name']+' minus '+left['name'];return result

def report(plan,results):
    by_name={r['name']:r for r in results};comparisons={}
    for name in ['reference_margin0','shuffled_margin0']:
        for baseline in ['baseline_margin0']:
            comparisons[name+'_minus_'+baseline]=compare(by_name[baseline],by_name[name])
    comparisons['shuffled_minus_reference']=compare(by_name['reference_margin0'],by_name['shuffled_margin0'])
    old=Path(plan['historical_baseline_directory'])
    comparisons['runtime_baseline0_minus_historical_baseline0']=compare(
        dict(name='historical_baseline0',output=str(old),score={'pauc_f':plan['historical_2b_exact']}),by_name['baseline_margin0'])
    payload=dict(plan=plan,results=results,paired_comparisons=comparisons,
        accepted_candidates=[r['name'] for r in results if r['name'] in ['reference_margin0','shuffled_margin0'] and r['meets_54_67']],
        limitations=['Quality floor is the user-specified 54.67 score; passing this floor alone does not establish improvement over the same-runtime baseline.',
            'Paired uncertainty covers video sampling, not training seeds or judge repeatability.',
            'PACE-Qwen 57.47 is historical context; its raw outputs are unavailable for a fresh paired comparison.',
            'All decisions use raw learned scores. Nonzero action-score margins are prohibited; no checkpoint or threshold selection on WEB.',
            'Prior margin-adjusted selection and scaling decisions are retired; this evaluation does not automatically launch training.'])
    write(OUT/'reports/results.json',payload)
    lines=['# Full WEB: second-continuation-seed checkpoints','',
        'All three conditions cover the same full 500 WEB videos and 8,803 decisions, with GPT-4.1, the official pauc-f-v1 scorer, omega 0.5, 32 causal frames, generated history, and greedy promoted-MoE answers capped at 96 tokens.',
        'All GPU computation ran on ip-10-3-70-177 and its eight H200s. Every model chooses the raw learned action argmax; no logit offsets, manual margins, or runtime calibration are applied. The saved historical 54.67 checkpoint is rerun under the same evaluation runtime.',
        '', '| Model / setting | WEB PAUC-F | Precision | Recall | Useful / wasted fires | Meets 54.67 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in results:
        s=r['score'];lines.append(f"| {r['name']} | {s['pauc_f']:.2f} | {s['fire_precision']:.2f} | {s['pauc_recall_component']:.2f} | {s['num_value_adding_fires']} / {s['num_wasted_fires']} | {r['meets_54_67']} |")
    lines+=['','Historical reference: 2B J1-S 54.67; promoted PACE-Qwen 57.47.']
    for name,c in comparisons.items():lines+=['',f"{name}: {c['delta_b_minus_a']:+.2f}, paired 95% interval [{c['delta_ci95'][0]:+.2f}, {c['delta_ci95'][1]:+.2f}]."]
    lines+=['',*payload['limitations']]
    (OUT/'reports/results.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ['logs','reports','code']:(OUT/d).mkdir(exist_ok=True)
    if socket.gethostname().split('.')[0]!=HOST:raise RuntimeError('wrong node')
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if len(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines())!=8:raise RuntimeError('requires exactly eight local GPUs')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs are occupied; no other jobs will be stopped')
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',
        PACE_DISTRIBUTED_BACKEND='gloo',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',TRITON_CACHE_DIR='/tmp/pace-context-eval-triton36')
    plan=freeze_plan()
    for name in ['run_qwen35_replication_full_web.py','eval_qwen35_development_sweep.py','eval_jevany_streaming.py','score_exp10_web_ego.py','merge_jevany_benchmark.py','qwen35_diagnostic_kernels.py']:
        dest=OUT/'code'/name
        if not dest.exists():shutil.copyfile(ROOT/'scripts'/name,dest)
    shutil.copyfile(ROOT/'src/pace_clm/evaluation_policy.py',OUT/'code/evaluation_policy.py')
    from run_qwen35_context_replication import warm_generator
    status('loading_full_web_evaluation',conditions=3,videos_per_condition=500,quality_floor=54.67,decision_policy='raw_learned_scores_only')
    warm_generator()
    conditions=plan['conditions'];scored=set()
    args=[GPU_PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        'scripts/eval_qwen35_development_sweep.py','--conditions',OUT/'conditions.json','--require-native-moe','--generator-kernels','torch']
    stop=threading.Event()
    def monitor():
        with (OUT/'logs/gpu_samples.csv').open('a') as f:
            while not stop.is_set():
                subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],stdout=f,stderr=subprocess.DEVNULL);f.flush();stop.wait(10)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    with (OUT/'logs/rollouts.log').open('a') as f:
        f.write(json.dumps({'command':list(map(str,args))})+'\n');f.flush()
        process=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                for c in conditions:
                    if c['name'] not in scored and (Path(c['output'])/'rollout_complete.json').exists():
                        score(c,plan);scored.add(c['name'])
                        status('full_web_rollouts',completed_scores=len(scored),conditions=3)
                time.sleep(5)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,args)
        finally:
            if process.poll() is None:process.terminate();process.wait()
            stop.set();thread.join(timeout=15)
    results=[score(c,plan) for c in conditions];report(plan,results)
    status('complete',report=str(OUT/'reports/results.md'),quality_floor=54.67,
        accepted_candidates=[r['name'] for r in results if r['name'] in ['reference_margin0','shuffled_margin0'] and r['meets_54_67']])
if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit('This full-WEB comparison already has an active runner.')
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise
