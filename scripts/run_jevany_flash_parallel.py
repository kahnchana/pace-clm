#!/usr/bin/env python3
"""Four resumable Flash-backed step-160 suites on allocation 38177."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from eval_jevany_tv_vad160 import ROOT, SG, PARENT, read, rows, sha, write
from stream_gemma.subtitles import bin_subtitles

NODE = 'ip-10-3-120-167'
ALLOCATION = 38177
DEFAULT = PARENT/'evaluation/web-ego-tv-subs-vad-flash-step160'
DOMAINS = ('web', 'ego', 'tv', 'vad')
SOURCES = {'web': PARENT/'evaluation/web-sol-step160',
           'ego': PARENT/'evaluation/ego-tv-vad-sol-step160/ego',
           'tv': PARENT/'evaluation/tv-subtitles-sol-step160',
           'vad': PARENT/'evaluation/ego-tv-vad-sol-step160/vad'}

def status(directory, state, **fields):
    write(directory/'status.json', dict(state=state,node=NODE,allocation=ALLOCATION,
          utc=datetime.now(timezone.utc).isoformat(),**fields))

def verify(directory):
    frozen = read(directory/'freeze.json')
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('frozen source changed: '+path)
    return frozen

def prepare(directory, smoke=False):
    from pace_clm.rl_anchor import verify_committed_checkpoint
    checkpoint = PARENT/'step-000160'
    verify_committed_checkpoint(checkpoint,160)
    if (directory/'freeze.json').exists():
        frozen=verify(directory)
        if frozen['smoke'] != smoke: raise ValueError('smoke/full mismatch')
        return
    directory.mkdir(parents=True,exist_ok=True)
    paths = {}
    for source in ('web-sol-step160','ego-tv-vad-sol-step160','tv-subtitles-sol-step160'):
        paths.update(read(PARENT/'evaluation'/source/'freeze.json')['source_hashes'])
    # Verify all reused inputs and implementation sources against their original freezes.
    for path,digest in paths.items():
        if sha(path)!=digest: raise ValueError('original frozen source changed: '+path)
    parent_specs={s['span_id']:s for s in rows(PARENT/'data/evaluation-specs.jsonl')}
    suites = {}
    for domain,expected in [('web',500),('ego',326),('tv',500),('vad',101)]:
        source=SOURCES[domain]
        selected=rows(source/'evaluation-manifest.jsonl')
        if len(selected)!=expected: raise ValueError('incomplete source: '+domain)
        cohorts=read(source/'cohorts.json')
        if smoke:
            selected=sorted(selected,key=lambda s:len(s['inference_source_indices']))[:8]
            for spec in selected:
                spec['inference_source_indices']=spec['inference_source_indices'][:32]
                spec['inference_timestamps']=spec['inference_timestamps'][:32]
                if domain=='tv':
                    spec['expected_subtitles_delivered']=sum(map(len,bin_subtitles(spec['extra_turns'],spec['inference_timestamps'])))
            selected=sorted(selected,key=lambda s:s['question_id'])
            chosen={s['question_id'] for s in selected}
            cohorts={k:[q for q in ids if q in chosen] for k,ids in cohorts.items()}
            cohorts={k:v for k,v in cohorts.items() if v}
        ids=[s['question_id'] for s in selected]
        if ids!=sorted(ids) or len(ids)!=len(set(ids)): raise ValueError('unordered/duplicate IDs')
        target=directory/domain;target.mkdir(exist_ok=True)
        gold=[dict(question_id=s['question_id'],conversation=[dict(role='user',content=s['question'])],answer=s.get('gold_answers',parent_specs.get(s['question_id'],{}).get('gold_answers'))) for s in selected]
        if any(g['answer'] is None for g in gold): raise ValueError('missing native gold')
        (target/'evaluation-manifest.jsonl').write_text(''.join(json.dumps(s,sort_keys=True)+'\n' for s in selected))
        write(target/'selection.json',ids);write(target/'cohorts.json',cohorts);write(target/'gold.json',gold)
        for name in ('evaluation-manifest.jsonl','selection.json','cohorts.json','gold.json'):
            path=target/name;paths[str(path.resolve())]=sha(path)
        suites[domain]=dict(episodes=len(ids),sol_baseline=str(source/'result.json'),decisions=sum(len(s['inference_source_indices']) for s in selected))
    for path in [Path(__file__),ROOT/'scripts/run_jevany_flash_parallel.sh',ROOT/'scripts/eval_jevany_flash_streaming.py',ROOT/'scripts/jevany_gemini_answer.py']:
        paths[str(path.resolve())]=sha(path)
    base=read(PARENT/'evaluation/web-sol-step160/freeze.json')
    write(directory/'freeze.json',dict(checkpoint=str(checkpoint),step=160,model='gemini-3.8-flash',node=NODE,allocation=ALLOCATION,
        gpus=8,workers_per_gpu=4,total_workers=32,smoke=smoke,suites=suites,source_hashes=paths,
        expected_runtime=base['expected_runtime'],controller_hashes=base['controller_hashes'],
        settings=dict(response_margin=0,visible_answer_tokens=96,thinking_budget=128,location='global',window_frames=32,
                      history='self-generated',memory='last-response/aged-out/v2',tv_subtitles='current-frame-only; measured FPS; frame zero'),
        authorization='User explicitly requested all four Flash suites on current node with eight free GPUs, 38177, and completion monitoring.',
        interpretation='Same step-160 learned controller and Sol causal observations/history. Flash answers feed subsequent controller history. Report WEB/EGO RL training and heldout cohorts separately. Smoke streams are truncated and never headline eligible.'))
    print(json.dumps(dict(prepared=str(directory),smoke=smoke,suites=suites)),flush=True)

def checked(directory,target,argv,log):
    verify(directory)
    with (target/log).open('a') as handle:
        process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        write(target/(log+'.process.json'),dict(pid=process.pid,argv=list(map(str,argv))))
        code=process.wait()
        if code: raise subprocess.CalledProcessError(code,argv)

def evaluate(directory,domain,frozen):
    from run_jevany_rl_full_current_node import checkpoint_view
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats, bootstrap
    target=directory/domain
    try:
        if (target/'result.json').exists(): return read(target/'result.json')
        rollout=target/'rollout';rollout.mkdir(exist_ok=True)
        specs={s['question_id']:s for s in rows(target/'evaluation-manifest.jsonl')}
        if not (rollout/'predictions.jsonl').exists():
            view=checkpoint_view(target,Path(frozen['checkpoint']))
            status(target,'inference',episodes=len(specs),gpus=8,workers=8,answer_model=frozen['model'])
            checked(directory,target,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_flash_streaming.py','--controller',view,
                '--validation-manifest',target/'evaluation-manifest.jsonl','--selection-ids',target/'selection.json',
                '--output-dir',rollout,'--response-margin','0','--answer-model',frozen['model'],
                '--subtitles','on' if domain=='tv' else 'off','--max-new-tokens','96','--record-action-logits'],'rollout.log')
            checked(directory,target,[sys.executable,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],'merge.log')
        predictions=rows(rollout/'predictions.jsonl')
        if [p['question_id'] for p in predictions]!=read(target/'selection.json'): raise ValueError('incomplete predictions')
        protocol=read(rollout/'manifest.json')
        if any(protocol[k]!=v for k,v in frozen['controller_hashes'].items()): raise ValueError('checkpoint differs')
        for pred in predictions:
            spec=specs[pred['question_id']];trace=pred['jevany_action_trace']
            if len(trace)!=len(spec['inference_source_indices']): raise ValueError('incomplete causal trace')
            if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace): raise ValueError('non-argmax decision')
            if len(pred['gemini_calls'])!=len(pred['model_response_list']): raise ValueError('missing API audit')
            if any(c['status']!=200 or c['model']!=frozen['model'] for c in pred['gemini_calls']): raise ValueError('wrong/failed answer model')
            if domain=='tv':
                if pred.get('subtitle_context_mode')!='current-frame-only' or pred.get('subtitles_delivered')!=spec['expected_subtitles_delivered']: raise ValueError('subtitle coverage mismatch')
                bins=bin_subtitles(spec['extra_turns'],spec['inference_timestamps'])
                for turn,lines in zip(trace,bins,strict=True):
                    if any(line not in turn['frame_time_text'] for line in lines): raise ValueError('subtitle missing from controller observation')
        if frozen['smoke']:
            result=dict(domain=domain,episodes=len(predictions),smoke=True,headline_eligible=False,
                        fires=sum(len(p['model_response_list']) for p in predictions),protocol=protocol)
            write(target/'result.json',result);status(target,'complete',smoke=True);return result
        status(target,'judging',episodes=len(predictions))
        gold=read(target/'gold.json');cohorts=read(target/'cohorts.json')
        metrics,usage=score_predictions(predictions,gold,target/'scoring',target/'judge-cache.sqlite')
        requests=rows(target/'scoring/requests.jsonl')
        outputs=[dict(custom_id=r['custom_id'],response=dict(status_code=200,body=dict(choices=[dict(message=dict(content='3'))])),error=None) for r in requests]
        timing,_=score_predictions(predictions,gold,target/'timing-only',target/'unused-timing-cache.sqlite',supplied_outputs=outputs)
        timing_result=dict(judge_model='timing',headline_eligible=False,cohorts={k:stats(timing,v) for k,v in cohorts.items()},note='Maximum answer score; timing/coverage diagnostic only.')
        write(target/'timing-only/result.json',timing_result)
        result=dict(step=160,answer_model=frozen['model'],domain=domain,cohorts={k:stats(metrics,v) for k,v in cohorts.items()},
                    timing_only=timing_result,per_episode=metrics,judge_usage=usage,protocol=protocol,interpretation=frozen['interpretation'])
        baseline=Path(frozen['suites'][domain]['sol_baseline'])
        if baseline.exists():
            result['paired_vs_step160_sol']={k:bootstrap(metrics,read(baseline)['per_episode'],v) for k,v in cohorts.items()}
        write(target/'result.json',result);status(target,'complete',cohorts=result['cohorts']);return result
    except BaseException as exc:
        status(target,'failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise

def run(directory):
    import torch
    from pace_clm.rl_anchor import verify_committed_checkpoint
    frozen=verify(directory)
    if socket.gethostname()!=NODE or subprocess.check_output(['squeue','-h','-j',str(ALLOCATION),'-o','%T %N'],text=True).strip()!='RUNNING '+NODE: raise RuntimeError('current authorized node/allocation only')
    lock=(ROOT/f'outputs/jevany-2b-rl/evaluation-gpu-{ALLOCATION}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(): raise RuntimeError('GPUs busy')
    if torch.cuda.device_count()!=8: raise RuntimeError('all eight GPUs required')
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual): raise RuntimeError('pinned runtime mismatch')
    write(directory/'runtime.json',actual);verify_committed_checkpoint(Path(frozen['checkpoint']),160)
    status(directory,'running',suites=list(DOMAINS),total_workers=32,workers_per_gpu=4)
    results,errors={},{}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate,directory,d,frozen):d for d in DOMAINS}
        for future in as_completed(futures):
            domain=futures[future]
            try: results[domain]=future.result()
            except BaseException as exc: errors[domain]=f'{type(exc).__name__}: {exc}'
            status(directory,'running',completed=list(results),errors=errors,total_workers=32,workers_per_gpu=4)
    if errors:
        status(directory,'partial_failure',completed=list(results),errors=errors);raise RuntimeError(str(errors))
    write(directory/'result.json',dict(step=160,answer_model=frozen['model'],smoke=frozen['smoke'],suites={k:{a:b for a,b in v.items() if a!='per_episode'} for k,v in results.items()}))
    status(directory,'complete',completed=list(results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='prepare': prepare(a.output.resolve(),a.smoke)
    else: run(a.output.resolve())
