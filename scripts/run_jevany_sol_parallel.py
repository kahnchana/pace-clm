#!/usr/bin/env python3
"""Run EGO, TV and VAD with one Sol-backed controller per suite per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from eval_jevany_tv_vad160 import ROOT, SG, PARENT, NODE, read, rows, sha, write

DEFAULT = PARENT/'evaluation/ego-tv-vad-sol-step160'
DOMAINS = ('ego','tv','vad')


def status(directory, state, **fields):
    write(directory/'status.json',dict(state=state,node=NODE,allocation=38226,
          utc=datetime.now(timezone.utc).isoformat(),**fields))


def verify(directory):
    frozen = read(directory/'freeze.json')
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('frozen source changed: '+path)
    return frozen


def prepare(directory):
    from pace_clm.rl_anchor import verify_committed_checkpoint
    checkpoint = PARENT/'step-000160'
    verify_committed_checkpoint(checkpoint,160)
    if (directory/'freeze.json').exists():
        verify(directory);return
    directory.mkdir(parents=True,exist_ok=True)
    original = PARENT/'evaluation/full-step160'
    tv_vad = PARENT/'evaluation/tv-vad-step160'
    parent_cohorts = read(PARENT/'data/cohorts.json')
    parent_specs = {s['span_id']:s for s in rows(PARENT/'data/evaluation-specs.jsonl')}
    manifest = {s['question_id']:s for s in rows(PARENT/'data/evaluation-manifest.jsonl')}
    tv_vad_specs = rows(tv_vad/'evaluation-manifest.jsonl')
    # Retain exactly the code and generator settings validated by full WEB.
    paths = dict(read(PARENT/'evaluation/web-sol-step160/freeze.json')['source_hashes'])
    for p in (Path(__file__), ROOT/'scripts/run_jevany_sol_parallel.sh',
              PARENT/'data/cohorts.json', PARENT/'data/evaluation-specs.jsonl',
              PARENT/'data/evaluation-manifest.jsonl', tv_vad/'evaluation-manifest.jsonl',
              original/'result.json', tv_vad/'result.json'):
        paths[str(p.resolve())] = sha(p)
    suites = {}
    for domain,expected in [('ego',326),('tv',500),('vad',101)]:
        target = directory/domain
        target.mkdir(exist_ok=True)
        if domain=='ego':
            cohorts = {k:v for k,v in parent_cohorts.items() if k.startswith('ego_') and v}
            ids = sorted(cohorts['ego_all'])
            selected = [manifest[q] for q in ids]
            gold = [dict(question_id=q,conversation=[dict(role='user',content=parent_specs[q]['question'])],
                         answer=parent_specs[q]['gold_answers']) for q in ids]
            baseline = original/'result.json'
        else:
            selected = sorted([s for s in tv_vad_specs if s['domain']==domain],key=lambda s:s['question_id'])
            ids = [s['question_id'] for s in selected]
            cohorts = {domain+'_all':ids}
            gold = [dict(question_id=s['question_id'],conversation=[dict(role='user',content=s['question'])],
                         answer=s['gold_answers']) for s in selected]
            baseline = tv_vad/'result.json'
        if len(ids)!=expected or len(set(ids))!=expected:
            raise ValueError('incomplete or duplicate suite: '+domain)
        if any(not Path(s['resolved_video_path']).is_file() or not s['inference_source_indices'] for s in selected):
            raise ValueError('missing or empty media: '+domain)
        (target/'evaluation-manifest.jsonl').write_text(''.join(json.dumps(s,sort_keys=True)+'\n' for s in selected))
        write(target/'selection.json',ids)
        write(target/'cohorts.json',cohorts)
        write(target/'gold.json',gold)
        for name in ('evaluation-manifest.jsonl','selection.json','cohorts.json','gold.json'):
            paths[str((target/name).resolve())] = sha(target/name)
        suites[domain] = dict(episodes=expected,baseline=str(baseline),
            decisions=sum(len(s['inference_source_indices']) for s in selected))
    protocol = read(original/'rollout/manifest.json')
    write(directory/'freeze.json',dict(checkpoint=str(checkpoint),step=160,model='gpt-5.6-sol',
        node=NODE,allocation=38226,gpus=8,workers_per_gpu=3,total_workers=24,
        suites=suites,source_hashes=paths,expected_runtime=read(PARENT/'runtime-contract.json'),
        controller_hashes={k:protocol[k] for k in ('controller_adapter_sha256','pointer_head_sha256')},
        settings=dict(response_margin=0,answer_tokens=96,reasoning_effort='none',window_frames=32,
                      subtitles='off',history='self-generated',memory='last-response/aged-out/v2'),
        authorization='User requested EGO, TV, VAD with step-160 controller + Sol, all three in parallel, three processes per GPU on this node.',
        interpretation='Same Sol answer generator and causal input protocol as WEB. EGO full includes RL training data; report heldout separately. Timing-only assumes maximum answer quality and is not headline eligible.'))
    print(json.dumps(dict(prepared=str(directory),suites=suites,workers_per_gpu=3)),flush=True)


def evaluate(directory,domain,frozen):
    from run_jevany_rl_full_current_node import checkpoint_view
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats, bootstrap
    target = directory/domain
    try:
        if (target/'result.json').exists():
            status(target,'complete');return read(target/'result.json')
        rollout = target/'rollout';rollout.mkdir(exist_ok=True)
        specs = {s['question_id']:s for s in rows(target/'evaluation-manifest.jsonl')}
        def checked(argv,log):
            verify(directory)
            with (target/log).open('a') as handle:
                process = subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                write(target/(log+'.process.json'),dict(pid=process.pid,argv=list(map(str,argv))))
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code,argv)
        if not (rollout/'predictions.jsonl').exists():
            view = checkpoint_view(target,Path(frozen['checkpoint']))
            status(target,'inference',episodes=len(specs),gpus=8,workers=8,answer_model=frozen['model'])
            checked([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_openai_streaming.py','--controller',view,
                '--validation-manifest',target/'evaluation-manifest.jsonl','--selection-ids',target/'selection.json',
                '--output-dir',rollout,'--response-margin','0','--answer-model',frozen['model'],
                '--max-new-tokens','96','--record-action-logits'],'rollout.log')
            checked([sys.executable,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],'merge.log')
        predictions = rows(rollout/'predictions.jsonl')
        if [p['question_id'] for p in predictions]!=read(target/'selection.json'):
            raise ValueError('incomplete predictions: '+domain)
        protocol = read(rollout/'manifest.json')
        if any(protocol[k]!=v for k,v in frozen['controller_hashes'].items()):
            raise ValueError('controller checkpoint differs')
        for pred in predictions:
            trace = pred['jevany_action_trace']
            if len(trace)!=len(specs[pred['question_id']]['inference_source_indices']):
                raise ValueError('incomplete causal trace')
            if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):
                raise ValueError('non-argmax decision')
            if len(pred['openai_calls'])!=len(pred['model_response_list']):
                raise ValueError('missing API call audit')
        status(target,'judging',episodes=len(predictions))
        gold = read(target/'gold.json')
        metrics,usage = score_predictions(predictions,gold,target/'scoring',target/'judge-cache.sqlite')
        cohorts = read(target/'cohorts.json')
        baseline = read(frozen['suites'][domain]['baseline'])['per_episode']
        requests = rows(target/'scoring/requests.jsonl')
        outputs = [dict(custom_id=r['custom_id'],response=dict(status_code=200,
                   body=dict(choices=[dict(message=dict(content='3'))])),error=None) for r in requests]
        timing,_ = score_predictions(predictions,gold,target/'timing-only',target/'unused-timing-cache.sqlite',supplied_outputs=outputs)
        timing_result = dict(judge_model='timing',headline_eligible=False,
            cohorts={k:stats(timing,v) for k,v in cohorts.items()},
            note='Maximum answer score, no API calls; timing/coverage diagnostic only.')
        write(target/'timing-only/result.json',timing_result)
        result = dict(step=160,answer_model=frozen['model'],domain=domain,
            cohorts={k:stats(metrics,v) for k,v in cohorts.items()},timing_only=timing_result,
            paired_vs_step160_moe={k:bootstrap(metrics,baseline,v) for k,v in cohorts.items()},
            per_episode=metrics,judge_usage=usage,protocol=protocol,interpretation=frozen['interpretation'])
        write(target/'result.json',result)
        status(target,'complete',cohorts=result['cohorts'])
        return result
    except BaseException as exc:
        status(target,'failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise


def run(directory):
    import torch
    from pace_clm.rl_anchor import verify_committed_checkpoint
    frozen = verify(directory)
    if socket.gethostname()!=NODE or subprocess.check_output(['squeue','-h','-j','38226','-o','%T %N'],text=True).strip()!='RUNNING '+NODE:
        raise RuntimeError('assigned evaluation node/allocation only')
    lock = (ROOT/'outputs/jevany-2b-rl/evaluation-gpu-38226.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPUs busy')
    if torch.cuda.device_count()!=8:
        raise RuntimeError('all eight local GPUs required')
    actual = dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual):
        raise RuntimeError('pinned controller runtime mismatch')
    write(directory/'runtime.json',actual)
    verify_committed_checkpoint(Path(frozen['checkpoint']),160)
    status(directory,'running',suites=list(DOMAINS),total_workers=24,workers_per_gpu=3)
    results,errors = {},{}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(evaluate,directory,d,frozen):d for d in DOMAINS}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
                results[domain] = dict(cohorts=result['cohorts'],timing_only=result['timing_only'],
                                       paired_vs_step160_moe=result['paired_vs_step160_moe'])
            except BaseException as exc:
                errors[domain] = f'{type(exc).__name__}: {exc}'
            status(directory,'running',completed=list(results),errors=errors,total_workers=24,workers_per_gpu=3)
    if errors:
        status(directory,'partial_failure',completed=list(results),errors=errors)
        raise RuntimeError(str(errors))
    write(directory/'result.json',dict(step=160,answer_model=frozen['model'],suites=results))
    status(directory,'complete',completed=list(results))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.output.resolve())
