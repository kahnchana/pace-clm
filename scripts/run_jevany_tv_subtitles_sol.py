#!/usr/bin/env python3
"""TV + Sol with native current-frame-only subtitles, alongside existing EGO."""
import argparse
import fcntl
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import json

from eval_jevany_tv_vad160 import ROOT, SG, PARENT, NODE, read, rows, sha, write
from run_jevany_sol_parallel import status
from stream_gemma.subtitles import bin_subtitles

DEFAULT = PARENT/'evaluation/tv-subtitles-sol-step160'
REFERENCE = SG/'outputs/evals/tv_subtitles_current/qwen-w1-long-neg/generate_summary.json'


def verify(directory):
    frozen = read(directory/'freeze.json')
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('frozen source changed: '+path)
    return frozen


def prepare(directory):
    from pace_clm.rl_anchor import verify_committed_checkpoint
    verify_committed_checkpoint(PARENT/'step-000160',160)
    if (directory/'freeze.json').exists():
        verify(directory);return
    directory.mkdir(parents=True,exist_ok=True)
    source = PARENT/'evaluation/tv-vad-step160/evaluation-manifest.jsonl'
    annotation = Path('/fsx/sfr/data/multimodal/online_video_language/ProactiveVideoQA/TV/anno.json')
    by_id = {r['question_id']:r for r in read(annotation)}
    specs = sorted([s for s in rows(source) if s['domain']=='tv'],key=lambda s:s['question_id'])
    reference = read(REFERENCE)
    assert reference['subtitles']=='on' and reference['subtitle_context_mode']=='current-frame-only'
    delivered = annotated = 0
    for s in specs:
        item = by_id[s['native_question_id']]
        question = next(t for t in item['conversation'] if t['role']=='user' and t.get('content'))
        s['extra_turns'] = [dict(time=float(t.get('time',0)),content=t['content']) for t in item['conversation']
                            if t is not question and t['role']=='user' and t.get('content')]
        # Reference --source-fps 0 uses measured FPS and starts at physical frame 0.
        indices = list(range(0,s['source_frame_count'],max(1,round(s['source_fps']))))
        s['inference_source_indices'] = indices
        s['inference_timestamps'] = [i/s['source_fps'] for i in indices]
        s['expected_subtitles_delivered'] = sum(map(len,bin_subtitles(s['extra_turns'],s['inference_timestamps'])))
        annotated += len(s['extra_turns']);delivered += s['expected_subtitles_delivered']
    if len(specs)!=500 or annotated!=reference['subtitles_annotated'] or delivered!=reference['subtitles_delivered']:
        raise ValueError(f'subtitle reference coverage mismatch: {len(specs)}, {annotated}, {delivered}')
    (directory/'evaluation-manifest.jsonl').write_text(''.join(json.dumps(s,sort_keys=True)+'\n' for s in specs))
    write(directory/'selection.json',[s['question_id'] for s in specs])
    write(directory/'cohorts.json',dict(tv_all=[s['question_id'] for s in specs]))
    write(directory/'gold.json',[dict(question_id=s['question_id'],conversation=[dict(role='user',content=s['question'])],
                                    answer=s['gold_answers']) for s in specs])
    paths = dict(read(PARENT/'evaluation/web-sol-step160/freeze.json')['source_hashes'])
    for p in [Path(__file__),ROOT/'scripts/run_jevany_tv_subtitles_sol.sh',ROOT/'scripts/eval_jevany_tv_subtitles_sol.py',
              ROOT/'scripts/run_jevany_sol_parallel.py',SG/'src/stream_gemma/subtitles.py',
              SG/'zs_infer/adapters/web.py',REFERENCE,source,annotation,
              *(directory/n for n in ('evaluation-manifest.jsonl','selection.json','cohorts.json','gold.json'))]:
        paths[str(p.resolve())] = sha(p)
    base = read(PARENT/'evaluation/web-sol-step160/freeze.json')
    frozen = dict(base,source_hashes=paths,suites=dict(tv=dict(episodes=500,
        baseline=str(PARENT/'evaluation/tv-vad-step160/result.json'))),
        subtitle_reference=str(REFERENCE),subtitles_annotated=annotated,subtitles_delivered=delivered,
        interpretation='Current-frame-only native subtitles match stream-gemma Qwen MoE reference. Visual-only baseline differs in subtitles and sampling origin; preserve both.',
        authorization='User explicitly requested re-running TV with subtitles now on this node, following stream-gemma MoE subtitle reference.')
    frozen['settings'] = dict(subtitles='on',subtitle_context_mode='current-frame-only',source_fps=0,
        sampling='measured FPS, start physical frame 0',response_margin=0,answer_tokens=96)
    write(directory/'freeze.json',frozen)
    print(json.dumps(dict(prepared=str(directory),questions=500,subtitles_delivered=delivered)),flush=True)


def run(directory):
    import torch
    from pace_clm.rl_anchor import verify_committed_checkpoint
    frozen = verify(directory)
    if socket.gethostname()!=NODE or subprocess.check_output(['squeue','-h','-j','38226','-o','%T %N'],text=True).strip()!='RUNNING '+NODE:
        raise RuntimeError('authorized node/allocation only')
    # Keep the active EGO orchestrator's global lock intact. This explicit
    # co-resident TV job owns an additional lock and admits only those EGO GPUs.
    lock = (ROOT/'outputs/jevany-2b-rl/tv-subtitles-sol-gpu-38226.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock = (ROOT/'outputs/jevany-2b-rl/evaluation-gpu-38226.lock').open('a')
    try:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    pids = set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split())
    for pid in pids:
        command = Path('/proc',pid,'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/eval_jevany_openai_streaming.py' not in command or '/ego-tv-vad-sol-step160/ego/' not in command:
            raise RuntimeError('unexpected GPU process: '+pid)
    free = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()
    if len(free)!=8 or min(map(int,free))<50000:
        raise RuntimeError('eight local GPUs with TV headroom required')
    actual = dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual):
        raise RuntimeError('pinned runtime mismatch')
    write(directory/'runtime.json',actual)
    verify_committed_checkpoint(Path(frozen['checkpoint']),160)
    evaluate(directory,'tv',frozen)


def evaluate(directory,domain,frozen):
    from run_jevany_rl_full_current_node import checkpoint_view
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats, bootstrap
    target = directory
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
                ROOT/'scripts/eval_jevany_tv_subtitles_sol.py','--controller',view,
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
        for pred in predictions:
            spec = specs[pred['question_id']]
            if pred.get('subtitle_context_mode')!='current-frame-only':
                raise ValueError('subtitle context mode mismatch')
            if pred.get('subtitles_delivered')!=spec['expected_subtitles_delivered']:
                raise ValueError('missing delivered subtitles')
            bins = bin_subtitles(spec['extra_turns'],spec['inference_timestamps'])
            for trace,lines in zip(pred['jevany_action_trace'],bins,strict=True):
                current_text = trace['frame_time_text']
                if any(line not in current_text for line in lines):
                    raise ValueError('subtitle missing from controller observation')
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
            paired_vs_visual_only_moe={k:bootstrap(metrics,baseline,v) for k,v in cohorts.items()},
            per_episode=metrics,judge_usage=usage,protocol=protocol,interpretation=frozen['interpretation'])
        write(target/'result.json',result)
        status(target,'complete',cohorts=result['cohorts'])
        return result
    except BaseException as exc:
        status(target,'failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.output.resolve())
