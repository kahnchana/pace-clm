#!/usr/bin/env python3
"""Explicitly authorized, resumable TV/VAD-only evaluation of RL step 160."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent/'stream-gemma'
PARENT = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160'
DEFAULT = PARENT/'evaluation/tv-vad-step160'
MEDIA = Path('/fsx/sfr/data/multimodal/online_video_language/ProactiveVideoQA')
NODE = 'ip-10-3-70-177'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(SG/'src'), str(SG)]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify(directory):
    frozen = read(directory/'freeze.json')
    for path, digest in frozen['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('frozen source changed: '+path)
    return frozen


def prepare(directory):
    from stream_gemma.inference.runtime import video_metadata
    from pace_clm.rl_anchor import verify_committed_checkpoint
    checkpoint = PARENT/'step-000160'
    verify_committed_checkpoint(checkpoint,160)
    if (directory/'freeze.json').exists():
        verify(directory)
        return
    directory.mkdir(parents=True,exist_ok=True)
    records, annotation_paths = [], []
    for domain, expected in [('tv',500),('vad',101)]:
        annotation = MEDIA/domain.upper()/'anno.json'
        source = read(annotation)
        if len(source)!=expected:
            raise ValueError('incomplete annotations: '+domain)
        annotation_paths.append(annotation)
        for item in source:
            path = MEDIA/domain.upper()/'videos'/item['video']
            if not path.is_file():
                raise FileNotFoundError(path)
            question = next(t['content'] for t in item['conversation'] if t['role']=='user' and t.get('content'))
            records.append((domain,item,path,question))

    def native(record):
        domain,item,path,question = record
        count,fps = video_metadata(path)
        stride = max(1,round(fps))
        # Preserve the WEB/EGO end-aligned native-frame sampling convention.
        indices = list(range((count-1)%stride,count,stride))
        return dict(question_id=domain+':'+item['question_id'],native_question_id=item['question_id'],
                    domain=domain,video_id=item['video'],question=question,
                    resolved_video_path=str(path.resolve()),inference_source_indices=indices,
                    inference_timestamps=[i/fps for i in indices],source_fps=fps,
                    source_frame_count=count,media_size=path.stat().st_size,
                    media_mtime_ns=path.stat().st_mtime_ns,gold_answers=item['answer'])

    with ThreadPoolExecutor(max_workers=8) as pool:
        specs = sorted(pool.map(native,records),key=lambda s:s['question_id'])
    ids = [s['question_id'] for s in specs]
    if len(ids)!=len(set(ids)) or any(not s['inference_source_indices'] for s in specs):
        raise ValueError('duplicate IDs or empty media')
    for s in specs:
        for answer in s['gold_answers']:
            if answer['reply_timespan'][1]<=answer['reply_timespan'][0]:
                raise ValueError('nonpositive native scoring window')
    (directory/'evaluation-manifest.jsonl').write_text(''.join(json.dumps(s,sort_keys=True)+'\n' for s in specs))
    write(directory/'selection.json',ids)
    write(directory/'cohorts.json',{d:[s['question_id'] for s in specs if s['domain']==d] for d in ('tv','vad')})
    paths = annotation_paths+[Path(__file__),ROOT/'scripts/eval_jevany_streaming.py',
        ROOT/'scripts/merge_jevany_benchmark.py',ROOT/'scripts/jevany_mixed_reward.py',
        ROOT/'scripts/jevany_rl_reward.py',ROOT/'scripts/eval_jevany_mixed_queue.py',
        ROOT/'src/pace_clm/jevany_streaming.py',ROOT/'src/pace_clm/qwen_controller_runtime.py',
        ROOT/'src/pace_clm/evaluation_policy.py',SG/'zs_infer/adapters/web.py',SG/'zs_infer/pauc_f.py',
        SG/'src/stream_gemma/inference/qwen_stream.py',SG/'src/stream_gemma/v24_memory.py',
        directory/'evaluation-manifest.jsonl',directory/'selection.json',directory/'cohorts.json']
    protocol = read(PARENT/'evaluation/full-step160/rollout/manifest.json')
    write(directory/'freeze.json',dict(checkpoint=str(checkpoint),step=160,node=NODE,allocation=38226,gpus=8,
        source_hashes={str(p.resolve()):sha(p) for p in paths},
        model_hashes={k:protocol[k] for k in ('controller_adapter_sha256','pointer_head_sha256','original_generator_adapter_sha256')},
        expected_runtime=read(PARENT/'runtime-contract.json'),
        protocol=dict(response_margin=0,decision='raw learned argmax',answer_model='promoted',max_new_tokens=96,
                      sampling='end-aligned native frames at approximately 1 FPS',frame_cap=None,window_frames=32,
                      subtitles='off; visual-only',memory='last-response/aged-out/v2',judge='gpt-4.1',omega=.5,
                      annotation_windows='native and unchanged'),
        counts={d:sum(s['domain']==d for s in specs) for d in ('tv','vad')},
        decisions=sum(len(s['inference_source_indices']) for s in specs),
        authorization='User requested TV/VAD for RL step 160 using eight GPUs on this node only; no other evaluations.',
        exposure='TV/VAD protected from training; historical benchmark reuse is not an untouched-test claim.'))
    print(json.dumps(dict(prepared=str(directory),episodes=len(ids),decisions=sum(len(s['inference_source_indices']) for s in specs))),flush=True)


def run(directory):
    from pace_clm.rl_anchor import verify_committed_checkpoint
    from run_jevany_rl_full_current_node import checkpoint_view
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats
    import torch
    if socket.gethostname()!=NODE:
        raise RuntimeError('authorized node only')
    allocation = subprocess.check_output(['squeue','-h','-j','38226','-o','%T %N'],text=True).strip()
    if allocation!='RUNNING '+NODE:
        raise RuntimeError('assigned allocation unavailable')
    lock = (ROOT/'outputs/jevany-2b-rl/evaluation-gpu-38226.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen = verify(directory)
    if len(subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines())!=8:
        raise RuntimeError('exactly eight GPUs required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPUs busy')
    actual = dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual):
        raise RuntimeError('pinned runtime mismatch: '+str(actual))
    write(directory/'runtime.json',actual)
    checkpoint = Path(frozen['checkpoint'])
    verify_committed_checkpoint(checkpoint,160)
    specs = {s['question_id']:s for s in rows(directory/'evaluation-manifest.jsonl')}
    for s in specs.values():
        info = Path(s['resolved_video_path']).stat()
        if info.st_size!=s['media_size'] or info.st_mtime_ns!=s['media_mtime_ns']:
            raise ValueError('media changed: '+s['resolved_video_path'])
    def status(state,**fields):
        write(directory/'status.json',dict(state=state,node=NODE,allocation=38226,
              utc=datetime.now(timezone.utc).isoformat(),**fields))
    def checked(argv,log):
        verify(directory)
        with (directory/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    try:
        if (directory/'result.json').exists():
            status('complete');return
        rollout = directory/'rollout';rollout.mkdir(exist_ok=True)
        if not (rollout/'predictions.jsonl').exists():
            view = checkpoint_view(directory,checkpoint)
            status('inference',episodes=len(specs),gpus=8)
            checked([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_streaming.py','--controller',view,
                '--validation-manifest',directory/'evaluation-manifest.jsonl',
                '--selection-ids',directory/'selection.json','--output-dir',rollout,
                '--response-margin','0','--answer-model','promoted','--max-new-tokens','96',
                '--record-action-logits'],'rollout.log')
            checked([sys.executable,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],'merge.log')
        protocol = read(rollout/'manifest.json')
        if any(protocol[k]!=v for k,v in frozen['model_hashes'].items()):
            raise ValueError('model differs from WEB/EGO step 160')
        predictions = rows(rollout/'predictions.jsonl')
        if [p['question_id'] for p in predictions]!=read(directory/'selection.json'):
            raise ValueError('incomplete predictions')
        for pred in predictions:
            trace = pred['jevany_action_trace']
            if len(trace)!=len(specs[pred['question_id']]['inference_source_indices']):
                raise ValueError('incomplete causal trace')
            for t in trace:
                if t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])]:
                    raise ValueError('action differs from raw learned argmax')
        status('judging',episodes=len(predictions))
        gold = [dict(question_id=q,conversation=[dict(role='user',content=s['question'])],answer=s['gold_answers']) for q,s in specs.items()]
        metrics,usage = score_predictions(predictions,gold,directory/'scoring',directory/'judge-cache.sqlite')
        result = dict(step=160,checkpoint=str(checkpoint),node=NODE,allocation=38226,
                      cohorts={k:stats(metrics,v) for k,v in read(directory/'cohorts.json').items()},
                      per_episode=metrics,judge_usage=usage,protocol=frozen['protocol'],exposure=frozen['exposure'])
        write(directory/'result.json',result)
        status('complete',cohorts=result['cohorts'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.output.resolve())
