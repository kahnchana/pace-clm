#!/usr/bin/env python3
"""Freeze and run the explicitly requested step-160 + Sol full-WEB evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys

from eval_jevany_tv_vad160 import ROOT, SG, PARENT, NODE, read, rows, sha, write

DEFAULT = PARENT/'evaluation/web-sol-step160'


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
    parent_cohorts = read(PARENT/'data/cohorts.json')
    cohorts = {k:v for k,v in parent_cohorts.items() if k.startswith('web_') and v}
    ids = sorted(cohorts['web_all'])
    if len(ids)!=500 or len(set(ids))!=500:
        raise ValueError('full WEB must have exactly 500 questions')
    specs = {s['question_id']:s for s in rows(PARENT/'data/evaluation-manifest.jsonl')}
    selected = [specs[q] for q in ids]
    (directory/'evaluation-manifest.jsonl').write_text(''.join(json.dumps(s,sort_keys=True)+'\n' for s in selected))
    write(directory/'selection.json',ids)
    write(directory/'cohorts.json',cohorts)
    paths = [Path(__file__),ROOT/'scripts/run_jevany_web_sol.sh',ROOT/'scripts/jevany_openai_answer.py',
        ROOT/'scripts/eval_jevany_openai_streaming.py',ROOT/'scripts/jevany_mixed_reward.py',
        ROOT/'scripts/jevany_rl_reward.py',ROOT/'scripts/eval_jevany_mixed_queue.py',
        ROOT/'scripts/eval_jevany_tv_vad160.py',ROOT/'scripts/merge_jevany_benchmark.py',
        ROOT/'src/pace_clm/jevany_streaming.py',ROOT/'src/pace_clm/qwen_controller_runtime.py',
        ROOT/'src/pace_clm/evaluation_policy.py',SG/'src/stream_gemma/inference/closed_api.py',
        SG/'src/stream_gemma/inference/qwen_stream.py',SG/'src/stream_gemma/v24_memory.py',
        SG/'zs_infer/pauc_f.py',SG/'datasets/proactive-videoqa-web/anno.json',
        directory/'evaluation-manifest.jsonl',directory/'selection.json',directory/'cohorts.json']
    baseline = PARENT/'evaluation/full-step160/result.json'
    paths.append(baseline)
    baseline_protocol = read(baseline.parent/'rollout/manifest.json')
    write(directory/'freeze.json',dict(checkpoint=str(checkpoint),step=160,node=NODE,allocation=38226,gpus=8,
        model='gpt-5.6-sol',answer_tokens=96,reasoning_effort='none',response_margin=0,
        source_hashes={str(p.resolve()):sha(p) for p in paths},baseline=str(baseline),
        controller_hashes={k:baseline_protocol[k] for k in ('controller_adapter_sha256','pointer_head_sha256')},
        expected_runtime=read(PARENT/'runtime-contract.json'),
        authorization='User requested WEB with our step-160 controller and gpt-5.6-sol using the scoring gateway environment.',
        interpretation='Self-generated Sol answers feed controller history; firing times can change. Full WEB contains RL training examples; report heldout separately.'))
    print(json.dumps(dict(prepared=str(directory),questions=len(ids))),flush=True)


def run(directory):
    import torch
    from pace_clm.rl_anchor import verify_committed_checkpoint
    from run_jevany_rl_full_current_node import checkpoint_view
    from jevany_mixed_reward import score_predictions
    from eval_jevany_mixed_queue import stats, bootstrap
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
        checkpoint = Path(frozen['checkpoint'])
        verify_committed_checkpoint(checkpoint,160)
        rollout = directory/'rollout';rollout.mkdir(exist_ok=True)
        if not (rollout/'predictions.jsonl').exists():
            view = checkpoint_view(directory,checkpoint)
            status('inference',episodes=500,gpus=8,answer_model=frozen['model'])
            checked([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_openai_streaming.py','--controller',view,
                '--validation-manifest',directory/'evaluation-manifest.jsonl',
                '--selection-ids',directory/'selection.json','--output-dir',rollout,
                '--response-margin','0','--answer-model',frozen['model'],'--max-new-tokens','96',
                '--record-action-logits'],'rollout.log')
            checked([sys.executable,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],'merge.log')
        predictions = rows(rollout/'predictions.jsonl')
        if [p['question_id'] for p in predictions]!=read(directory/'selection.json'):
            raise ValueError('incomplete predictions')
        protocol = read(rollout/'manifest.json')
        if any(protocol[k]!=v for k,v in frozen['controller_hashes'].items()):
            raise ValueError('controller differs from committed step-160 WEB baseline')
        specs = {s['span_id']:s for s in rows(PARENT/'data/evaluation-specs.jsonl')}
        for pred in predictions:
            trace = pred['jevany_action_trace']
            if len(trace)!=specs[pred['question_id']]['rollout_frames']:
                raise ValueError('incomplete causal trace')
            if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace):
                raise ValueError('controller changed raw learned argmax')
            if len(pred['openai_calls'])!=len(pred['model_response_list']):
                raise ValueError('missing API call audit')
        status('judging',episodes=500)
        gold = [dict(question_id=p['question_id'],conversation=[dict(role='user',content=specs[p['question_id']]['question'])],
                     answer=specs[p['question_id']]['gold_answers']) for p in predictions]
        metrics,usage = score_predictions(predictions,gold,directory/'scoring',directory/'judge-cache.sqlite')
        cohorts = read(directory/'cohorts.json')
        baseline = read(frozen['baseline'])['per_episode']
        write(directory/'result.json',dict(step=160,answer_model=frozen['model'],cohorts={k:stats(metrics,v) for k,v in cohorts.items()},
            paired_vs_step160_moe={k:bootstrap(metrics,baseline,v) for k,v in cohorts.items()},
            per_episode=metrics,judge_usage=usage,protocol=protocol,interpretation=frozen['interpretation']))
        status('complete',cohorts=read(directory/'result.json')['cohorts'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.output.resolve())
