#!/usr/bin/env python3
"""Score-independent asynchronous evaluations; endpoint reserved for training node."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_jevany_estp128 import ROOT, SG, PARENT, read, rows, sha, write, dump
from jevany_mixed_reward import score_predictions
from eval_jevany_mixed_queue import stats, bootstrap
from run_jevany_rl_full_current_node import checkpoint_view
from pace_clm.rl_anchor import verify_committed_checkpoint

PY = SG / '.venv/bin/python'


def verify_sources(run):
    for path, expected in read(run / 'workflow-freeze.json')['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('immutable source changed: ' + path)


def checked(argv, log):
    with Path(log).open('a') as handle:
        subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def chosen_cohorts(cohorts, final):
    return {name: ids for name, ids in cohorts.items()
            if ids and (final or name not in ('estp_it_test', 'estp_bench'))}


def evaluate(run, job, final):
    verify_sources(run)
    directory = run / 'evaluation' / job['name']
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(job['checkpoint'])
    verify_committed_checkpoint(checkpoint, job['step'])
    if job['kind'] == 'parity':
        checked([PY, run / 'code/probe_estp_parity.py', '--run', run], directory / 'parity.log')
        write(directory / 'result.json', dict(job=job, parity=read(run/'cross-node-parity.json')))
        return
    cohorts = chosen_cohorts(read(run/'data/cohorts.json'), final)
    specs = {s['span_id']:s for s in rows(run/'data/evaluation-specs.jsonl')}
    ids = sorted({q for values in cohorts.values() for q in values})
    write(directory/'selection.json', ids)
    rollout = directory/'rollout'
    rollout.mkdir(exist_ok=True)
    if not (rollout/'predictions.jsonl').exists():
        view = checkpoint_view(directory, checkpoint)
        checked([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',ROOT/'scripts/eval_jevany_streaming.py',
            '--controller',view,'--validation-manifest',run/'data/evaluation-manifest.jsonl',
            '--selection-ids',directory/'selection.json','--output-dir',rollout,'--response-margin','0',
            '--answer-model','promoted','--max-new-tokens','96','--record-action-logits'],directory/'rollout.log')
        checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],directory/'merge.log')
    predictions = rows(rollout/'predictions.jsonl')
    if [p['question_id'] for p in predictions] != ids:
        raise ValueError('incomplete evaluation')
    for pred in predictions:
        trace = pred.get('jevany_action_trace', [])
        if len(trace) != specs[pred['question_id']]['rollout_frames']:
            raise ValueError('incomplete causal trace')
        if any(t['action'] != ('Silence','Hold','Response')[max(range(3), key=lambda i:t['logits'][i])] for t in trace):
            raise ValueError('evaluation changed raw learned argmax')
    # ESTP-Bench contains native zero-width/out-of-clip intervals. Preserve all
    # cases for its official scorer, rather than coercing them to PAUC windows.
    benchmark_ids = set(cohorts.get('estp_bench', []))
    pauc_predictions = [p for p in predictions if p['question_id'] not in benchmark_ids]
    gold = [dict(question_id=p['question_id'],conversation=[dict(role='user',content=specs[p['question_id']]['question'])],
                 answer=specs[p['question_id']]['gold_answers']) for p in pauc_predictions]
    metrics, usage = score_predictions(pauc_predictions, gold, directory/'scoring', run/'cache/evaluation.sqlite')
    result = dict(job=job, execution_node=socket.gethostname(), allocation=38177 if final else 38226,
        cohorts={k:stats(metrics,v) for k,v in cohorts.items() if k != 'estp_bench'}, per_episode=metrics, judge_usage=usage,
        quality_policy='diagnostic only; complete all 128 updates irrespective of scores',
        seen_data_label='Existing WEB/EGO/original training identities seen; IT dev/test excluded from this phase; historical benchmark reuse retained')
    baseline_path = run/('evaluation/final-reference-step32/result.json' if final else 'evaluation/baseline-step32/result.json')
    if baseline_path.exists() and job['name'] != baseline_path.parent.name:
        baseline = read(baseline_path)['per_episode']
        result['paired_vs_step32'] = {k:bootstrap(metrics,baseline,v) for k,v in cohorts.items()
                                     if all(q in baseline for q in v)}
    parent_baseline = read(PARENT/'evaluation/baseline-sft/result.json')['per_episode']
    result['paired_vs_sft'] = {k:bootstrap(metrics,parent_baseline,v) for k,v in cohorts.items()
                              if all(q in parent_baseline for q in v)}
    if final:
        # Official ESTP-F1 uses absolute native timestamps and its own judge rubric.
        from judge_exp10_utility import load_gateway_key
        from zs_infer.scoring import score_estp
        load_gateway_key()
        for label, annotation in [('estp_bench', SG/'datasets/estp/estp_bench_sq.json'),
                                  ('estp_it_test', Path('/fsx/sfr/data/multimodal/online_video_language/ESTP-IT/ESTP_IT/estp.json'))]:
            output = directory/label
            output.mkdir(exist_ok=True)
            selected = []
            for pred in predictions:
                if pred['question_id'] not in cohorts[label]:
                    continue
                spec = specs[pred['question_id']]
                selected.append(dict(question_id=pred['question_id'],video_id=spec['video_id'],
                    clip_id=spec['clip_id'],qa_index=spec['qa_index'],task_type=spec['task_type'],
                    estp_turns=[dict(role='assistant',time=r['time']+spec['clip_start_time'],content=r['content'])
                                for r in pred['model_response_list']]))
            dump(output/'predictions.jsonl',selected)
            result[label+'_official_estp_f1'] = score_estp(output,output/'predictions.jsonl',annotation,
                SG/'third_party/estp_f1/estp_f1.py',judge=True,judge_model='gpt-4.1',
                task_timing_assumption='eyewo',tau_ant=1.,tau_lat=2.,default_answer_score=5.)
        if baseline_path.exists() and job['name'] != baseline_path.parent.name:
            base=read(baseline_path)
            result['official_estp_f1_change_vs_step32']={label:
                result[label+'_official_estp_f1']['estp_f1_x100']-base[label+'_official_estp_f1']['estp_f1_x100']
                for label in ('estp_bench','estp_it_test')}
    write(directory/'result.json', result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--final-local',action='store_true');args=parser.parse_args();run=args.run.resolve()
    node='ip-10-3-120-167' if args.final_local else 'ip-10-3-70-177'
    allocation=38177 if args.final_local else 38226
    if socket.gethostname()!=node:
        raise RuntimeError('authorized evaluation node only')
    if subprocess.check_output(['squeue','-h','-j',str(allocation),'-o','%T %N'],text=True).strip()!='RUNNING '+node:
        raise RuntimeError('authorized allocation unavailable')
    lock=(ROOT/'outputs/jevany-2b-rl'/('gpu.lock' if args.final_local else 'evaluation-gpu-38226.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('evaluation GPUs busy')
    if len(subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines())!=8:
        raise RuntimeError('eight GPUs required')
    import torch
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    expected=read(run/'runtime-contract.json')
    if any(actual[k]!=expected[k] for k in actual):raise RuntimeError('pinned evaluation runtime mismatch')
    write(run/('pinned-final-runtime.json' if args.final_local else 'pinned-evaluation-runtime.json'),actual)
    def status(state,**fields):
        write(run/('final-evaluation-status.json' if args.final_local else 'evaluation-status.json'),
              dict(state=state,utc=datetime.now(timezone.utc).isoformat(),node=node,**fields))
    status('waiting')
    while True:
        pending=[read(p) for p in (run/'queue').glob('*.json') if not (run/'evaluation'/p.stem/'result.json').exists()]
        pending=[j for j in pending if j['kind'].startswith('final')==args.final_local]
        if not pending:
            if args.final_local:
                raise RuntimeError('final checkpoint job missing')
            state=read(run/'workflow-status.json')['state'] if (run/'workflow-status.json').exists() else None
            if state=='complete':
                status('complete');return
            time.sleep(10);continue
        job=min(pending,key=lambda j:(j['priority'],j['step'],j['name']))
        if args.final_local and read(run/'latest.json')['step']!=128:
            raise RuntimeError('training must complete all updates before final evaluation')
        status('evaluating',job=job['name'])
        try:
            evaluate(run,job,args.final_local)
        except Exception as exc:
            status('retrying',job=job['name'],error=f'{type(exc).__name__}: {exc}')
            with (run/'evaluation-retries.log').open('a') as handle:
                traceback.print_exc(file=handle)
            # Preserve completed shards/judgments; retry the identical frozen job.
            time.sleep(30);continue
        if args.final_local:
            status('complete',job=job['name']);return
        status('waiting',last_completed=job['name'])


if __name__=='__main__':
    main()
