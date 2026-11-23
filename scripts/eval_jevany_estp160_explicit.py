#!/usr/bin/env python3
"""User-authorized step-160 explicit-only inference and unchanged ESTP PAUC-F."""
import fcntl
import os
import socket
import subprocess
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

from prepare_jevany_estp128 import ROOT, SG, read, rows, sha, write, dump
from run_jevany_rl_full_current_node import checkpoint_view
from pace_clm.rl_anchor import verify_committed_checkpoint
from zs_infer import estp_pauc_f as estp

RUN = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160'
SOURCE = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128'
OUT = RUN/'evaluation/estp-explicit-step160'
PY = SG/'.venv/bin/python'


def status(state, **fields):
    write(OUT/'status.json', dict(state=state, utc=datetime.now(timezone.utc).isoformat(),
          node=socket.gethostname(), allocation=38177, step=160, **fields))


def checked(argv, log):
    with (OUT/log).open('a') as handle:
        subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'workflow.lock').open('a') as workflow_lock:
        fcntl.flock(workflow_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if socket.gethostname() != 'ip-10-3-120-167':
            raise RuntimeError('authorized current node only')
        allocation = subprocess.check_output(
            ['squeue','-h','-j','38177','-o','%T %N'], text=True).strip()
        if allocation != 'RUNNING ip-10-3-120-167':
            raise RuntimeError('authorized allocation unavailable')
        checkpoint = RUN/'step-000160'
        verify_committed_checkpoint(checkpoint, 160)
        import torch
        actual = dict(torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version())
        if any(actual[k] != read(RUN/'runtime-contract.json')[k] for k in actual):
            raise RuntimeError('pinned runtime mismatch')
        cache_keys = ('TMPDIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                      'CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR')
        if any(not os.environ.get(k, '').startswith('/opt/dlami/nvme/') for k in cache_keys):
            raise RuntimeError('temporary caches must use local NVMe')
        manifest = estp.build_manifest(estp.DEFAULT_ANNOTATION, estp.DEFAULT_VIDEO_ROOT, horizon='clip')
        records = manifest['records']
        specs = {f"{s['video_id']}/{s['clip_id']}/{s['qa_index']}":s
                 for s in rows(SOURCE/'data/evaluation-specs.jsonl')
                 if s.get('data_source') == 'estp_bench'}
        selected = {r['question_id']:specs[r['question_id']] for r in records}
        ids = sorted(s['span_id'] for s in selected.values())
        if len(ids) != 851 or len(set(ids)) != 851:
            raise ValueError('explicit selection must contain exactly 851 unique cases')
        freeze = dict(checkpoint=str(checkpoint), checkpoint_commit_sha256=sha(checkpoint/'commit.json'),
            selection=ids, metric=estp.SCORER_VERSION, horizon='clip', match_window='strict',
            headline_omega=.5, response_margin=0, answer_model='promoted', max_new_tokens=96,
            source_manifest_sha256=sha(SOURCE/'data/evaluation-manifest.jsonl'),
            source_specs_sha256=sha(SOURCE/'data/evaluation-specs.jsonl'),
            annotation_sha256=manifest['summary']['annotation_sha256'],
            scorer_sha256=sha(estp.__file__), official_scorer_sha256=sha(estp.OFFICIAL_SCORER),
            evaluator_sha256=sha(ROOT/'scripts/eval_jevany_streaming.py'), runtime=actual,
            authorization='User requested step-160 explicit proactive PAUC-F on current eight-GPU node')
        if (OUT/'freeze.json').exists() and read(OUT/'freeze.json') != freeze:
            raise ValueError('frozen evaluation inputs changed')
        write(OUT/'freeze.json', freeze)
        write(OUT/'selection.json', ids)
        scoring = OUT/'estp_pauc_explicit'
        estp.write_manifest(manifest, scoring)
        rollout = OUT/'rollout'
        rollout.mkdir(exist_ok=True)
        if not (rollout/'predictions.jsonl').exists():
            with (ROOT/'outputs/jevany-2b-rl/gpu.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                             '--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('evaluation GPUs busy')
                devices = subprocess.check_output(['nvidia-smi','--query-gpu=name',
                                                    '--format=csv,noheader'], text=True).splitlines()
                if len(devices) != 8:
                    raise RuntimeError('eight GPUs required')
                view = checkpoint_view(OUT, checkpoint)
                status('inference', cases=len(ids), gpus=8)
                checked([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    ROOT/'scripts/eval_jevany_streaming.py','--controller',view,
                    '--validation-manifest',SOURCE/'data/evaluation-manifest.jsonl',
                    '--selection-ids',OUT/'selection.json','--output-dir',rollout,
                    '--response-margin','0','--answer-model','promoted','--max-new-tokens','96',
                    '--record-action-logits'], 'rollout.log')
                checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout], 'merge.log')
        predictions = rows(rollout/'predictions.jsonl')
        if [p['question_id'] for p in predictions] != ids:
            raise ValueError('incomplete explicit evaluation')
        by_span = {s['span_id']:s for s in selected.values()}
        by_qid = {}
        for pred in predictions:
            spec = by_span[pred['question_id']]
            trace = pred.get('jevany_action_trace', [])
            if len(trace) != spec['rollout_frames']:
                raise ValueError('incomplete causal trace')
            if any(t['action'] != ('Silence','Hold','Response')[max(range(3), key=lambda i:t['logits'][i])]
                   for t in trace):
                raise ValueError('evaluation changed raw learned argmax')
            by_qid[f"{spec['video_id']}/{spec['clip_id']}/{spec['qa_index']}"] = (pred, spec)
        converted = []
        for record in records:
            pred, spec = by_qid[record['question_id']]
            fires = [dict(role='assistant', content=t['content'],
                          absolute_time=float(t['time']) + spec['clip_start_time'],
                          time=float(t['time']) + spec['clip_start_time'] - record['eval_start_time'],
                          fire_id=i) for i,t in enumerate(pred['model_response_list'])]
            if any(not f['content'].strip() for f in fires):
                raise ValueError('empty generated answer')
            converted.append(dict(question_id=record['question_id'], task_type=record['task_type'],
                eval_start_time=record['eval_start_time'], eval_end_time=record['eval_end_time'],
                model_response_list=fires))
        dump(scoring/'predictions.jsonl', converted)
        write(scoring/'source-freeze.json',dict(freeze_sha256=sha(OUT/'freeze.json'),
              rollout_sha256=sha(rollout/'predictions.jsonl'), new_inference=True,
              conversion='relative rollout timestamps + native clip origin; same as step32/128 adapter'))
        status('gpt_scoring', cases=len(records), physical_fires=sum(len(p['model_response_list']) for p in converted))
        from judge_exp10_utility import load_gateway_key
        load_gateway_key()
        estp.cmd_judge(SimpleNamespace(manifest=scoring/'explicit_manifest.jsonl',
            predictions=scoring/'predictions.jsonl', official_scorer=estp.OFFICIAL_SCORER,
            pred_dir=scoring, judge_model='gpt-4.1', preflight_only=False,
            concurrency=16, max_attempts=5, request_timeout=120.))
        summary = read(scoring/'pauc_f_summary-gpt-4.1.json')
        if not summary['headline_eligible'] or len(summary['per_task']) != 8:
            raise ValueError('incomplete judged explicit scores')
        write(OUT/'result.json',dict(step=160, checkpoint=str(checkpoint), summary=summary,
              raw_argmax_verified=True, cases=len(records)))
        status('complete', explicit_all_pauc_f=summary['all']['omega_0.5'],
               per_task=summary['headline_task_pauc_f'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
