#!/usr/bin/env python3
"""Final raw-argmax checkpoint on the 20 perfect episodes with most responses."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from jevany_rl_reward import ROOT, SG, OFFICIAL, MODEL, judge_requests, sha, write
from run_jevany_e2_full_web import validate_judgments
from run_jevany_rl_full_current_node import checkpoint_view
from pace_clm.rl_anchor import verify_committed_checkpoint
from zs_infer.pauc_f import score_pauc_f

RUN=ROOT/'outputs/jevany-2b-rl/episode-rloo-v2'
PY=SG/'.venv/bin/python'
NODE='ip-10-3-120-167'


def read(path):return json.loads(Path(path).read_text())


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl(path,values):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(value,sort_keys=True)+'\n' for value in values))
    temporary.replace(path)


def prepare(output):
    data=read(RUN/'perfect-episodes.json')
    selected=sorted(data['episodes'],key=lambda row:(-row['responses'],row['step'],row['span_id']))[:20]
    if len(selected)!=20 or len({r['span_id'] for r in selected})!=20:raise ValueError('need 20 unique episodes')
    specs={r['span_id']:r for r in rows(RUN/'data/episodes.jsonl')}
    with (RUN/'reward-variance-groups.csv').open() as handle:
        sampled={r['span_id']:r for r in csv.DictReader(handle)}
    records=[];gold=[];selection=[]
    for item in selected:
        spec=specs[item['span_id']];stop=spec['rollout_stop']
        if spec['rollout_start']!=0 or spec['decode_start']!=0 or spec['prefix_turns'] or spec['initial_memory']:
            raise ValueError('expected empty continuous causal initialization at zero')
        if any(g['time_seconds']+4>stop for g in spec['gold_points']):raise ValueError('incomplete gold window')
        if item['responses']!=len(spec['gold_points']):raise ValueError('perfect sample response/gold mismatch')
        qid=spec['span_id'];sample=sampled[qid]
        original=[float(sample[f'reward_{i}']) for i in range(1,5)]
        if max(original)!=100.:raise ValueError('selected episode lacks a perfect original sample')
        selection.append({**item,'source_policy_version':item['step']-1,
                          'original_four_pauc_f':original,'video_path':spec['video_path']})
        records.append({'question_id':qid,'video_id':spec['video_id'],'question':spec['question'],
                        'resolved_video_path':spec['video_path'],
                        'inference_source_indices':list(range(stop)),
                        'inference_timestamps':[float(i) for i in range(stop)]})
        gold.append({'question_id':qid,'conversation':[{'role':'user','content':spec['question']}],
                     'answer':[{'content':g['content'],'reply_timespan':[g['time_seconds'],g['time_seconds']+4],
                                'related_timespan':[g['time_seconds'],g['time_seconds']+4]} for g in spec['gold_points']]})
    latest=read(RUN/'latest.json')
    if latest['step']!=23:raise ValueError('expected final trained checkpoint step 23')
    checkpoint=Path(latest['checkpoint']);verify_committed_checkpoint(checkpoint,23)
    for name,value in (('selection.json',selection),('gold.json',gold)):
        path=output/name
        if path.exists() and read(path)!=value:raise ValueError('frozen selection or gold changed')
        if not path.exists():write(path,value)
    manifest=output/'episodes.jsonl'
    body=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
    if manifest.exists() and manifest.read_text()!=body:raise ValueError('frozen frame manifest changed')
    if not manifest.exists():manifest.write_text(body)
    sources=[Path(__file__),ROOT/'scripts/eval_jevany_streaming.py',ROOT/'scripts/merge_jevany_benchmark.py',
             ROOT/'scripts/jevany_rl_reward.py',ROOT/'src/pace_clm/jevany_streaming.py',
             ROOT/'src/pace_clm/qwen_controller_runtime.py',ROOT/'src/pace_clm/evaluation_policy.py',
             SG/'zs_infer/pauc_f.py',SG/'zs_infer/scoring.py',
             RUN/'perfect-episodes.json',RUN/'reward-variance-groups.csv',RUN/'data/episodes.jsonl',
             checkpoint/'controller_adapter/adapter_model.safetensors',checkpoint/'pointer_head.pt',
             checkpoint/'model_identity.json',checkpoint/'commit.json',manifest,output/'gold.json',output/'selection.json']
    source_hashes={str(p):sha(p) for p in sources}
    video_hashes={item['video_path']:sha(item['video_path']) for item in selection}
    freeze={'node':NODE,'allocation':38177,'gpus':list(range(8)),'checkpoint':str(checkpoint),
            'checkpoint_step':23,'selection':'top 20 by response count in original perfect trajectories; ties by earlier source update, then span ID',
            'episodes':20,'decisions':sum(len(r['inference_source_indices']) for r in records),
            'gold_events':sum(len(r['answer']) for r in gold),'source_hashes':source_hashes,'video_hashes':video_hashes,
            'inference':'unchanged canonical evaluator; raw learned argmax; response_margin=0; promoted frozen generator; 96 answer tokens',
            'context':'continuous self-generated history; 32 frames; same integer source indices and absolute second timestamps as training',
            'scoring':'unchanged GPT-4.1 cumulative-answer rubric; pauc-f-v1 omega .5; complete original four-second gold windows',
            'scope':'selected training episodes, not held-out generalization; original perfect samples used stochastic actions and policy versions before their training updates'}
    path=output/'freeze.json'
    if path.exists() and read(path)!=freeze:raise ValueError('rerun configuration changed')
    if not path.exists():write(path,freeze)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in source_hashes.items():
        target=snapshot/(digest+Path(name).suffix)
        if not target.exists():shutil.copyfile(name,target)
    return freeze,selection


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=RUN/'diagnostics/final-top20-perfect-v1')
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze,selection=prepare(output)
    if args.prepare_only:print(json.dumps({k:freeze[k] for k in ('episodes','decisions','gold_events','checkpoint_step')}));return
    if socket.gethostname()!=NODE:raise RuntimeError('authorized node only')
    allocation=subprocess.check_output(['squeue','-h','-j','38177','-o','%T %N'],text=True).strip()
    if allocation!='RUNNING '+NODE:raise RuntimeError('original allocation unavailable')
    gpu_lock=(ROOT/'outputs/jevany-2b-rl/gpu.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(devices)!=8 or any('H200' not in d for d in devices):raise RuntimeError('requires original eight H200s')
    logs=output/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        TRITON_CACHE_DIR='/tmp/pace-rl-triton',TORCHINDUCTOR_COMPILE_THREADS='4',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))
    def status(state,**fields):
        row={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'node':NODE,**fields}
        write(output/'status.json',row);print(json.dumps(row),flush=True)
    def command(argv,name):
        for path,digest in freeze['source_hashes'].items():
            if sha(path)!=digest:raise ValueError('frozen source/input changed: '+path)
        with (logs/name).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    try:
        started=time.monotonic();rollout=output/'rollout';rollout.mkdir(exist_ok=True)
        if not (rollout/'predictions.jsonl').exists():
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('node GPUs are busy; do not interrupt another task')
            status('generating',episodes=20,decisions=freeze['decisions'])
            view=checkpoint_view(output,Path(freeze['checkpoint']))
            command([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',ROOT/'scripts/eval_jevany_streaming.py',
                     '--controller',view,'--validation-manifest',output/'episodes.jsonl','--output-dir',rollout,
                     '--response-margin','0','--answer-model','promoted','--max-new-tokens','96','--record-action-logits'],'rollout.log')
            command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',rollout],'merge.log')
        predictions=rows(rollout/'predictions.jsonl');ids=[r['span_id'] for r in selection]
        if [p['question_id'] for p in predictions]!=ids:raise ValueError('missing, duplicate or reordered episodes')
        for item,prediction in zip(selection,predictions,strict=True):
            trace=prediction['jevany_action_trace']
            if len(trace)!=item['seconds'] or [x['frame_index'] for x in trace]!=list(range(item['seconds'])):
                raise ValueError('incomplete episode decisions')
            for x in trace:
                if not all(math.isfinite(v) for v in x['logits']) or x['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:x['logits'][i])]:
                    raise ValueError('action is not the raw learned argmax')
            fires=[x['frame_index'] for x in trace if x['action']=='Response']
            if fires!=[r['time'] for r in prediction['model_response_list']]:raise ValueError('missing or duplicate physical fires')
        status('gpt_scoring',episodes=20)
        request_path=output/'requests.jsonl';response_path=output/'responses.jsonl'
        OFFICIAL.create_openai_batch_input(str(rollout/'predictions.jsonl'),str(output/'gold.json'),str(request_path),-1,MODEL)
        requests=rows(request_path)
        responses,usage=judge_requests(requests,ROOT/'outputs/jevany-2b-rl/reward-audit-v1/gpt-cache.sqlite')
        validate_judgments(requests,responses);jsonl(response_path,responses)
        scores=output/'scores.json'
        OFFICIAL.process_openai_batch_output(str(rollout/'predictions.jsonl'),str(output/'gold.json'),str(response_path),str(scores))
        aggregate=score_pauc_f(rollout/'predictions.jsonl',scores,omega=.5,judge_model=MODEL)
        by_id={r['question_id']:r for r in read(scores)}
        if set(by_id)!=set(ids):raise ValueError('missing episode judgments')
        results=[]
        for item,prediction in zip(selection,predictions,strict=True):
            directory=output/'per-episode'/item['span_id'];directory.mkdir(parents=True,exist_ok=True)
            jsonl(directory/'prediction.jsonl',[prediction]);write(directory/'scores.json',[by_id[item['span_id']]])
            metrics=score_pauc_f(directory/'prediction.jsonl',directory/'scores.json',omega=.5,judge_model=MODEL)
            results.append({**item,'final_pauc_f':metrics['pauc_f'],'final_responses':metrics['num_fires'],
                            'final_useful':metrics['num_value_adding_fires'],'final_wasted':metrics['num_wasted_fires'],
                            'final_precision':metrics['fire_precision'],'final_temporal_recall':metrics['pauc_recall_component'],
                            'delta_from_original_best':metrics['pauc_f']-100})
        result={'checkpoint_step':23,'episodes':20,'mean_episode_pauc_f':sum(r['final_pauc_f'] for r in results)/20,
                'global_pauc_f':aggregate,'episodes_still_100':sum(abs(r['final_pauc_f']-100)<1e-9 for r in results),
                'original_best_mean_episode_pauc_f':100.,'original_perfect_responses':freeze['gold_events'],
                'judge_usage':usage,'elapsed_seconds':time.monotonic()-started,'per_episode':results,
                'scope':freeze['scope']}
        write(output/'result.json',result)
        fields=['video_id','question','seconds','source_policy_version','responses','perfect_samples','original_four_pauc_f',
                'final_pauc_f','final_responses','final_useful','final_wasted','final_precision','final_temporal_recall','delta_from_original_best','span_id']
        with (output/'comparison.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
        lines=['# Final checkpoint on 20 formerly perfect episodes','',
               f"Mean episode PAUC-F: {result['mean_episode_pauc_f']:.4f}; aggregate PAUC-F: {aggregate['pauc_f']:.4f}; still perfect: {result['episodes_still_100']}/20.",
               f"Final useful / wasted responses: {aggregate['num_value_adding_fires']} / {aggregate['num_wasted_fires']}. Original perfect trajectories had {freeze['gold_events']} useful responses and zero waste.",'',
               'Selected training episodes, not held-out data. Original best-of-four samples used earlier policy versions and stochastic actions; rerun uses final checkpoint 23 and raw learned argmax. Identical frames, causal initialization, trusted intervals, generator, gold windows, and official scoring contract.','',
               '| Video | Gold events | Original best | Final PAUC-F | Final useful / wasted |',
               '|---|---:|---:|---:|---:|']
        for r in results:lines.append(f"| {r['video_id']} | {r['responses']} | 100 | {r['final_pauc_f']:.2f} | {r['final_useful']} / {r['final_wasted']} |")
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        status('complete',mean_episode_pauc_f=result['mean_episode_pauc_f'],global_pauc_f=aggregate['pauc_f'],episodes_still_100=result['episodes_still_100'])
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':main()
