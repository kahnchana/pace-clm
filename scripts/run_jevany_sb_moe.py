#!/usr/bin/env python3
"""Resumable PO/PO-I campaign for the committed step-160 2B + promoted MoE."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from eval_jevany_tv_vad160 import ROOT, SG, PARENT, NODE, read, rows, sha, write
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.inference.qwen_stream import stream_v25_qwen
from zs_infer.adapters.common import decode_seconds
from zs_infer.adapters import streamingbench as po
from zs_infer.sb_po_interact import score_records
from zs_infer.sb_po_trained import run_trained_policy

DEFAULT = PARENT/'evaluation/sb-po-poi-moe-step160'
POLICIES = ('independent','append','replace_scoped')
ANNOTATION = SG/'datasets/streamingbench/questions_proactive.json'
MANIFEST = SG/'outputs/evals/sb_po_i/manifest.jsonl'


def status(directory,state,**fields):
    write(directory/'status.json',dict(state=state,utc=datetime.now(timezone.utc).isoformat(),
                                     node=NODE,allocation=38226,**fields))


def assignments(tasks):
    result = [[] for _ in range(8)];loads = [0.]*8
    for index in sorted(range(len(tasks)),key=lambda i:(-tasks[i]['cost'],tasks[i]['id'])):
        rank = min(range(8),key=lambda r:(loads[r],r))
        result[rank].append(index);loads[rank] += tasks[index]['cost']
    return result


def verify(directory):
    frozen = read(directory/'freeze.json')
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen source changed: '+path)
    return frozen


def prepare(directory):
    from pace_clm.rl_anchor import verify_committed_checkpoint
    from pace_clm.stream_gemma_adapter import validate_promoted_contract
    verify_committed_checkpoint(PARENT/'step-000160',160)
    if (directory/'freeze.json').exists():verify(directory);return
    directory.mkdir(parents=True,exist_ok=True)
    examples,_ = po.load_examples(ANNOTATION,SG/'datasets/streamingbench/videos')
    episodes = rows(MANIFEST)
    if len(examples)!=250 or len(episodes)!=50 or sum(len(e['requests']) for e in episodes)!=250:
        raise ValueError('incomplete PO / PO-I populations')
    if sha(MANIFEST)!='3314dbd34e3e8e18a978ffc6dcbe194e1f8a3b87edf9f592178ebc188a0329dc':
        raise ValueError('PO-I manifest differs from promoted reference')
    if any(not e['video'] or not Path(e['video']).is_file() for e in examples+episodes):
        raise ValueError('missing source media')
    tasks = [dict(id='po-'+e['qid'].replace('/','-'),kind='po',example=e,
                  cost=max(1.,e['window_end']-e['query_time']+1)) for e in examples]
    tasks += [dict(id=e['episode_id'],kind='poi',episode=e,
                   cost=3*(e['evaluation_end']-e['evaluation_start']+1)) for e in episodes]
    write(directory/'tasks.json',tasks);write(directory/'assignments.json',assignments(tasks))
    contract = validate_promoted_contract(SG)
    paths = [Path(__file__),ROOT/'scripts/run_jevany_sb_moe.sh',ROOT/'scripts/jevany_po_moe_runtime.py',
        ROOT/'scripts/eval_jevany_streaming.py',ROOT/'scripts/generate_selected_responses.py',
        ROOT/'src/pace_clm/qwen_controller_runtime.py',ROOT/'src/pace_clm/jevany_streaming.py',
        ROOT/'src/pace_clm/evaluation_policy.py',SG/'zs_infer/sb_po_trained.py',
        SG/'zs_infer/sb_po_interact.py',SG/'zs_infer/adapters/streamingbench.py',
        SG/'zs_infer/adapters/common.py',SG/'src/stream_gemma/inference/qwen_stream.py',
        SG/'src/stream_gemma/v24_memory.py',SG/'third_party/streamingbench/src/data/count.py',
        ANNOTATION,MANIFEST,directory/'tasks.json',directory/'assignments.json']
    write(directory/'freeze.json',dict(checkpoint=str(PARENT/'step-000160'),step=160,gpus=8,
        source_hashes={str(p.resolve()):sha(p) for p in paths},generator=contract,
        generator_adapter_sha256=sha(Path(contract['adapter'])/'adapter_model.safetensors'),
        expected_runtime=read(PARENT/'runtime-contract.json'),
        settings=dict(fps=1,window_frames=32,response_margin=0,action='raw learned argmax',
                      answer_tokens=128,generator='frozen promoted Qwen MoE',memory='last-response/aged-out/v2',
                      subtitles='off',poi_policies=POLICIES),
        populations=dict(po=250,poi_videos=50,poi_requests=250,poi_primary_transitions=185),
        authorization='User approved the PO / PO-I plan and requested completion with final numbers on this node.'))
    print(json.dumps(dict(prepared=str(directory),tasks=len(tasks),rank_costs=[sum(tasks[i]['cost'] for i in a) for a in assignments(tasks)])),flush=True)


def check_actions(trace):
    for t in trace:
        if t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])]:
            raise ValueError('action differs from raw learned argmax')
        if not 1<=t['input_frames']<=32:raise ValueError('invalid causal window')


def run_po(adapter,example,*,end_override=None):
    end = example['window_end'] if end_override is None else end_override
    images,indices = decode_seconds(example['video'],start=example['query_time'],end=end,
                                    source_fps=1,include_end=True,return_source_indices=True)
    times = [float(i) for i in indices]
    adapter.reset()
    fires = stream_v25_qwen(adapter.runtime,images,question=example['question'],system_prompt=SYSTEM_PROMPT,
        max_rounds=32,max_new_tokens=128,frame_times=times,memory_mode='last-response',
        memory_instruction_version='v2',memory_select='aged-out') if len(images)>=2 else []
    check_actions(adapter.trace)
    if len(images)>=2 and len(adapter.trace)!=len(times):raise ValueError('incomplete PO trace')
    first = fires[0] if fires else dict(time=float(end or example['query_time']),content='')
    return dict(qid=example['qid'],answered=float(first['time']) if fires else False,
        dialog_history=[dict(role='assistant',time=float(first['time']),content=first['content'])],
        all_fires=fires,frame_times=times,controller_trace=adapter.trace)


def validate_poi(record,episode):
    if record['requests']!=episode['requests'] or set(record['policies'])!=set(POLICIES):
        raise ValueError('PO-I request population changed')
    times = record['frame_times']
    if times[0]!=episode['evaluation_start'] or abs(times[-1]-episode['evaluation_end'])>=1.01:
        raise ValueError('incomplete PO-I observation horizon')
    for policy,result in record['policies'].items():
        if [t['time'] for t in result['trace']]!=times or len(result['controller_trace'])!=len(times):
            raise ValueError('incomplete PO-I trace')
        check_actions(result['controller_trace'])
        if result['fires']!=[f for t in result['trace'] for f in t['accepted']]:
            raise ValueError('PO-I fire audit mismatch')
        for index,(t,control) in enumerate(zip(result['trace'],result['controller_trace'],strict=True)):
            active = max((r for r in episode['requests'] if r['query_time']<=t['time']),key=lambda r:r['query_time'])
            if t['active_request_id']!=active['request_id'] or t['frame_times'][-1]!=t['time']:
                raise ValueError('wrong active request or future frame')
            if policy=='independent' and min(t['frame_times'])<active['query_time']:
                raise ValueError('independent request retained previous window')
            if policy!='independent' and t['frame_times']!=times[max(0,index-31):index+1]:
                raise ValueError('persistent visual history changed')
            if policy!='append' and t['memory_owner_request_id'] not in (None,active['request_id']):
                raise ValueError('memory scoped to wrong request')
            if len(t['frame_times'])!=control['input_frames']:
                raise ValueError('controller window differs from request hook')
            if any(not f['content'].strip() or f['time']!=t['time'] for f in t['accepted']):
                raise ValueError('invalid accepted fire')


def run_poi(adapter,episode):
    images,indices = decode_seconds(episode['video'],start=episode['evaluation_start'],end=episode['evaluation_end'],
                                    source_fps=1,include_end=True,return_source_indices=True)
    times = [float(i) for i in indices]
    record = {k:episode[k] for k in ('episode_id','source_video','requests','evaluation_start','evaluation_end')}
    record.update(frame_times=times,policies={})
    for policy in POLICIES:
        adapter.reset()
        result = run_trained_policy(adapter.runtime,images,times,episode,policy)
        result['controller_trace'] = adapter.trace
        record['policies'][policy] = result
    validate_poi(record,episode)
    return record


def smoke(adapter,directory,tasks):
    episode = next(t['episode'] for t in tasks if t['id']=='sb-po-i-sample_15')
    record = run_poi(adapter,episode)
    write(directory/'smoke/poi.json',record)
    examples = {t['example']['qid']:t['example'] for t in tasks if t['kind']=='po'}
    comparisons = []
    for index,request in enumerate(episode['requests']):
        qid = f"{request['source_entry_index']}/{request['source_question_index']}"
        example = examples[qid]
        interval_end = episode['requests'][index+1]['query_time']-1 if index+1<len(episode['requests']) else episode['evaluation_end']
        end = min(example['window_end'],interval_end)
        ordinary = run_po(adapter,example,end_override=end)
        write(directory/f'smoke/po-{qid.replace("/","-")}.json',ordinary)
        expected = [f for f in record['policies']['independent']['fires'] if f['request_id']==request['request_id'] and f['time']<=end]
        expected = [{k:f[k] for k in ('time','content')} for f in expected]
        actual = [{k:f[k] for k in ('time','content')} for f in ordinary['all_fires']]
        if actual!=expected:raise ValueError('ordinary / independent smoke fire mismatch: '+qid)
        comparisons.append(dict(qid=qid,passed=True,first_fire=actual[:1]))
    write(directory/'records/sb-po-i-sample_15.json',dict(kind='poi',record=record))
    write(directory/'smoke-passed.json',dict(passed=True,requests=5,policies=POLICIES,comparisons=comparisons))


def worker(directory):
    import torch
    from jevany_po_moe_runtime import ControllerMoE
    verify(directory)
    rank = int(__import__('os').environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    tasks = read(directory/'tasks.json')
    adapter = ControllerMoE(PARENT/'step-000160')
    if rank==0 and not (directory/'smoke-passed.json').exists():
        smoke(adapter,directory,tasks)
    while not (directory/'smoke-passed.json').exists():time.sleep(2)
    for index in read(directory/'assignments.json')[rank]:
        task = tasks[index];path = directory/'records'/(task['id']+'.json')
        if path.exists():continue
        record = run_po(adapter,task['example']) if task['kind']=='po' else run_poi(adapter,task['episode'])
        write(path,dict(kind=task['kind'],record=record))
        print(json.dumps(dict(rank=rank,completed=task['id'],kind=task['kind'])),flush=True)
    write(directory/f'rank-{rank}.done.json',dict(success=True))


def po_counts(records,examples):
    official = corrected = answer = blank_credit = 0
    for r in records:
        e = examples[r['qid']];last = r['dialog_history'][-1]
        time_ok = abs(last['time']-e['ground_truth_time'])<=2
        official += time_ok
        first = next((f for f in r['all_fires'] if f['content'].strip()),None)
        correct = bool(first and abs(first['time']-e['ground_truth_time'])<=2)
        corrected += correct
        answer += bool(correct and e['keywords'] in first['content'])
        blank_credit += bool(time_ok and not last['content'].strip())
    n = len(records)
    return dict(questions=n,official_time_accuracy=100*official/n,corrected_time_accuracy=100*corrected/n,
                answer_accuracy=100*answer/n,official_time_correct=official,corrected_time_correct=corrected,
                answer_correct=answer,blank_placeholder_timing_credit=blank_credit)


def score(directory):
    tasks = read(directory/'tasks.json')
    saved = {t['id']:read(directory/'records'/(t['id']+'.json')) for t in tasks}
    ordinary = [saved[t['id']]['record'] for t in tasks if t['kind']=='po']
    interactive = [saved[t['id']]['record'] for t in tasks if t['kind']=='poi']
    examples = {t['example']['qid']:t['example'] for t in tasks if t['kind']=='po'}
    for t in tasks:
        if t['kind']=='poi':validate_poi(saved[t['id']]['record'],t['episode'])
    po_dir = directory/'po';po_dir.mkdir(exist_ok=True)
    output = po.assemble(read(ANNOTATION),ordinary,model_key='PACE2BMoE')
    write(po_dir/'output_proactive.json',output)
    subprocess.run([sys.executable,str(SG/'third_party/streamingbench/src/data/count.py'),
        '--model','PACE2BMoE','--task','proactive','--src',str((po_dir/'output_proactive.json').resolve())],cwd=po_dir,check=True)
    metrics = po_counts(ordinary,examples)
    official = read(po_dir/'PACE2BMoE_stats.json')
    if sum(v.get('time_correct',0) for v in official.values())!=metrics['official_time_correct']:
        raise ValueError('official timing scorer mismatch')
    if sum(v.get('answer_correct',0) for v in official.values())!=metrics['answer_correct']:
        raise ValueError('official answer scorer mismatch')
    write(po_dir/'result.json',metrics)
    poi_dir = directory/'poi';poi_dir.mkdir(exist_ok=True)
    (poi_dir/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in interactive))
    poi_metrics = score_records(interactive,bootstrap_samples=10000,seed=42)
    if any(int(v['counts']['transitions'])!=185 for v in poi_metrics['policies'].values()):
        raise ValueError('PO-I primary population changed')
    write(poi_dir/'result.json',poi_metrics)
    result = dict(step=160,generator='promoted Qwen MoE',po=metrics,poi=poi_metrics)
    write(directory/'result.json',result)
    status(directory,'complete',po=metrics,poi_policies=list(poi_metrics['policies']))
    print(json.dumps(result),flush=True)


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
    if torch.cuda.device_count()!=8:raise RuntimeError('eight GPUs required')
    actual = dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual):raise RuntimeError('pinned runtime mismatch')
    write(directory/'runtime.json',actual)
    verify_committed_checkpoint(Path(frozen['checkpoint']),160)
    if (directory/'result.json').exists():return
    status(directory,'smoke_and_inference',po_completed=0,poi_completed=0)
    with (directory/'worker.log').open('a') as handle:
        process = subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                                   str(Path(__file__).resolve()),'worker','--output',str(directory)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        while process.poll() is None:
            records = list((directory/'records').glob('*.json'))
            status(directory,'inference' if (directory/'smoke-passed.json').exists() else 'smoke',
                   po_completed=sum(p.name.startswith('po-') for p in records),
                   poi_completed=sum(p.name.startswith('sb-po-i-') for p in records))
            time.sleep(10)
        if process.returncode:
            status(directory,'failed',error='worker failure; completed records preserved',exit_code=process.returncode)
            raise RuntimeError('PO/PO-I workers failed')
    if len(list(directory.glob('rank-*.done.json')))!=8:raise ValueError('missing successful worker markers')
    status(directory,'scoring')
    score(directory)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run','worker','score'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args = parser.parse_args()
    globals()[args.mode](args.output.resolve())
