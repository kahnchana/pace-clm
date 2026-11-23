#!/usr/bin/env python3
"""Resume uncompleted WEB episodes; explicitly audit user-requested perfect imputation."""
import argparse
import fcntl
import json
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from run_jevany_flash_parallel import DEFAULT, NODE, ALLOCATION, verify as verify_original
from eval_jevany_tv_vad160 import ROOT, SG, read, rows, sha, write

SOURCE=DEFAULT/'web'
TARGET=SOURCE/'retry-remaining'

def status(state,**fields):
    value=dict(state=state,node=NODE,allocation=ALLOCATION,utc=datetime.now(timezone.utc).isoformat(),**fields)
    write(TARGET/'status.json',value);write(SOURCE/'status.json',value)

def verify():
    frozen=read(TARGET/'freeze.json')
    for path,digest in frozen['source_hashes'].items():
        if sha(path)!=digest: raise ValueError('frozen recovery source changed: '+path)
    return frozen

def prepare():
    frozen=verify_original(DEFAULT)
    TARGET.mkdir(exist_ok=True);rollout=TARGET/'rollout';rollout.mkdir(exist_ok=True)
    if (TARGET/'freeze.json').exists(): return verify()
    paths=dict(frozen['source_hashes'])
    for name in ('evaluation-manifest.jsonl','selection.json','cohorts.json','gold.json'):
        shutil.copyfile(SOURCE/name,TARGET/name)
        paths[str((TARGET/name).resolve())]=sha(TARGET/name)
    copied=0
    for rank in range(8):
        source=SOURCE/'rollout'/f'predictions.rank{rank:02d}.jsonl'
        destination=rollout/source.name
        if destination.exists(): raise ValueError('unexpected pre-existing recovery shard')
        shutil.copyfile(source,destination);copied+=len(rows(source))
        paths[str(source.resolve())]=sha(source)
    if copied!=277: raise ValueError('expected 277 preserved original episodes')
    for path in (Path(__file__),ROOT/'scripts/retry_jevany_flash_web.sh',ROOT/'scripts/eval_jevany_flash_web_retry.py'):
        paths[str(path.resolve())]=sha(path)
    write(TARGET/'freeze.json',dict(frozen,source_hashes=paths,reused_original_episodes=copied,
          recovery_authorization='User explicitly requested remaining WEB retries; skip blocked entry and give it a perfect score if still blocked.',
          failure_policy='Retry each unfinished episode. Record provider-content-blocked episodes separately and continue other independent videos. No controller action or answer substitution.',
          imputation_policy='Post-hoc 100 PAUC-F for blocked episodes at all omegas; virtual one useful fire per gold event and recall 100%. Report measured-only and imputed-inclusive results separately.'))
    return verify()

def aggregate_with_imputation(measured,failed_gold_events,failed_episodes):
    """Apply explicitly labeled perfect per-event precision and recall contributions."""
    useful=measured['num_value_adding_fires'];fires=measured['num_fires'];gold=measured['num_gold_turns']
    precision=(useful+failed_gold_events)/(fires+failed_gold_events) if fires+failed_gold_events else 0.
    recall=(measured['pauc_recall_component']/100*gold+failed_gold_events)/(gold+failed_gold_events) if gold+failed_gold_events else 0.
    return dict(pauc_f=100*2*precision*recall/(precision+recall) if precision+recall else 0.,
                fire_precision=100*precision,temporal_recall=100*recall,
                measured_episodes=measured['num_videos'],imputed_episodes=failed_episodes,
                imputed_gold_events=failed_gold_events,imputed_virtual_useful_fires=failed_gold_events,
                headline_eligible=False,label='Includes user-requested perfect-score imputation; not a fully measured benchmark result')

def score(predictions,failures,frozen):
    from jevany_mixed_reward import score_predictions
    from zs_infer.pauc_f import score_pauc_f
    by_id={p['question_id']:p for p in predictions};failed={p['question_id']:p for p in failures}
    wanted=read(TARGET/'selection.json')
    if set(by_id)&set(failed) or set(by_id)|set(failed)!=set(wanted) or len(predictions)!=len(by_id) or len(failures)!=len(failed): raise ValueError('missing/duplicate/unexpected success or failure IDs')
    specs={r['question_id']:r for r in rows(TARGET/'evaluation-manifest.jsonl')}
    for pred in predictions:
        trace=pred['jevany_action_trace']
        if len(trace)!=len(specs[pred['question_id']]['inference_source_indices']): raise ValueError('incomplete successful controller trace')
        if any(t['action']!=('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace): raise ValueError('non-argmax decision')
        if len(pred['gemini_calls'])!=len(pred['model_response_list']) or any(c['status']!=200 or c['model']!='gemini-3.8-flash' for c in pred['gemini_calls']): raise ValueError('invalid successful API trace')
    gold={g['question_id']:g for g in read(TARGET/'gold.json')}
    ordered=[by_id[q] for q in wanted if q in by_id]
    (TARGET/'rollout/predictions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in ordered))
    write(TARGET/'blocked-episodes.json',failures)
    status('judging',completed=len(predictions),blocked=len(failures),total=len(wanted))
    measured_metrics,usage=score_predictions(ordered,[gold[q] for q in wanted if q in by_id],TARGET/'scoring',TARGET/'judge-cache.sqlite')
    judged={r['question_id']:r for r in read(TARGET/'scoring/scores.json')}
    cohorts=read(TARGET/'cohorts.json');result={}
    temp=TARGET/'cohort-scoring';temp.mkdir(exist_ok=True)
    for cohort,ids in cohorts.items():
        valid=[q for q in ids if q in by_id];missing=[q for q in ids if q in failed]
        failed_events=sum(len(gold[q]['answer']) for q in missing)
        (temp/'predictions.jsonl').write_text(''.join(json.dumps(by_id[q])+'\n' for q in valid))
        write(temp/'scores.json',[judged[q] for q in valid])
        result[cohort]={}
        for omega in (0.,.5,1.):
            measured=score_pauc_f(temp/'predictions.jsonl',temp/'scores.json',omega=omega)
            result[cohort][str(omega)]=dict(measured_only=measured,
                user_requested_perfect_imputation=aggregate_with_imputation(measured,failed_events,len(missing)))
    imputed={q:dict(pauc_f=100.,omegas=[0.,.5,1.],gold_events=len(gold[q]['answer']),measured=False,
                reason=failed[q]['provider_call']['response']['promptFeedback'],authorization='User requested perfect score if blocked entry does not work') for q in failed}
    output=dict(step=160,answer_model='gemini-3.8-flash',measured_episodes=len(predictions),blocked_episodes=len(failures),
                blocked_ids=sorted(failed),cohorts=result,per_episode_measured=measured_metrics,per_episode_imputed=imputed,
                judge_usage=usage,imputation_policy=frozen['imputation_policy'],all_500_accounted_for=True)
    write(TARGET/'result.json',output)
    parent=read(DEFAULT/'status.json');write(DEFAULT/'status.before-web-retry.json',parent)
    write(DEFAULT/'status.json',dict(state='complete_with_user_requested_web_imputation',node=NODE,allocation=ALLOCATION,
          completed=['web','ego','tv','vad'],web_measured=len(predictions),web_imputed=len(failures),web_result=str(TARGET/'result.json'),utc=datetime.now(timezone.utc).isoformat()))
    lines=['# Step-160 controller + Gemini Flash: WEB recovery','',f'{len(predictions)} episodes measured; {len(failures)} provider-blocked episodes assigned perfect scores at the user’s explicit request.','',
           '| Cohort | Omega | Measured only | Including perfect imputation |','|---|---:|---:|---:|']
    for c in ('web_all','web_train','web_heldout'):
        for w in ('0.0','0.5','1.0'):
            r=result[c][w];lines.append(f"| {c} | {w} | {r['measured_only']['pauc_f']:.2f} | {r['user_requested_perfect_imputation']['pauc_f']:.2f} |")
    lines.extend(['','Imputed values are not model-generated answers or GPT judgments. Original artifacts and all provider failure audits remain preserved.',''])
    (TARGET/'completion-report.md').write_text('\n'.join(lines))
    status('complete_with_imputation' if failures else 'complete',completed=len(predictions),blocked=len(failures),total=len(wanted),result=str(TARGET/'result.json'))
    print('\n'.join(lines),flush=True)

def run():
    import torch
    if socket.gethostname()!=NODE or subprocess.check_output(['squeue','-h','-j',str(ALLOCATION),'-o','%T %N'],text=True).strip()!='RUNNING '+NODE: raise RuntimeError('authorized current node/allocation only')
    lock=(ROOT/f'outputs/jevany-2b-rl/evaluation-gpu-{ALLOCATION}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(): raise RuntimeError('GPUs busy')
    frozen=prepare()
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    if any(actual[k]!=frozen['expected_runtime'][k] for k in actual) or torch.cuda.device_count()!=8: raise RuntimeError('pinned eight-GPU runtime mismatch')
    write(TARGET/'runtime.json',actual)
    argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',ROOT/'scripts/eval_jevany_flash_web_retry.py',
          '--controller',SOURCE/'full_run/checkpoints/step-000160','--validation-manifest',TARGET/'evaluation-manifest.jsonl',
          '--selection-ids',TARGET/'selection.json','--output-dir',TARGET/'rollout','--response-margin','0',
          '--answer-model','gemini-3.8-flash','--subtitles','off','--max-new-tokens','96','--record-action-logits']
    for attempt in range(1,4):
        verify();status('inference',reused=277,total=500,workers=8,attempt=attempt)
        with (TARGET/'rollout.log').open('a') as log:
            process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(TARGET/'process.json',dict(pid=process.pid,argv=list(map(str,argv)),attempt=attempt))
            code=process.wait()
        if code==0: break
        if attempt==3: status('failed',exit_code=code);raise RuntimeError('inference failed after three attempts')
        time.sleep(10)
    predictions=[];failures=[]
    for rank in range(8):
        predictions.extend(rows(TARGET/'rollout'/f'predictions.rank{rank:02d}.jsonl'))
        file=TARGET/'rollout'/f'failures.rank{rank:02d}.jsonl'
        if file.exists(): failures.extend(rows(file))
    verify();score(predictions,failures,frozen)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args()
    if a.mode=='prepare': prepare()
    else: run()
