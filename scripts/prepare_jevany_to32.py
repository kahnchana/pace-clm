#!/usr/bin/env python3
"""Freeze the user-authorized continuation without changing the stopped run."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import statistics

from pace_clm.rl_mixed import episode_balanced_schedule

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2'
DEFAULT=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)


def schedule_to32(episodes,original):
    schedule=episode_balanced_schedule(episodes,budget=1000000)[:32]
    if len(schedule)!=32 or schedule[:len(original)]!=original:
        raise ValueError('continuation must preserve the entire original schedule prefix')
    return schedule


def expanded_replay_rows(original,schedule):
    """Extend credits by cycling frozen examples, preserving the existing prefix."""
    if not original:raise ValueError('empty supervised replay bank')
    required=max(b['anchor_start']+b['anchor_count'] for b in schedule)
    return [original[i%len(original)] for i in range(max(len(original),required))]


def extend_replay_bank(source,target,schedule):
    lines=source.read_text().splitlines(keepends=True)
    if not lines or any(not line.strip() for line in lines):raise ValueError('invalid replay bank')
    extended=expanded_replay_rows(lines,schedule)
    target.mkdir(parents=True,exist_ok=True)
    (target/'replay.jsonl').write_text(''.join(line if line.endswith('\n') else line+'\n' for line in extended))
    frozen=read(source.parent/'freeze.json')
    frozen['continuation_replay']={'source_manifest':str(source),'source_sha256':sha(source),
        'source_rows':len(lines),'expanded_rows':len(extended),
        'policy':'repeat the frozen bank in original order; no novel examples or coefficient changes',
        'expanded_manifest_sha256':sha(target/'replay.jsonl')}
    write(target/'freeze.json',frozen)
    return target/'replay.jsonl'


def aggregate(metrics,ids):
    values=[metrics[q] for q in ids]
    useful=sum(m['num_value_adding_fires'] for m in values);fires=sum(m['num_fires'] for m in values)
    gold=sum(m['num_gold_turns'] for m in values)
    area=sum(m['pauc_recall_component']/100*m['num_gold_turns'] for m in values)
    precision=useful/fires if fires else 0.;recall=area/gold if gold else 0.
    return {'episodes':len(values),'pauc_f':100*2*precision*recall/(precision+recall) if precision+recall else 0.,
        'mean_episode_pauc_f':statistics.mean(m['pauc_f'] for m in values),
        'fire_precision':100*precision,'temporal_recall':100*recall,
        'useful':useful,'wasted':fires-useful,'fires':fires,'gold_events':gold}


def borrow_evaluation(parent,run,name):
    source=parent/'evaluation'/name/'result.json'
    if not source.exists():return False
    target=run/'evaluation'/name/'result.json'
    if target.exists():return True
    result=copy.deepcopy(read(source));cohorts=read(run/'data/cohorts.json')
    if result.get('per_episode'):
        metrics=result['per_episode']
        selected={'gate':cohorts['gate']} if result['job']['kind'] in ('gate','baseline_smoke') else cohorts
        result['cohorts']={k:aggregate(metrics,ids) for k,ids in selected.items() if all(q in metrics for q in ids)}
        for key in ('paired_vs_sft','paired_vs_mixed_v1'):result.pop(key,None)
    result['borrowed_from']=str(source);result['borrowed_result_sha256']=sha(source)
    result['cohort_note']='Source split identities unchanged; sampled/unsampled groups refer to the complete planned 32-update schedule.'
    write(target,result)
    for subdir in ('rollout','scoring'):
        old=source.parent/subdir;new=target.parent/subdir
        if old.exists() and not new.exists():new.symlink_to(old,target_is_directory=True)
    return True


def prepare(run):
    if (run/'continuation.json').exists():raise FileExistsError('continuation already prepared; inspect existing freeze')
    latest=read(PARENT/'latest.json')
    if latest['step']!=15:raise ValueError('expected exact stopped step 15')
    checkpoint=Path(latest['checkpoint']);commit=read(checkpoint/'commit.json')
    for name,digest in commit['files'].items():
        if sha(checkpoint/name)!=digest:raise ValueError('checkpoint hash mismatch: '+name)
    original=read(PARENT/'data/schedule.json');episodes=rows(PARENT/'data/episodes.jsonl')
    schedule=schedule_to32(episodes,original);budget=schedule[-1]['exposure_after']
    if budget!=138460:raise ValueError('unexpected extension exposure')
    for path,digest in read(PARENT/'workflow-freeze.json')['source_hashes'].items():
        if sha(Path(path))!=digest:raise ValueError('parent frozen source changed: '+path)
    run.mkdir(parents=True,exist_ok=True);data=run/'data';code=run/'code'
    shutil.copytree(PARENT/'data',data);code.mkdir()
    write(data/'schedule.json',schedule)
    sampled={episodes[i]['span_id'] for batch in schedule for i in batch['episode_indices']}
    cohorts=read(data/'cohorts.json')
    for domain in ('web','ego'):
        cohorts[domain+'_sampled']=[q for q in cohorts[domain+'_train'] if q in sampled]
        cohorts[domain+'_unsampled']=[q for q in cohorts[domain+'_train'] if q not in sampled]
    write(data/'cohorts.json',cohorts);write(data/'sampled-identities.json',sorted(sampled))
    audit=read(data/'audit.json');audit.update(version='mixed-web-ego-episode-balanced-v2-to32',
        parent_run=str(PARENT),full_steps=[16,24,32],fast_steps=[20,28])
    audit['counts'].update(updates=32,scheduled_decisions=budget,domain_decisions=schedule[-1]['cumulative_domain_decisions'],
        sampled_episodes={d:sum(s['domain']==d and s['span_id'] in sampled for s in episodes) for d in ('ours','web','ego')},
        cohorts={k:len(v) for k,v in cohorts.items()})
    audit['prepared_hashes']={f.name:sha(f) for f in data.iterdir() if f.is_file() and f.name!='audit.json'}
    write(data/'audit.json',audit)
    anchor=extend_replay_bank(Path(read(PARENT/'freeze.json')['anchor_manifest']),run/'anchor-extended',schedule)
    trainer=(ROOT/'scripts/run_jevany_mixed_rloo.py').read_text()
    trainer=trainer.replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
    trainer=trainer.replace('a.decision_budget<=131072','a.decision_budget<=200000')
    trainer=trainer.replace("raise RuntimeError('near-silence: zero fires across all event-containing training trajectories')",
        "write_json(batch/'near-silence-warning.json', {'warning':'zero fires across event-containing training trajectories','policy':'user authorized continuation to step 32'})")
    (code/'run_jevany_mixed_rloo.py').write_text(trainer)
    workflow=(ROOT/'scripts/run_jevany_mixed_workflow.py').read_text()
    workflow=workflow.replace("ANCHOR=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1/anchor/replay.jsonl'",f'ANCHOR=Path({str(anchor)!r})')
    workflow=workflow.replace("sampling_unit=audit.get('sampling_unit','decision')", "sampling_unit=audit.get('sampling_unit','decision')")
    workflow=workflow.replace("sources+=list((run/'data').glob('*'))", "full=audit['full_steps'];fast=audit['fast_steps']\n    sources+=list((run/'data').glob('*'))\n    sources+=list((run/'code').glob('*.py'))+[run/'launch.sh',run/'runtime-contract.json',ROOT/'scripts/prepare_jevany_to32.py']")
    workflow=workflow.replace("'decision_budget':100000", "'decision_budget':audit['counts']['scheduled_decisions']")
    workflow=workflow.replace("'coverage_gate':{'useful_fraction_of_sft':.9", "'coverage_gate':{'enabled':False,'user_override':'continue to step 32 regardless of coverage','useful_fraction_of_sft':.9")
    workflow=workflow.replace("ROOT/'scripts/run_jevany_mixed_rloo.py','--output'", "run/'code/run_jevany_mixed_rloo.py','--output'")
    workflow=workflow.replace("'--decision-budget','100000'", "'--decision-budget',str(plan['decision_budget'])")
    workflow=workflow.replace("        status('training')", "        if not (run/'extension-resume-verification.json').exists():\n            status('restoring_step15')\n            command([PY,ROOT/'scripts/verify_jevany_mixed_resume.py','--run',run],'restore-extension.log')\n            restored=read(run/'resume_verification.json')\n            if not restored['verified'] or restored['step']!=15 or restored['additional_optimizer_steps']!=0:raise ValueError('step-15 restore gate failed')\n            write(run/'extension-resume-verification.json',restored)\n        status('training')")
    (code/'run_extension_workflow.py').write_text(workflow)
    evaluator=(ROOT/'scripts/eval_jevany_mixed_queue.py').read_text()
    evaluator=evaluator.replace("if reasons:write(run/'stop-request.json',", "if reasons:write(run/f'coverage-warning-step{job[\"step\"]}.json',")
    (code/'eval_jevany_mixed_queue.py').write_text(evaluator)
    for name in ('anchor-calibration.json','gate-passed.json','cross-node-parity.json','runtime-contract.json'):
        shutil.copy2(PARENT/name,run/name)
    for i in range(1,16):
        for name in (f'step-{i:06d}',f'batch-{i-1:06d}'):
            (run/name).symlink_to(PARENT/name,target_is_directory=True)
    write(run/'latest.json',{'step':15,'checkpoint':str(run/'step-000015')})
    parent_freeze=read(PARENT/'freeze.json');f=copy.deepcopy(parent_freeze)
    f.update(output=str(run),spans=str(data/'episodes.jsonl'),decision_budget=budget,episode_batches=schedule,anchor_manifest=str(anchor))
    replacements={str(PARENT/'data'/n):str(data/n) for n in ('episodes.jsonl','audit.json','schedule.json')}
    replacements[str(ROOT/'scripts/run_jevany_mixed_rloo.py')]=str(code/'run_jevany_mixed_rloo.py')
    old_anchor=Path(parent_freeze['anchor_manifest'])
    replacements[str(old_anchor)]=str(anchor)
    replacements[str(old_anchor.parent/'freeze.json')]=str(anchor.parent/'freeze.json')
    f['hashes']={replacements.get(path,path):sha(Path(replacements.get(path,path))) for path in f['hashes']}
    write(run/'freeze.json',f)
    for name in ('baseline-sft-smoke','baseline-sft','baseline-prior','gate-step1','final-step1','full-step6','full-step12','final-step15'):
        borrow_evaluation(PARENT,run,name)
    launch='''#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
PY="$ROOT/../stream-gemma/.venv/bin/python"
RUN="__RUN__"
"$PY" - "$RUN" <<'CHECK'
import json,sys,socket,torch
from pathlib import Path
run=Path(sys.argv[1]);expected=json.loads((run/'runtime-contract.json').read_text())
actual={'node':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
assert all(actual[k]==expected[k] for k in ('torch','cuda','cudnn')),actual
(run/('pinned-runtime-'+actual['node']+'.json')).write_text(json.dumps(actual,indent=2)+'\\n')
print('Pinned runtime:',json.dumps(actual),flush=True)
CHECK
case "${1:-}" in
 prepare) exec "$PY" -u "$RUN/code/run_extension_workflow.py" --run "$RUN" --prepare-only ;;
 train)
  [[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]]
  exec "$PY" -u "$RUN/code/run_extension_workflow.py" --run "$RUN" ;;
 eval)
  [[ "$(hostname)" == ip-10-3-70-177 && "${SLURM_JOB_ID:-}" == 38226 ]]
  exec "$PY" -u "$RUN/code/wait_and_evaluate.py" --run "$RUN" ;;
 *) exit 2 ;;
esac
'''.replace('__RUN__',str(run))
    (run/'launch.sh').write_text(launch)
    wait='''import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from prepare_jevany_to32 import ROOT,PARENT,borrow_evaluation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
(run/'evaluation-status.json').write_text(json.dumps({'state':'waiting_for_existing_evaluation','utc':datetime.now(timezone.utc).isoformat(),'node':'ip-10-3-70-177'})+'\\n')
with (ROOT/'outputs/jevany-2b-rl/evaluation-gpu-38226.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 fcntl.flock(lock,fcntl.LOCK_UN)
borrow_evaluation(PARENT,run,'reference-mixed-v1-step16')
os.execv(sys.executable,[sys.executable,'-u',str(run/'code/eval_jevany_mixed_queue.py'),'--run',str(run)])
'''
    (code/'wait_and_evaluate.py').write_text(wait)
    (run/'cache').mkdir()
    for name in ('training.sqlite','evaluation.sqlite'):
        with sqlite3.connect(PARENT/'cache'/name) as source:
            with sqlite3.connect(run/'cache'/name) as dest:source.backup(dest)
    write(run/'continuation.json',{'authorized_utc':datetime.now(timezone.utc).isoformat(),
        'user_instruction':'continue training until step 32 no matter what; keep evaluations running',
        'parent_run':str(PARENT),'resume_step':15,'target_step':32,'additional_updates':17,
        'preserved_original_schedule_updates':len(original),'decision_budget':budget,
        'additional_decisions':budget-schedule[14]['exposure_after'],'episode_counts':{'ours':128,'web':64,'ego':64},
        'full_steps':[16,24,32],'fast_steps':[20,28],'coverage_policy':'warnings only; no coverage or near-silence quality stops',
        'parent_checkpoint_commit_sha256':sha(checkpoint/'commit.json'),'original_stop_request':read(PARENT/'stop-request.json'),
        'source_snapshot_hashes':{p.name:sha(p) for p in code.glob('*.py')}})
    return {'run':str(run),'target_step':32,'decisions':budget,'additional_decisions':budget-schedule[14]['exposure_after']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args()
    print(json.dumps(prepare(a.output.resolve())),flush=True)


if __name__=='__main__':main()
