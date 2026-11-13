#!/usr/bin/env python3
"""Reuse immutable v1 splits and freeze the approved 4:2:2 episode schedule."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pace_clm.rl_mixed import episode_balanced_schedule,validate_spec

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2'
SOURCE=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def dump(p,x):p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in x))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();out=a.output.resolve();data=out/'data'
    if data.exists():
        audit=read(data/'audit.json')
        for name,digest in audit['prepared_hashes'].items():
            if sha(data/name)!=digest:raise ValueError('prepared data changed: '+name)
        print(json.dumps(audit['counts']));return
    source=SOURCE/'data';old=read(source/'audit.json')
    if sha(source/'episodes.jsonl')!=old['manifest_sha256']:raise ValueError('source pool hash changed')
    if sha(source/'evaluation-manifest.jsonl')!=old['evaluation_sha256']:raise ValueError('evaluation manifest changed')
    episodes=rows(source/'episodes.jsonl');schedule=episode_balanced_schedule(episodes)
    for s in episodes:validate_spec(s)
    cohorts=read(source/'cohorts.json');cohorts['gate']=[episodes[i]['span_id'] for i in schedule[0]['episode_indices']]
    train_ids={s['span_id'] for s in episodes}
    for name in ('web_heldout','ego_heldout','ours_heldout'):
        if set(cohorts[name])&train_ids:raise ValueError('heldout enters training pool')
    sampled={episodes[i]['span_id'] for b in schedule for i in b['episode_indices']}
    for d in ('web','ego'):
        cohorts[d+'_all']=cohorts[d+'_train']+cohorts[d+'_heldout']
        cohorts[d+'_sampled']=[q for q in cohorts[d+'_train'] if q in sampled]
        cohorts[d+'_unsampled']=[q for q in cohorts[d+'_train'] if q not in sampled]
    specs={s['span_id']:s for s in rows(source/'evaluation-specs.jsonl')}
    specs.update({s['span_id']:s for s in episodes if s['span_id'] in cohorts['gate']})
    wanted=sorted({q for values in cohorts.values() for q in values});evaluation=[specs[q] for q in wanted]
    manifests=[{'question_id':s['span_id'],'video_id':s['video_id'],'question':s['question'],
        'resolved_video_path':s['video_path'],'inference_source_indices':s['source_indices'],
        'inference_timestamps':s['timestamps']} for s in evaluation]
    counts={'updates':len(schedule),'scheduled_decisions':schedule[-1]['exposure_after'],
        'domain_decisions':schedule[-1]['cumulative_domain_decisions'],
        'sampled_episodes':{d:sum(s['domain']==d and s['span_id'] in sampled for s in episodes) for d in ('ours','web','ego')},
        'cohorts':{k:len(v) for k,v in cohorts.items()},'unique_eval_episodes':len(evaluation)}
    if counts['updates']!=23 or counts['scheduled_decisions']!=98836 or counts['sampled_episodes']!={'ours':92,'web':46,'ego':46}:
        raise ValueError('schedule differs from approved CPU calculation: '+str(counts))
    data.mkdir(parents=True)
    for name in ('episodes.jsonl','video-hashes.jsonl'):shutil.copyfile(source/name,data/name)
    write(data/'cohorts.json',cohorts);write(data/'fast-cohorts.json',read(source/'fast-cohorts.json'))
    dump(data/'evaluation-specs.jsonl',evaluation);dump(data/'evaluation-manifest.jsonl',manifests)
    write(data/'schedule.json',schedule)
    write(data/'sampled-identities.json',sorted(sampled))
    audit={**old,'sampling_unit':'episode','version':'mixed-web-ego-episode-balanced-v2','counts':counts,
        'source_run':str(SOURCE),'source_audit_sha256':sha(source/'audit.json'),
        'mixture':'50/25/25 episodes; exact 4:2:2 per update, without replacement; four rollouts per episode',
        'manifest_sha256':sha(data/'episodes.jsonl'),'evaluation_sha256':sha(data/'evaluation-manifest.jsonl'),
        'prepared_hashes':{f.name:sha(f) for f in data.iterdir() if f.is_file()}}
    write(data/'audit.json',audit);print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
