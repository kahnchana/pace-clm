#!/usr/bin/env python3
"""Freeze native WEB/EGO splits and the 50/25/25 decision mixture."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path

from prepare_jevany_episode_rl import episode
from prepare_jevany_rl import ROOT,SG,resolved
from prepare_qwen35_control import aliases,exclusions,identities
from stream_gemma.data import _parse_record
from stream_gemma.inference.runtime import video_metadata
from zs_infer.adapters.web import load_examples
from pace_clm.rl_mixed import mixture_schedule,validate_spec
from jevany_rl_reward import write,sha

DEFAULT=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1'
OLD=ROOT/'outputs/jevany-2b-rl/episode-rloo-v2'


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path,values):path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))


def priority(value):return hashlib.sha256(('mixed-42:'+str(value)).encode()).hexdigest()


def ours_spec(row,domain='ours'):
    row=copy.deepcopy(row);n=row['rollout_stop'];row['domain']=domain
    row['source_indices']=list(range(n));row['timestamps']=[float(i) for i in range(n)]
    row['gold_answers']=[{'content':g['content'],'reply_timespan':[g['time_seconds'],g['time_seconds']+4],
                         'related_timespan':[g['time_seconds'],g['time_seconds']+4]} for g in row['gold_points']]
    return row


def native_spec(domain,example,annotation):
    fps={'web':2,'ego':6}[domain];count,measured=video_metadata(example['path'])
    indices=list(range((count-1)%fps,count,fps));times=[i/fps for i in indices]
    if not indices:raise ValueError('empty native video')
    qid=domain+':'+str(annotation['question_id'])
    gold=copy.deepcopy(annotation['answer'])
    result={'span_id':qid,'native_question_id':annotation['question_id'],'domain':domain,
            'video_id':annotation['video'],'video_path':resolved(example['path']),'question':example['question'],
            'decode_start':0,'rollout_start':0,'rollout_stop':len(indices),'rollout_frames':len(indices),
            'prefix_turns':[],'initial_memory':None,'source_indices':indices,'timestamps':times,
            'gold_answers':gold,'gold_points':[{'time_seconds':g['reply_timespan'][0],'content':g['content']} for g in gold],
            'source_fps':fps,'measured_fps':measured,'source_frame_count':count}
    validate_spec(result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve();data=out/'data';data.mkdir(parents=True,exist_ok=True)
    if ((data/'audit.json').exists() and (data/'video-hashes.jsonl').exists()
            and 'gate' in json.loads((data/'cohorts.json').read_text())):
        print(json.dumps(json.loads((data/'audit.json').read_text())['counts']));return
    ours=[ours_spec(r) for r in rows(OLD/'data/episodes.jsonl')]
    _,provenance=exclusions();protected=set()
    for item in provenance:
        path=Path(item['path'])
        if path in (SG/'datasets/proactive-videoqa-web/anno.json',SG/'datasets/proactive-videoqa-ego/anno.json'):continue
        obj=rows(path) if path.suffix=='.jsonl' else json.loads(path.read_text())
        for identity in identities(obj):protected.update(aliases(identity))
    cohorts={};train=ours.copy();native=[];native_annotations={}
    for domain in ('web','ego'):
        source=SG/f'datasets/proactive-videoqa-{domain}/anno.json'
        examples,annotations=load_examples(source,SG/f'datasets/proactive-videoqa-{domain}/videos')
        if len(examples)!=len(annotations):raise ValueError('unresolved native videos')
        by_id={r['question_id']:r for r in annotations}
        specs=[native_spec(domain,e,by_id[e['question_id']]) for e in examples]
        groups={min(aliases(r['video_id'])) for r in specs}
        forced={min(aliases(r['video_id'])) for r in specs if aliases(r['video_id'])&protected}
        available=sorted(groups-forced,key=priority)
        n=min(round(.8*len(groups)),len(available));training_groups=set(available[:n])
        seen=[r for r in specs if min(aliases(r['video_id'])) in training_groups]
        hold=[r for r in specs if min(aliases(r['video_id'])) not in training_groups]
        if not seen or not hold:raise ValueError('empty native split')
        cohorts[domain+'_train']=[r['span_id'] for r in seen]
        cohorts[domain+'_heldout']=[r['span_id'] for r in hold]
        train.extend(seen);native.extend(specs);native_annotations[domain]=sha(source)
    # Reconstruct complete trusted intervals from the registered validation pool.
    allowed={resolved(r['resolved_video_path']) for r in rows(ROOT/'outputs/jevany-j1sr/validation/frozen.jsonl')}
    names={Path(p).name for p in allowed};metadata=json.loads((SG/'datasets/markii/metadata.json').read_text())
    val={};rejected=Counter();target_alias=set()
    for s in train:
        if s['domain'] in ('web','ego'):target_alias.update(aliases(s['video_path']))
    for line in (SG/'datasets/markii/val.jsonl').open():
        relative,question,text=_parse_record(json.loads(line));meta=metadata.get(str(Path(relative).with_suffix('')))
        if not meta:continue
        path=meta.get('path') or str(SG/'datasets/markii/videos'/relative)
        if Path(path).name not in names or resolved(path) not in allowed:continue
        if aliases(path)&target_alias:rejected['native_overlap']+=1;continue
        try:spec=ours_spec(episode(question,text,meta,path),'ours')
        except ValueError as exc:rejected[str(exc)]+=1;continue
        if spec['span_id'] in val and val[spec['span_id']]!=spec:raise ValueError('conflicting validation request')
        val[spec['span_id']]=spec
    eligible=sorted(val.values(),key=lambda r:priority(r['span_id']))
    if len(eligible)<128:raise ValueError(f'only {len(eligible)} trusted validation episodes available; rejected={dict(rejected)}')
    validation=eligible[:128]
    ours_paths={r['video_path'] for r in ours}
    if any(r['video_path'] in ours_paths for r in validation):raise ValueError('ours train/validation leakage')
    cohorts['ours_heldout']=[r['span_id'] for r in validation]
    cohorts['ours_train']=[r['span_id'] for r in sorted(ours,key=lambda r:priority(r['span_id']))[:128]]
    perfect=json.loads((OLD/'perfect-episodes.json').read_text())['episodes']
    cohorts['perfect20']=[r['span_id'] for r in sorted(perfect,key=lambda r:(-r['responses'],r['step'],r['span_id']))[:20]]
    schedule=mixture_schedule(train)
    cohorts['gate']=[train[i]['span_id'] for i in schedule[0]['episode_indices']]
    all_specs={r['span_id']:r for r in [*ours,*validation,*native]}
    for s in all_specs.values():validate_spec(s)
    train_ids={r['span_id'] for r in train}
    for name in ('web_heldout','ego_heldout','ours_heldout'):
        if set(cohorts[name])&train_ids:raise ValueError('held-out episode enters training')
    eval_ids=sorted(set(q for ids in cohorts.values() for q in ids))
    evaluation=[all_specs[q] for q in eval_ids]
    manifests=[{'question_id':s['span_id'],'video_id':s['video_id'],'question':s['question'],
                'resolved_video_path':s['video_path'],'inference_source_indices':s['source_indices'],
                'inference_timestamps':s['timestamps']} for s in evaluation]
    fast={name:sorted(ids,key=priority)[:min(32,len(ids))] for name,ids in cohorts.items() if name not in ('perfect20','gate')}
    fast['perfect20']=cohorts['perfect20']
    dump(data/'episodes.jsonl',train);dump(data/'evaluation-specs.jsonl',evaluation);dump(data/'evaluation-manifest.jsonl',manifests)
    write(data/'cohorts.json',cohorts);write(data/'fast-cohorts.json',fast)
    write(data/'schedule.json',schedule)
    counts={'training':dict(Counter(r['domain'] for r in train)),
            'cohorts':{k:len(v) for k,v in cohorts.items()},'unique_eval_episodes':len(evaluation),
            'ours_validation_eligible':len(eligible),'ours_validation_rejections':dict(rejected),
            'scheduled_decisions':schedule[-1]['exposure_after'],'updates':len(schedule),
            'domain_decisions':schedule[-1]['cumulative_domain_decisions'],
            'max_episode_frames':{d:max(r['rollout_frames'] for r in train if r['domain']==d) for d in ('ours','web','ego')}}
    audit={'counts':counts,'seed':42,'mixture':'50/25/25 by sampled RL decisions; source losses normalized separately and weighted 50/25/25',
           'splits':'seeded source-video groups; protected non-WEB/EGO evaluation and validation identities forced into holdout',
           'native_gold':'all original answer objects and reply/related windows retained unchanged, including zero-start and overlapping windows',
           'native_frames':'canonical end-aligned stride: source FPS 2 WEB / 6 EGO to 1 FPS; timestamps are original indices / source FPS',
           'native_annotation_hashes':native_annotations,'other_exclusions':provenance,
           'source_manifest_sha256':sha(OLD/'data/episodes.jsonl'),
           'manifest_sha256':sha(data/'episodes.jsonl'),'evaluation_sha256':sha(data/'evaluation-manifest.jsonl')}
    write(data/'audit.json',audit)
    # Freeze content hashes of every media input, concurrently, with no GPU work.
    paths=sorted({s['video_path'] for s in [*train,*evaluation]})
    previous={r['path']:r['sha256'] for r in rows(data/'video-hashes.jsonl')} if (data/'video-hashes.jsonl').exists() else {}
    def one(path):return {'path':path,'sha256':previous.get(path) or sha(path)}
    with ThreadPoolExecutor(max_workers=8) as pool:dump(data/'video-hashes.jsonl',pool.map(one,paths))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
