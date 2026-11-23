#!/usr/bin/env python3
"""Prepare the explicitly authorized fresh, 50:50 PACE/ESTP-IT SFT run."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pace_clm.rl_estp import estp_spec
from pace_clm.sft_pace_estp import EXPLICIT_TASKS, action_plan, training_frames, supervised_row

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160'
IT = Path('/fsx/sfr/data/multimodal/online_video_language/ESTP-IT/ESTP_IT')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def dump(path, values):
    with Path(path).open('w') as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(run):
    data = run/'data'; data.mkdir(parents=True, exist_ok=True)
    if (run/'prepared.json').exists():
        saved = read(run/'prepared.json')
        for path, value in saved['hashes'].items():
            if sha(path) != value:
                raise ValueError('prepared artifact changed: '+path)
        return saved
    splits = read(PARENT/'data/estp-video-splits.json')
    assert all(not set(splits[a]) & set(splits[b]) for a, b in [('train','dev'),('train','test'),('dev','test')])
    write(data/'estp-video-splits.json', splits)
    source = read(IT/'estp.json')
    rejected = Counter(); candidates = {'train': [], 'dev': []}
    for split in candidates:
        for video in splits[split]:
            for clip, questions in source[video].items():
                for index, qa in enumerate(questions):
                    try:
                        if any(t.get('role')=='user' for t in qa.get('conversation', [])):
                            raise ValueError('dynamic_question_requires_separate_protocol')
                        spec = estp_spec(video, clip, index, qa, IT/'full_scale_2fps_max384'/f'{video}.mp4',
                                         training=split == 'train')
                        spec['data_source'] = 'estp_it'
                        spec['span_id'] = f'estp-it:{video}/{clip}/{index}'
                        if split == 'dev' and spec['task_type'] not in EXPLICIT_TASKS:
                            continue
                        if split == 'train':
                            action_plan(spec)
                        candidates[split].append(spec)
                    except (ValueError, KeyError, TypeError, OverflowError) as exc:
                        rejected[f'{split}:{exc}'] += 1
        print(json.dumps({'split': split, 'candidates': len(candidates[split])}), flush=True)
    paths = sorted({s['video_path'] for values in candidates.values() for s in values})
    def video_metadata(path):
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,r_frame_rate','-of','json',path],
            text=True,capture_output=True,timeout=30,check=True)
        stream=json.loads(result.stdout)['streams'][0]
        numerator,denominator=map(float,stream['r_frame_rate'].split('/'))
        return int(stream['nb_frames']),numerator/denominator
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = {}
        for index,(path,value) in enumerate(zip(paths,pool.map(video_metadata,paths))):
            metadata[path]=value
            if index%100==0:print(json.dumps({'media_checked':index+1,'media_total':len(paths)}),flush=True)
    write(data/'media-metadata.json', metadata)
    for split in candidates:
        accepted = []
        for spec in candidates[split]:
            count, fps = metadata[spec['video_path']]
            if abs(fps-2) > .01 or spec['source_indices'][-1] >= count:
                rejected[f'{split}:physical_media_bounds'] += 1
                continue
            accepted.append(spec)
        candidates[split] = accepted
    train = candidates['train']; pairs = []; counts = Counter()
    for index, spec in enumerate(train):
        labels, _ = action_plan(spec)
        for frame in training_frames(spec):
            pairs.append((index, frame)); counts[labels[frame]] += 1
    if not pairs or not candidates['dev']:
        raise ValueError('empty ESTP source')
    dump(data/'estp-train.jsonl', train)
    np.save(data/'estp-training-pairs.npy', np.asarray(pairs, dtype=np.int32))
    pace = ROOT/'outputs/jevany-j1s/manifests/train.jsonl'
    offsets = []; pace_counts = Counter(); pace_videos = set()
    with pace.open('rb') as handle:
        while True:
            offset = handle.tell(); line = handle.readline()
            if not line: break
            row = json.loads(line)
            if row['split'] != 'train': raise ValueError('PACE heldout in training')
            if not Path(row['video_path']).is_file(): raise FileNotFoundError(row['video_path'])
            offsets.append((offset,len(line))); pace_counts[row['label']] += 1; pace_videos.add(row['video_id'])
    np.save(data/'pace-offsets.npy', np.asarray(offsets, dtype=np.int64))
    existing = [json.loads(line) for line in (PARENT/'data/evaluation-specs.jsonl').read_text().splitlines()]
    cohorts_old = read(PARENT/'data/cohorts.json')
    cohorts = {k: cohorts_old[k] for k in ('web_all','web_train','web_heldout','ego_all','ego_train','ego_heldout')}
    wanted = set(cohorts['web_all']+cohorts['ego_all'])
    evaluation = [s for s in existing if s['span_id'] in wanted]+candidates['dev']
    cohorts['estp_it_explicit_dev'] = [s['span_id'] for s in candidates['dev']]
    assert len(wanted) == 826 and len(evaluation) == len(wanted)+len(candidates['dev'])
    dump(data/'evaluation-specs.jsonl', evaluation)
    dump(data/'evaluation-manifest.jsonl', [dict(question_id=s['span_id'],video_id=s['video_id'],question=s['question'],
        resolved_video_path=s['video_path'],inference_source_indices=s['source_indices'],
        inference_timestamps=s['timestamps']) for s in evaluation])
    write(data/'cohorts.json', cohorts)
    # Fixed, task-stratified sampled readiness probe, eight episodes at each milestone.
    probe = []
    for task in EXPLICIT_TASKS:
        eligible = [s for s in candidates['dev'] if s['task_type']==task]
        reachable=[]
        for spec in eligible:
            try: action_plan(spec)
            except ValueError: continue
            reachable.append(spec)
        eligible=reachable
        if eligible:
            probe.append(min(eligible, key=lambda s: hashlib.sha256(s['span_id'].encode()).hexdigest()))
    dump(data/'readiness-specs.jsonl', probe)
    dump(data/'label-audit-examples.jsonl', [supervised_row(train[i], frame) for i,frame in pairs[:64]])
    counts_out = dict(pace_rows=len(offsets),pace_labels=dict(pace_counts),pace_videos=len(pace_videos),
        estp_training_episodes=len(train),estp_training_videos=len({s['video_id'] for s in train}),
        estp_training_rows=len(pairs),estp_labels=dict(counts),
        estp_explicit_dev_episodes=len(candidates['dev']),estp_dev_tasks=dict(Counter(s['task_type'] for s in candidates['dev'])),
        rejected=dict(rejected))
    write(data/'audit.json', dict(counts=counts_out,pace_manifest=str(pace),
        label_policy='first native-window sampled frame Response; rest of that window Hold; outside Silence; reject overlaps/unreachable training windows',
        training_sampling='all ESTP Response targets; at most one Hold and four Silence per Response; PACE native rows unchanged; 50:50 presentations',
        history='teacher-forced causal past actions/answers only; current answer hidden; at most 32 frames',
        evaluation='full WEB 500 and EGO 326; all eligible explicit IT dev SQ cases; native windows; no ESTP-Bench or IT test'))
    result = dict(counts=counts_out,hashes={str(p):sha(p) for p in data.iterdir() if p.is_file()},
                  source_hashes={str(p):sha(p) for p in [pace,IT/'estp.json',PARENT/'data/estp-video-splits.json']})
    write(run/'prepared.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().run.resolve())['counts'],indent=2),flush=True)
