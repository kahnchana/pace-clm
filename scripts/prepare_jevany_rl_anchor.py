#!/usr/bin/env python3
"""Freeze bounded, balanced SFT replay from the existing RL training-video pool."""
import argparse
from collections import Counter
import heapq
import hashlib
import json
from pathlib import Path
import random

from jevany_rl_reward import ROOT, sha, write
from prepare_qwen35_control import aliases, exclusions


def select(source, allowed_paths, denied, per_class, seed=42):
    heaps={key:[] for key in ('Silence','Hold','Response')}
    counts=Counter();eligible=0
    with source.open('rb') as handle:
        for index,line in enumerate(handle):
            row=json.loads(line)
            if row['video_path'] not in allowed_paths or aliases(row['video_path'])&denied:continue
            if row.get('split')!='train' or row['label'] not in heaps:raise ValueError('non-training SFT row')
            counts[row['label']]+=1;eligible+=1
            key=int.from_bytes(hashlib.sha256(str(seed).encode()+b':'+str(index).encode()+b':'+line).digest(),'big')
            item=(-key,index,line)
            heap=heaps[row['label']]
            if len(heap)<per_class:heapq.heappush(heap,item)
            elif key < -heap[0][0]:heapq.heapreplace(heap,item)
    if any(len(h)!=per_class for h in heaps.values()):raise ValueError('insufficient balanced training rows')
    buckets={k:[(i,json.loads(line)) for _,i,line in sorted(v,reverse=True)] for k,v in heaps.items()}
    return buckets,counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'outputs/jevany-j1s/manifests/train.jsonl'
    spans=ROOT/'outputs/jevany-2b-rl/data/spans.jsonl'
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'replay.jsonl').exists():raise FileExistsError('replay already frozen')
    allowed={json.loads(line)['video_path'] for line in spans.read_text().splitlines()}
    denied,provenance=exclusions()
    buckets,counts=select(source,allowed,denied,42*32)
    rows=[];ids=[]
    for step in range(32):
        batch=[item for k in buckets for item in buckets[k][step*42:(step+1)*42]]
        random.Random(42+step).shuffle(batch)
        ids.extend(i for i,_ in batch);rows.extend(row for _,row in batch)
    target=a.output/'replay.jsonl'
    target.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write(a.output/'selection.json',ids)
    write(a.output/'freeze.json',{'source':str(source),'source_sha256':sha(source),'spans_sha256':sha(spans),
          'replay_sha256':sha(target),'selection_sha256':sha(a.output/'selection.json'),
          'seed':42,'rounds':32,'per_round':126,'per_class_per_round':42,
          'eligible_counts':dict(counts),'selected_counts':dict(Counter(row['label'] for row in rows)),
          'training_only':True,'restricted_to_existing_rl_video_pool':True,'benchmark_overlap':0,
          'exclusion_sources':provenance,'selection':'lowest seeded SHA256 per label; globally shuffled each round',
          'calibration':'round-0 synchronized RL and unweighted CE gradient norms; target CE/RL=0.15; freeze coefficient thereafter',
          'inference':'raw argmax, response_margin=0, no score adjustments'})
    print(json.dumps({'rows':len(rows),'labels':dict(Counter(row['label'] for row in rows))}))


if __name__=='__main__':main()
