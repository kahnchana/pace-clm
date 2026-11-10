#!/usr/bin/env python3
"""Supplement immutable mixture plan audits with rank, padding and input checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/jevany-qwen35-2b-mixture-12h'
PLANS=BASE/'plans'
LABELS=('Silence','Response','Hold')


def main():
    audit=json.loads((PLANS/'audit.json').read_text())
    order=np.load(PLANS/'representative.npy',mmap_mode='r')
    labels=np.load(PLANS/'labels.npy',mmap_mode='r')
    lengths=np.load(PLANS/'lengths.npy',mmap_mode='r')
    batches=len(labels)//128
    rows=batches*128
    matrix=lengths[:rows].reshape(-1,128)
    padding=1-matrix.sum(axis=1)/(128*matrix.max(axis=1))
    rank_counts=[dict(zip(LABELS,np.bincount(labels[rank::8],minlength=3).tolist())) for rank in range(8)]
    # Check byte ranges against original rows at a deterministic dispersed sample,
    # and summarize a contiguous early prefix for exposure diversity.
    chosen=np.unique(np.concatenate((np.arange(min(32768,len(order))),
        np.random.default_rng(42).choice(len(order),8192,replace=False))))
    sources,contexts,memories,videos=Counter(),Counter(),Counter(),set()
    row_hash=hashlib.sha256()
    with Path(audit['source']).open('rb') as f:
        for index in chosen:
            offset,size=map(int,order[index]); f.seek(offset); line=f.read(size)
            row=json.loads(line)
            assert row['label']==LABELS[int(labels[index])]
            assert row['turns'][-1]['frame_index']==row['frame_index']
            assert len(row['turns'])<=32
            row_hash.update(line)
            if index<32768:
                sources[str(Path(row['video_path']).parent.parent)] += 1
                contexts[len(row['turns'])] += 1
                memories[bool(row.get('memory'))] += 1
                videos.add(row['video_id'])
    report=dict(exact_input_byte_ranges_checked=len(chosen),sample_sha256=row_hash.hexdigest(),
        full_rank_class_counts=rank_counts,
        estimated_padding_fraction_mean=float(padding.mean()),
        estimated_padding_fraction_p50_p95=np.quantile(padding,[.5,.95]).tolist(),
        prefix_rows=min(32768,len(order)),prefix_unique_videos=len(videos),
        prefix_context_frames=dict(contexts),prefix_memory=dict(memories),prefix_sources=dict(sources),
        padding_note='Proxy using 144 slots per frame plus question text; actual tokens may differ.')
    (PLANS/'exposure_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='prefix_sources'}))

if __name__=='__main__':
    main()
