#!/usr/bin/env python3
"""Freeze a clean anchor and matched fresh/replay data before training."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import random

from prepare_qwen35_control import aliases, exclusions
from qwen35_sft_campaign import ROOT, rows, sha, stratum, write


def prepare_anchor(out):
    source = ROOT / 'outputs/jevany-j1s/manifests/train.jsonl'
    expected = '89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222'
    if sha(source) != expected:
        raise ValueError('historical anchor changed')
    denied, provenance = exclusions()
    target = out / 'data/anchor.jsonl'
    if target.exists():
        audit = json.loads((out / 'data/anchor_audit.json').read_text())
        if sha(target) != audit['sha256']:
            raise ValueError('anchor changed after freezing')
        return audit
    target.parent.mkdir(parents=True, exist_ok=True)
    kept, removed, tail = [], Counter(), []
    for line in source.open('rb'):
        row = json.loads(line)
        if (aliases(row['video_id']) | aliases(row['video_path'])) & denied:
            removed[row['label']] += 1
        else:
            kept.append(line)
    # Every rank must consume the same count, including the final partial update.
    count = len(kept) // 8 * 8
    tail = [json.loads(line) for line in kept[count:]]
    target.write_bytes(b''.join(kept[:count]))
    counts = Counter(json.loads(line)['label'] for line in kept[:count])
    audit = dict(source=str(source), source_sha256=expected, sha256=sha(target),
                 presentations=count, labels=dict(counts), excluded_labels=dict(removed),
                 alignment_reserved_rows=tail, exclusion_sources=provenance,
                 historical_order_preserved=True,
                 historical_reproduction_limit='Known validation and other-benchmark overlaps removed; clean-anchor batch comparison, not an exact historical reproduction.')
    write(out / 'data/anchor_audit.json', audit, frozen=True)
    print(json.dumps(dict(stage='anchor_frozen', rows=count, removed=dict(removed))), flush=True)
    return audit


def prepare_expansion(out):
    anchor = list(rows(out / 'data/anchor.jsonl'))
    if (out / 'data/expansion_audit.json').exists():
        audit = json.loads((out / 'data/expansion_audit.json').read_text())
        for name in ('fresh', 'replay'):
            if sha(out / f'data/{name}.jsonl') != audit[name + '_sha256']:
                raise ValueError('expansion data changed')
        return audit
    source = ROOT / 'outputs/jevany-qwen35-2b-mixture-12h/shared/manifests/eligible.jsonl'
    source_audit = json.loads((source.parent / 'audit.json').read_text())
    expected = source_audit['eligible_sha256']
    denied, _ = exclusions()
    anchor_videos = set().union(*(aliases(r['video_id']) | aliases(r['video_path']) for r in anchor))
    blocked = anchor_videos | denied
    quotas = Counter(stratum(r) for r in anchor)
    label_quotas = Counter(r['label'] for r in anchor)
    heaps, fallback = defaultdict(list), defaultdict(list)
    seen = set()
    digest = hashlib.sha256()
    counts = Counter()

    def admit(heap, item, quota):
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    with source.open('rb') as handle:
        index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            row = json.loads(line)
            ids = aliases(row['video_id']) | aliases(row['video_path'])
            if not ids & blocked:
                identity = hashlib.sha256(json.dumps([row['video_path'], row['question'], row['frame_index']]).encode()).digest()
                if identity not in seen:
                    seen.add(identity)
                    priority = -int.from_bytes(hashlib.sha256(b'sft-campaign-42:' + identity).digest()[:8], 'big')
                    item = (priority, offset, len(line), stratum(row))
                    key = stratum(row)
                    if quotas[key]:
                        admit(heaps[key], item, quotas[key])
                    admit(fallback[row['label']], item, label_quotas[row['label']])
                    counts[row['label']] += 1
            index += 1
            if index % 500000 == 0:
                print(json.dumps(dict(stage='selecting_fresh', scanned=index, eligible_unique=dict(counts))), flush=True)
    if digest.hexdigest() != expected:
        raise ValueError('eligible source hash changed')
    chosen = {key: sorted(heap, reverse=True) for key, heap in heaps.items()}
    selected_offsets = {item[1] for heap in chosen.values() for item in heap}
    remaining = {label: [item for item in sorted(heap, reverse=True) if item[1] not in selected_offsets]
                 for label, heap in fallback.items()}
    replay_buckets = defaultdict(list)
    for row in anchor:
        replay_buckets[stratum(row)].append(row)
    rng = random.Random(42)
    for bucket in replay_buckets.values():
        rng.shuffle(bucket)
    relaxations = Counter()
    with source.open('rb') as handle, (out / 'data/fresh.jsonl').open('wb') as fresh, (out / 'data/replay.jsonl').open('w') as replay:
        for row in anchor:
            key = stratum(row)
            if chosen.get(key):
                item = chosen[key].pop()
            else:
                if not remaining[row['label']]:
                    raise ValueError(f'insufficient new positions for {key}')
                item = remaining[row['label']].pop()
                relaxations[str(key)] += 1
            handle.seek(item[1])
            line = handle.read(item[2])
            candidate = json.loads(line)
            if candidate['label'] != row['label']:
                raise ValueError('fresh/replay label mismatch')
            fresh.write(line)
            replay.write(json.dumps(replay_buckets[key].pop(), sort_keys=True) + '\n')
    report = dict(presentations=len(anchor), fresh_sha256=sha(out / 'data/fresh.jsonl'),
                  replay_sha256=sha(out / 'data/replay.jsonl'), source=str(source), source_sha256=expected,
                  unique_candidate_positions=sum(counts.values()), eligible_counts=dict(counts),
                  fresh_anchor_video_overlap=0, fresh_duplicate_positions=0,
                  exact_per_update_label_matching=True, stratum_relaxations=dict(relaxations),
                  policy='Seeded position-hash selection; match label/context/memory/source slots; relax only within label if a stratum is exhausted.')
    write(out / 'data/expansion_audit.json', report, frozen=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expansion', action='store_true')
    args = parser.parse_args()
    prepare_anchor(args.output)
    if args.expansion:
        prepare_expansion(args.output)


if __name__ == '__main__':
    main()
