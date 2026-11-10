#!/usr/bin/env python3
"""Immutable byte-offset mixture plans: no row rewriting, sampling, or deduplication."""
from __future__ import annotations
import argparse
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from prepare_qwen35_control import exclusions, aliases as common_aliases

LABELS = ('Silence', 'Response', 'Hold')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def aliases(value):
    return common_aliases(value)


def identities(row):
    result = set()
    for key in ('video_id', 'video', 'video_path', 'resolved_video_path', 'question_id'):
        if row.get(key) is not None:
            result.update(aliases(row[key]))
    return result


def quota(counts, consumed, end):
    target = counts * (end / int(counts.sum())) - consumed
    take = np.maximum(0, np.floor(target).astype(np.int64))
    take = np.minimum(take, counts - consumed)
    while int(take.sum()) < end - int(consumed.sum()):
        available = take < counts - consumed
        k = int(np.argmax(np.where(available, target - take, -np.inf)))
        take[k] += 1
    return take


def make_order(labels, lengths, seed=42):
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=3)
    queues = []
    for k in range(3):
        ids = np.flatnonzero(labels == k)
        # Shuffle short local groups rather than individual frames for cache reuse.
        groups = [ids[i:i+8] for i in range(0, len(ids), 8)]
        rng.shuffle(groups)
        queues.append(np.concatenate(groups) if groups else np.array([], dtype=np.int64))
    consumed = np.zeros(3, dtype=np.int64)
    order = np.empty(len(labels), dtype=np.int64)
    for start in range(0, len(labels), 4096):
        end = min(start + 4096, len(labels))
        window_quota = quota(counts, consumed, end)
        local = []
        for k in range(3):
            ids = queues[k][consumed[k]:consumed[k]+window_quota[k]]
            local.append(ids[np.argsort(lengths[ids], kind='stable')])
        cursors = np.zeros(3, dtype=np.int64)
        for batch_start in range(start, end, 128):
            batch_end = min(batch_start+128, end)
            take = quota(counts, consumed, batch_end)
            # Ensure window selection remains exact at the last updates.
            remaining = window_quota - cursors
            take = np.minimum(take, remaining)
            while take.sum() < batch_end-batch_start:
                k = int(np.argmax(np.where(take < remaining,
                    counts * (batch_end / len(labels)) - consumed - take, -np.inf)))
                take[k] += 1
            ids = np.concatenate([local[k][cursors[k]:cursors[k]+take[k]] for k in range(3)])
            ids = ids[np.argsort(lengths[ids], kind='stable')]
            # Round-robin striding spreads the sorted lengths across all eight ranks.
            # Rotate strided rank ownership so tie ordering cannot assign the
            # few Hold rows to the same ranks on every update.
            order[batch_start:batch_end] = np.roll(ids, (batch_start//128) % 8)
            consumed += take
            cursors += take
        if not np.array_equal(cursors, window_quota):
            raise ValueError('window quota mismatch')
    if not np.array_equal(np.sort(order), np.arange(len(labels))):
        raise ValueError('row multiset changed')
    return order


def audit(labels, order):
    ordered = labels[order]
    target = np.bincount(labels, minlength=3) / len(labels)
    cumulative = np.zeros(3, dtype=np.int64)
    max_window = np.zeros(3)
    max_prefix = np.zeros(3)
    update_min = np.full(3, 128)
    update_max = np.zeros(3, dtype=np.int64)
    max_absence = np.zeros(3, dtype=np.int64)
    absence = np.zeros(3, dtype=np.int64)
    for start in range(0, len(labels), 4096):
        values = ordered[start:start+4096]
        counts = np.bincount(values, minlength=3)
        cumulative += counts
        # Short terminal window is subject to integer rounding.
        tolerance = np.array([.01,.01,.005]) + 1/len(values)
        deviation = np.abs(counts / len(values) - target)
        if np.any(deviation > tolerance):
            raise ValueError(f'window proportions failed at {start}: {deviation}')
        max_window = np.maximum(max_window, deviation)
        max_prefix = np.maximum(max_prefix, np.abs(cumulative/cumulative.sum()-target))
    for start in range(0, len(labels)-127, 128):
        counts = np.bincount(ordered[start:start+128], minlength=3)
        update_min = np.minimum(update_min, counts)
        update_max = np.maximum(update_max, counts)
        absence = np.where(counts == 0, absence+1, 0)
        max_absence = np.maximum(max_absence, absence)
    return dict(target=dict(zip(LABELS,target.tolist())),
        maximum_window_absolute_deviation=max_window.tolist(),
        maximum_window_prefix_absolute_deviation=max_prefix.tolist(),
        per_update_min=update_min.tolist(), per_update_max=update_max.tolist(),
        longest_updates_without_class=max_absence.tolist(), exact_permutation_verified=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'audit.json').exists():
        raise FileExistsError('frozen plan already exists')
    excluded, provenance = exclusions()
    exclusion_hashes = {item['path']: item['sha256'] for item in provenance}
    offsets, sizes, labels, lengths = array('Q'), array('I'), array('B'), array('I')
    sources, memories, videos = Counter(), Counter(), set()
    excluded_rows = 0
    offset = 0
    source_hash = hashlib.sha256()
    with a.source.open('rb') as f:
        for number, line in enumerate(f):
            source_hash.update(line)
            row = json.loads(line)
            if identities(row) & excluded:
                excluded_rows += 1
            else:
                if len(row['turns']) > 32 or any(int(t['frame_index']) > int(row['frame_index']) for t in row['turns']):
                    raise ValueError('noncausal context')
                offsets.append(offset); sizes.append(len(line)); labels.append(LABELS.index(row['label']))
                lengths.append(144*len(row['turns']) + len(row['question'])//4)
                sources[Path(row['video_path']).parts[-4] if len(Path(row['video_path']).parts)>4 else 'unknown'] += 1
                memories[bool(row.get('memory'))] += 1
                videos.add(row['video_id'])
            offset += len(line)
            if number % 500000 == 0:
                print(json.dumps({'scanned': number, 'eligible':len(labels)}), flush=True)
    labels = np.asarray(labels); lengths = np.asarray(lengths)
    order = make_order(labels, lengths)
    report = audit(labels, order)
    index = np.column_stack((np.asarray(offsets), np.asarray(sizes)))
    np.save(a.output/'representative.npy', index[order])
    # Existing bounded sorting policy, over the common eligible source order.
    control = np.arange(len(labels))
    for start in range(0,len(labels),8320):
        ids = control[start:start+8320].copy()
        control[start:start+8320] = ids[np.argsort(lengths[ids],kind='stable')]
    np.save(a.output/'control.npy', index[control])
    # Both byte-offset plans are exact permutations of the common eligible pool.
    assert np.array_equal(np.sort(control),np.sort(order))
    np.save(a.output/'labels.npy',labels[order])
    np.save(a.output/'lengths.npy',lengths[order])
    report.update(source=str(a.source.resolve()), source_sha256=source_hash.hexdigest(),
        representative_sha256=digest(a.output/'representative.npy'), control_sha256=digest(a.output/'control.npy'),
        eligible_rows=len(labels), excluded_rows=excluded_rows, unique_videos=len(videos),
        exclusion_files=exclusion_hashes, source_counts=dict(sources), memory_counts=dict(memories),
        causal_history_checked=True, row_contents_preserved='original immutable byte ranges',
        full_a_b_row_multiset_equal=True, unused_terminal_rows=len(labels)%128,
        control_first_4096_counts=np.bincount(labels[control[:4096]],minlength=3).tolist())
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
