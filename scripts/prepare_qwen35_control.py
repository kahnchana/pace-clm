#!/usr/bin/env python3
"""Freeze eligible J1-S rows and retain the historical bounded ordering policy."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from bucket_jevany_manifest import estimated_tokens

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / 'stream-gemma'


def aliases(value):
    name = str(value).split('?')[0].rstrip('/').rsplit('/', 1)[-1]
    name = re.sub(r'\.(mp4|mkv|webm|avi)$', '', name, flags=re.I)
    result = {name}
    if name.startswith('v_'):
        result.add(name[2:])
    for item in tuple(result):
        # WEB clip suffix; Ego4D cropped timestamp suffix.
        result.add(re.sub(r'\.\d+$', '', item))
        match = re.match(r'^([0-9a-fA-F-]{36})(?:[-_]\d+.*)?$', item)
        if match:
            result.add(match[1])
    return result


def identities(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from identities(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in {'video', 'video_id', 'video_path', 'resolved_video_path', 'relative_video_path'} and isinstance(value, str):
                yield value
            elif re.fullmatch(r'[0-9a-fA-F-]{36}', key):
                yield key
            if isinstance(value, (list, dict)):
                yield from identities(value)


def exclusions():
    paths = [SG / 'datasets' / name for name in (
        'proactive-videoqa-web/anno.json', 'proactive-videoqa-ego/anno.json',
        'ovo/ovo_bench_new.json', 'streamingbench/questions_proactive.json',
        'estp/estp_bench_sq.json')]
    paths += [Path('/fsx/sfr/data/multimodal/online_video_language/ProactiveVideoQA') / name / 'anno.json'
              for name in ('TV', 'VAD')]
    paths += [ROOT / name for name in (
        'outputs/pace-clm-mixtures/validation/selection.jsonl',
        'outputs/pace-clm-mixtures/validation/final.jsonl',
        'outputs/pace-clm-gate/validation/selection.jsonl',
        'outputs/pace-clm-gate/validation/final.jsonl',
        'outputs/jevany-j1sr/validation/frozen.jsonl',
        'outputs/jevany-j1sr-v2/validation/frozen.jsonl',
        'outputs/jevany-j1sr-v3/validation/canary64.jsonl')]
    paths += sorted((SG / 'outputs/evals').glob('markii-paucf-200-*/**/subset_200_manifest.jsonl'))
    denied, records = set(), []
    for path in paths:
        data = path.read_bytes()
        obj = [json.loads(line) for line in data.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(data)
        ids = set(identities(obj))
        if not ids:
            raise ValueError(f'no video identities extracted: {path}')
        for value in ids:
            denied.update(aliases(value))
        records.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'video_identities': len(ids)})
    return denied, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT / 'outputs/jevany-qwen35-2b-full/manifests/j1s.jsonl')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ('eligible.jsonl', 'control.jsonl', 'audit.json'):
        if (a.output / name).exists():
            raise FileExistsError(a.output / name)
    denied, provenance = exclusions()
    (a.output / 'exclusions.json').write_text(json.dumps({'sources': provenance, 'aliases': sorted(denied)}, indent=2))
    labels, removed, sources, contexts, ranks = Counter(), Counter(), Counter(), Counter(), [Counter() for _ in range(8)]
    source_hash, eligible_hash, order_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    videos, positions = set(), set()
    windows, prefixes, buffer = [], [], []
    emitted = 0
    probe = []
    longest = []
    # Same historical policy: stable token-length sort within 416*20 source rows.
    def emit(out):
        nonlocal emitted, buffer, longest
        ranked = sorted(buffer, key=lambda x: estimated_tokens(x[1]))
        assert Counter(x[0] for x in ranked) == Counter(x[0] for x in buffer)
        for line, row in ranked:
            out.write(line)
            order_hash.update(line)
            ranks[emitted % 8][row['label']] += 1
            emitted += 1
        if len(probe) < 128 * 24:
            # Contiguous batches at distributed source locations, selected by window number.
            if len(windows) % 50 == 0:
                for start in (0, max(0, len(ranked)//2//128*128), max(0, len(ranked)//128*128-128)):
                    probe.extend(line for line, _ in ranked[start:start+128])
        longest = sorted(longest + ranked[-128:], key=lambda x: estimated_tokens(x[1]))[-128:]
        counts = Counter(row['label'] for _, row in ranked)
        updates = [Counter(row['label'] for _, row in ranked[i:i+128]) for i in range(0, len(ranked), 128)]
        windows.append({'end_row': emitted, 'labels': dict(counts), 'per_update_ranges': {k: [min(c[k] for c in updates), max(c[k] for c in updates)] for k in ('Hold','Response','Silence')}})
        prefixes.append({'rows': emitted, 'labels': dict(labels)})
        buffer = []
    with a.source.open('rb') as inp, (a.output / 'eligible.jsonl').open('wb') as pool, (a.output / 'control.jsonl').open('wb') as out:
        for index, line in enumerate(inp):
            source_hash.update(line)
            row = json.loads(line)
            if (aliases(row['video_id']) | aliases(row['video_path'])) & denied:
                removed[row['label']] += 1
                continue
            if row['turns'][-1]['frame_index'] != row['frame_index'] or len(row['turns']) > 32:
                raise ValueError('noncausal or oversized input')
            pool.write(line)
            eligible_hash.update(line)
            labels[row['label']] += 1
            videos.add(row['video_id'])
            positions.add((row['video_id'], row['question'], row['frame_index']))
            sources[str(Path(row['video_path']).relative_to(SG / 'datasets/markii/videos')).split('/')[0]] += 1
            contexts[(len(row['turns']), bool(row.get('memory')))] += 1
            buffer.append((line,row))
            if len(buffer) == 8320:
                emit(out)
            if index and index % 500000 == 0:
                print(json.dumps({'scanned': index, 'eligible': sum(labels.values()), 'excluded': sum(removed.values())}), flush=True)
        if buffer:
            emit(out)
    expected_source = json.loads(a.source.with_suffix('.summary.json').read_text())['manifest_sha256']
    if source_hash.hexdigest() != expected_source:
        raise ValueError('source hash drift')
    (a.output / 'probe.jsonl').write_bytes(b''.join(probe[:128*24]))
    (a.output / 'stress.jsonl').write_bytes(b''.join(line for line, _ in longest))
    report = {'presentations': emitted, 'labels': dict(labels), 'class_proportions': {k:v/emitted for k,v in labels.items()},
              'excluded_labels': dict(removed), 'videos': len(videos), 'unique_positions': len(positions),
              'source_sha256': source_hash.hexdigest(), 'eligible_sha256': eligible_hash.hexdigest(), 'control_sha256': order_hash.hexdigest(),
              'order_policy': 'stable estimated-token sort in bounded 8320-row source/episode windows',
              'exact_row_conservation': 'verified raw-byte multiset equality in every bounded window; no added/dropped/modified eligible rows',
              'reserved_development_confirmation': 'existing frozen selection/final/video-disjoint validation sets excluded by aliases',
              'sources': dict(sources), 'contexts_and_memory': {str(k):v for k,v in contexts.items()},
              'rank_labels': list(map(dict,ranks)), 'windows': windows, 'prefixes': prefixes,
              'exclusions': provenance, 'probe_rows': min(len(probe),128*24), 'stress_rows': len(longest)}
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('windows','prefixes','exclusions','contexts_and_memory')}), flush=True)


if __name__ == '__main__':
    main()
