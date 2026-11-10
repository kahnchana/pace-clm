#!/usr/bin/env python3
"""Freeze training-only spans from the existing SFT video cohort and original labels."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent/'stream-gemma'
sys.path[:0] = [str(ROOT/'src'), str(SG/'src'), str(SG)]

from prepare_jevany_decisions import make_args, sha256
from prepare_qwen35_control import aliases, exclusions
from stream_gemma.data import _conversation_and_range, _parse_record
from stream_gemma.grpo.span import SpanSamplerConfig, _candidate_specs


@lru_cache(maxsize=None)
def resolved(path):
    return str(Path(path).resolve())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--states', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output/'spans.jsonl'
    if target.exists():
        raise FileExistsError(target)
    source = ROOT/'outputs/jevany-j1s/manifests/train.jsonl'
    cohort, cohort_paths = set(), set()
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            cohort_paths.add(row['video_path'])
            cohort.add(resolved(row['video_path']))
    cohort_names = {Path(path).name for path in cohort_paths}
    print(json.dumps({'stage': 'cohort_loaded', 'videos': len(cohort)}), flush=True)
    denied, provenance = exclusions()
    metadata = json.loads((SG/'datasets/markii/metadata.json').read_text())
    args = make_args()
    config = SpanSamplerConfig(num_states=a.states, min_span_frames=32,
                               max_span_frames=32, candidates_per_video_per_bucket=16,
                               seed=a.seed)
    rng = random.Random(a.seed)
    buckets = {0: [], 1: [], 2: []}
    removed, records = 0, 0
    with (SG/'datasets/markii/train.jsonl').open() as handle:
        for line in handle:
            relative, question, text = _parse_record(json.loads(line))
            meta = metadata.get(str(Path(relative).with_suffix('')))
            if not meta:
                continue
            path = meta.get('path') or str(SG/'datasets/markii/videos'/relative)
            if path not in cohort_paths and Path(path).name not in cohort_names:
                continue
            if resolved(path) not in cohort:
                continue
            if aliases(path) & denied:
                removed += 1
                continue
            built = _conversation_and_range(question=question, text_stream=text,
                                            metadata=meta, source_fps=1,
                                            keep_trailing_silence=True)
            if not built:
                continue
            conversation, source_range = built
            specs = _candidate_specs(question=question, video_path=path,
                                    conversation=conversation, source_range=source_range,
                                    args=args, config=config, rng=rng)
            # Check original event starts on both sides of each boundary.
            from stream_gemma.episode import flatten_stream_conversation, paint_v24_states
            from stream_gemma.data import _paint_kwargs
            painted = paint_v24_states(flatten_stream_conversation(conversation), **_paint_kwargs(args))
            starts = [source_range.start+e.start for e in painted.kept_events]
            for spec in specs:
                if any(t < spec.rollout_start < t+4 or t < spec.rollout_stop < t+4 for t in starts):
                    continue
                row = spec.to_manifest_record()
                # Include question in identity: one video may have several requests.
                row['span_id'] = hashlib.sha256(
                    f'{spec.span_id}:{question}'.encode()).hexdigest()[:24]
                buckets[min(2, len(spec.gold_points))].append(row)
            records += 1
            if records % 500 == 0:
                print(json.dumps({'source_records': records,
                                  'candidates': {k: len(v) for k,v in buckets.items()}}), flush=True)
    chosen, used = [], set()
    for bucket, fraction in ((1, .5), (2, .25), (0, .25)):
        rng.shuffle(buckets[bucket])
        take = round(a.states*fraction)
        unique = []
        for row in buckets[bucket]:
            if row['span_id'] not in used:
                used.add(row['span_id'])
                unique.append(row)
            if len(unique) == take:
                break
        if len(unique) < take:
            raise ValueError(f'insufficient spans in bucket {bucket}: {len(unique)}/{take}')
        chosen.extend(unique)
    rng.shuffle(chosen)
    target.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in chosen))
    report = {'states': len(chosen), 'candidate_buckets': {k: len(v) for k,v in buckets.items()},
              'selected_buckets': dict(Counter(min(2, len(r['gold_points'])) for r in chosen)),
              'cohort_videos': len(cohort), 'source_records': records,
              'excluded_records': removed, 'excluded_sources': provenance,
              'source_sha256': sha256(source), 'spans_sha256': sha256(target),
              'annotations_sha256': sha256(SG/'datasets/markii/train.jsonl'),
              'metadata_sha256': sha256(SG/'datasets/markii/metadata.json'), 'seed': a.seed}
    (a.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'excluded_sources'}), flush=True)


if __name__ == '__main__':
    main()
