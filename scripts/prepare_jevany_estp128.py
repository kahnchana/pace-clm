#!/usr/bin/env python3
"""Freeze the authorized step-32 to step-128 ESTP-IT continuation."""
import argparse
import copy
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pace_clm.rl_estp import continuation_schedule, estp_spec, priority, video_splits
from prepare_jevany_to32 import extend_replay_bank
from prepare_qwen35_control import aliases, exclusions
from stream_gemma.inference.runtime import video_metadata

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / 'stream-gemma'
PY = SG / '.venv/bin/python'
PARENT = ROOT / 'outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32'
DEFAULT = ROOT / 'outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128'
IT = Path('/fsx/sfr/data/multimodal/online_video_language/ESTP-IT/ESTP_IT')
BENCH = SG / 'datasets/estp'


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def dump(p, values):
    Path(p).write_text(''.join(json.dumps(v, sort_keys=True, allow_nan=False) + '\n' for v in values))


def trainer_command(run):
    plan = read(run / 'workflow-freeze.json')
    return list(map(str, [PY, run / 'code/run_jevany_estp_rloo.py', '--output', run,
        '--spans', run / 'data/episodes.jsonl', '--checkpoint', read(PARENT / 'freeze.json')['checkpoint'],
        '--anchor-manifest', run / 'anchor-extended/replay.jsonl', '--learners', '7', '--starts', '2',
        '--rollouts', '4', '--microbatch', '24', '--physical-batch', '24', '--collection-batch', '24',
        '--generator-batch', '8', '--frame-cache', '2048', '--decision-budget', plan['decision_budget'],
        '--reward-version', 'pauc-f-aligned-gpt-v1', '--full-run', '--sampling-unit', 'episode']))


def snapshot_trainer(run):
    source = (ROOT / 'scripts/run_jevany_mixed_rloo.py').read_text()
    source = source.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})')
    source = source.replace("    p.add_argument('--resume', action='store_true')",
                            "    p.add_argument('--resume', action='store_true')\n    p.add_argument('--freeze-only', action='store_true')")
    source = source.replace('a.decision_budget<=131072', 'a.decision_budget<=2000000')
    begin = source.index("    if socket.gethostname()!='ip-10-3-120-167':")
    end = source.index('    spans = ', begin)
    block = source[begin:end]
    block = block.replace('    a.output = a.output.resolve()\n    a.output.mkdir(parents=True, exist_ok=True)\n', '')
    source = source[:begin] + '    a.output = a.output.resolve()\n    a.output.mkdir(parents=True, exist_ok=True)\n    if not a.freeze_only:\n' + ''.join('    ' + line if line.strip() else line for line in block.splitlines(keepends=True)) + source[end:]
    source = source.replace("episode_batches=training_schedule(spans,sampling_unit=a.sampling_unit,budget=a.decision_budget,seed=a.seed)",
        f"from pace_clm.rl_estp import continuation_schedule\n    episode_batches=continuation_schedule(spans,json.loads(Path({str(PARENT / 'data/schedule.json')!r}).read_text()),seed=a.seed)")
    source = source.replace("('resume', 'updates','full_run')", "('resume', 'updates','full_run','freeze_only')")
    source = source.replace("freeze['domain_objective']='0.5 mean ours + 0.25 mean WEB + 0.25 mean EGO; constant 32 loss scale'",
        "freeze['domain_objective']='0.5 mean ours + 0.25 mean WEB + 0.25 mean EGO/ESTP-IT family; constant 32 loss scale'")
    source = source.replace("    freeze['hashes'] =", f"    files += [ROOT/'src/pace_clm/rl_estp.py', ROOT/'scripts/jevany_estp_control.py', Path({str(PARENT / 'data/schedule.json')!r})]\n    freeze['hashes'] =")
    source = source.replace("    if a.resume:\n        if json.loads(freeze_path.read_text())", "    if a.freeze_only:\n        if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:raise ValueError('existing freeze changed')\n        write_json(freeze_path,freeze)\n        print(json.dumps({'frozen':str(freeze_path),'updates':len(episode_batches)}),flush=True)\n        return\n    if a.resume:\n        if json.loads(freeze_path.read_text())")
    source = source.replace("from jevany_mixed_control import", "from jevany_estp_control import")
    source = source.replace("raise RuntimeError('near-silence: zero fires across all event-containing training trajectories')",
        "write_json(batch/'near-silence-warning.json', {'warning':'zero fires across event-containing trajectories','policy':'complete through step 128 irrespective of scores'})")
    source = source.replace("'domain_trajectory_counts':source_counts", "'domain_trajectory_counts':source_counts, 'data_source_trajectory_counts':dict(__import__('collections').Counter(t['spec'].get('data_source',t['spec']['domain']) for t in traces))")
    (run / 'code/run_jevany_estp_rloo.py').write_text(source)
    probe = (ROOT / 'scripts/probe_jevany_mixed_parity.py').read_text()
    probe = probe.replace("initial=Path(cfg['checkpoint'])", f"initial=Path({str(PARENT / 'step-000032')!r})")
    probe = probe.replace('batch-000000/scored.json', 'batch-000032/scored.json')
    (run / 'code/probe_estp_parity.py').write_text(probe)


def select_episodes(source, videos, limit, *, training):
    selected, task_counts, rejections = [], Counter(), Counter()
    queues = {}
    for video in videos:
        candidates = []
        for clip, questions in source[video].items():
            for index, question in enumerate(questions):
                try:
                    if any(t.get('role') == 'user' for t in question.get('conversation', [])):
                        raise ValueError('IT_dynamic_question_requires_separate_protocol')
                    spec = estp_spec(video, clip, index, question, IT / 'full_scale_2fps_max384' / f'{video}.mp4', training=training)
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    rejections[str(exc)] += 1
                    continue
                # IT heldouts use the same SQ protocol, without training reachability filtering.
                spec['data_source'] = 'estp_it'
                spec['span_id'] = f'estp-it:{video}/{clip}/{index}'
                candidates.append((clip, index, question, spec['task_type']))
        if candidates:
            queues[video] = candidates
    # One question per distinct video before selecting a second question.
    for round_index in range(100):
        for video, candidates in queues.items():
            if not candidates:
                continue
            chosen = min(candidates, key=lambda q: (task_counts[q[3]], priority(f'{video}/{q[0]}/{q[1]}')))
            candidates.remove(chosen)
            clip, index, question, task = chosen
            spec = estp_spec(video, clip, index, question, IT / 'full_scale_2fps_max384' / f'{video}.mp4', training=training)
            spec['data_source'] = 'estp_it'
            spec['span_id'] = f'estp-it:{video}/{clip}/{index}'
            selected.append(spec)
            task_counts[task] += 1
            if len(selected) == limit:
                return selected, dict(rejections), dict(task_counts), len(queues)
    raise ValueError('insufficient validated ESTP-IT episodes')


def prepare(run):
    if (run / 'prepared.json').exists():
        frozen = read(run / 'prepared.json')
        for path, expected in frozen['hashes'].items():
            if sha(path) != expected:
                raise ValueError('prepared artifact changed: ' + path)
        return frozen['counts']
    if run.exists():
        if (run/'freeze.json').exists() or (run/'latest.json').exists():
            raise FileExistsError('prepared training state already exists; inspect before retrying')
        archive=run.with_name(run.name+datetime.now(timezone.utc).strftime('.prepare-failed-%Y%m%dT%H%M%S'))
        run.rename(archive)
    if read(PARENT / 'latest.json')['step'] != 32 or read(PARENT / 'workflow-status.json')['state'] != 'complete':
        raise ValueError('parent step32 training incomplete')
    checkpoint = PARENT / 'step-000032'
    for name, digest in read(checkpoint / 'commit.json')['files'].items():
        if sha(checkpoint / name) != digest:
            raise ValueError('parent checkpoint hash mismatch: ' + name)
    run.mkdir(parents=True)
    (run / 'data').mkdir()
    (run / 'code').mkdir()
    (run / 'cache').mkdir()
    (run / 'queue').mkdir()
    (run / 'evaluation').mkdir()
    print('Verified step32 checkpoint hashes; inspecting ESTP annotations.', flush=True)
    source = read(IT / 'estp.json')
    denied, provenance = exclusions()
    protected = [v for v, clips in source.items() if any(aliases(value) & denied for value in [v, *clips])]
    splits = video_splits(set(source) - set(protected))
    write(run / 'data/estp-video-splits.json', splits)
    write(run / 'data/estp-exclusions.json', dict(protected_videos=protected, sources=provenance,
        policy='all clips/questions from a source video stay in one split; preserve existing protected benchmark/validation identities'))
    audit = {}
    new = []
    for name, size in [('train', 111), ('dev', 128), ('test', 128)]:
        chosen, rejected, tasks, eligible_videos = select_episodes(source, splits[name], size, training=name == 'train')
        audit[name] = dict(source_videos=len(splits[name]), selected_episodes=len(chosen), eligible_source_videos=eligible_videos,
            selected_source_videos=len({s['video_id'] for s in chosen}), rejected_records=rejected, selected_task_counts=tasks)
        dump(run / f'data/estp-{name}.jsonl', chosen)
        new += chosen
        print(name, json.dumps(audit[name]), flush=True)
    # Preserve every benchmark SQ case. Never filter evaluation by reachability.
    bench_source = read(BENCH / 'estp_bench_sq.json')
    bench = [estp_spec(v, c, i, q, BENCH / 'videos' / f'{v}.mp4', training=False, benchmark=True)
             for v, cs in bench_source.items() for c, qs in cs.items() for i, q in enumerate(qs)]
    if len(bench) != 1212:
        raise ValueError('expected all 1212 ESTP-Bench SQ cases')
    print('Checking physical media bounds.', flush=True)
    paths = sorted({s['video_path'] for s in [*new, *bench]})
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = dict(zip(paths, pool.map(lambda p: video_metadata(p), paths)))
    write(run/'data/media-metadata.json',metadata)
    media_horizons=[]
    for s in [*new, *bench]:
        count, fps = metadata[s['video_path']]
        if s['data_source']=='estp_bench' and s['source_indices'][-1]>=count:
            media_horizons.append(dict(span_id=s['span_id'],annotated_end=s['clip_end_time'],media_end=count/fps))
            bounded=estp_spec(s['video_id'],s['clip_id'],s['qa_index'],
                bench_source[s['video_id']][s['clip_id']][s['qa_index']],s['video_path'],
                training=False,benchmark=True,frame_count=count)
            s.clear();s.update(bounded)
        if abs(fps - 2) > .01 or s['source_indices'][-1] >= count:
            raise ValueError('invalid ESTP physical frame contract: ' + s['span_id'])
        s['source_frame_count'] = count
        s['measured_fps'] = fps
    original = rows(PARENT / 'data/episodes.jsonl')
    training = original + new[:111]
    prefix = read(PARENT / 'data/schedule.json')
    schedule = continuation_schedule(training, prefix)
    dump(run / 'data/episodes.jsonl', training)
    write(run / 'data/schedule.json', schedule)
    write(run / 'data/parent-schedule.json', prefix)
    sampled = {training[i]['span_id'] for b in schedule for i in b['episode_indices']}
    write(run / 'data/sampled-identities.json', sorted(sampled))
    cohorts = read(PARENT / 'data/cohorts.json')
    for d in ('web', 'ego'):
        cohorts[d + '_sampled'] = [q for q in cohorts[d + '_train'] if q in sampled]
        cohorts[d + '_unsampled'] = [q for q in cohorts[d + '_train'] if q not in sampled]
    # Empty cohorts are not sent to the scorer.
    cohorts = {k: v for k, v in cohorts.items() if v and k != 'gate'}
    cohorts.update(estp_it_dev=[s['span_id'] for s in new[111:239]],
                   estp_it_test=[s['span_id'] for s in new[239:]], estp_bench=[s['span_id'] for s in bench])
    evaluation = rows(PARENT / 'data/evaluation-specs.jsonl') + new[111:] + bench
    dump(run / 'data/evaluation-specs.jsonl', evaluation)
    dump(run / 'data/evaluation-manifest.jsonl', [dict(question_id=s['span_id'], video_id=s['video_id'],
        question=s['question'], resolved_video_path=s['video_path'], inference_source_indices=s['source_indices'],
        inference_timestamps=s['timestamps']) for s in evaluation])
    write(run / 'data/cohorts.json', cohorts)
    for s in new:
        if s['span_id'] in sampled and s['video_id'] not in splits['train']:
            raise ValueError('ESTP heldout leak')
    budget = schedule[-1]['exposure_after']
    counts = dict(updates=128, additional_updates=96, scheduled_decisions=budget,
        additional_decisions=budget-prefix[-1]['exposure_after'],
        additional_episodes=dict(Counter(training[i].get('data_source', training[i]['domain'])
            for b in schedule[32:] for i in b['episode_indices'])),
        cohort_sizes={k:len(v) for k,v in cohorts.items()}, estp=audit,
        max_decisions_per_batch=max(b['decisions'] for b in schedule))
    write(run / 'data/audit.json', dict(version='mixed-web-ego-estp-v3-to128', sampling_unit='episode', counts=counts,
        benchmark_short_media_horizons=media_horizons,
        annotation_sha256=sha(IT/'estp.json'), benchmark_annotation_sha256=sha(BENCH/'estp_bench_sq.json'),
        timestamp_contract='1fps on native 2fps media, clip-relative timestamps and unchanged native answer windows',
        training_filter='complete explicit SQ records, no dynamic users, every original answer window reachable on fixed grid; no widening',
        evaluation_filter='no reachability filtering; all 1212 ESTP-Bench SQ cases retained',
        reward='official PAUC-F/100 training proxy; official ESTP-F1 separately at endpoint; generator frozen',
        mixture='4 original / 2 WEB / 2 EGO+ESTP-IT; weights .5/.25/.25 by episode family',
        quality_policy='complete all 128 updates regardless of evaluation scores; correctness mandatory'))
    extend_replay_bank(Path(read(PARENT/'freeze.json')['anchor_manifest']), run/'anchor-extended', schedule)
    for name in ('anchor-calibration.json', 'runtime-contract.json'):
        shutil.copy2(PARENT/name, run/name)
    for step in range(1,33):
        for name in (f'step-{step:06d}', f'batch-{step-1:06d}'):
            (run/name).symlink_to(PARENT/name, target_is_directory=True)
    write(run/'latest.json', dict(step=32,checkpoint=str(run/'step-000032')))
    snapshot_trainer(run)
    plan = dict(version='mixed-web-ego-estp-v3-to128', schedule=schedule, decision_budget=budget,
        full_steps=[48,64,80,96,112,128], fast_steps=[], sampling_unit='episode',
        parent_run=str(PARENT), resume_checkpoint=str(checkpoint),
        training_node='ip-10-3-120-167', training_allocation=38177,
        evaluation_node='ip-10-3-70-177', evaluation_allocation=38226,
        coverage_gate=dict(enabled=False), quality_stopping=False,
        final_evaluation='training node after release of all eight GPUs',
        source_weights=dict(ours=.5,web=.25,ego_family=.25), source_hashes={})
    write(run/'workflow-freeze.json',plan)
    subprocess.run(trainer_command(run)+['--updates','128','--freeze-only'],check=True,cwd=ROOT)
    for name in ('training.sqlite', 'evaluation.sqlite'):
        with sqlite3.connect(PARENT/'cache'/name) as source_db:
            with sqlite3.connect(run/'cache'/name) as target_db:
                source_db.backup(target_db)
    print('Hashing new ESTP media inputs.', flush=True)
    previous = rows(PARENT/'data/video-hashes.jsonl')
    with ThreadPoolExecutor(max_workers=8) as pool:
        media = list(pool.map(lambda p: dict(path=p,sha256=sha(p)),paths))
    dump(run/'data/video-hashes.jsonl',previous+media)
    frozen_files = [*list((run/'data').glob('*')), *list((run/'code').glob('*.py')),
        ROOT/'scripts/prepare_jevany_estp128.py',ROOT/'scripts/run_jevany_estp128.py',ROOT/'scripts/eval_jevany_estp128.py',
        ROOT/'scripts/jevany_estp_control.py',ROOT/'src/pace_clm/rl_estp.py', run/'runtime-contract.json',
        SG/'third_party/estp_f1/estp_f1.py']
    # Include every inherited training input/scorer dependency in the new immutable manifest.
    plan['source_hashes'] = {**read(run/'freeze.json')['hashes'], **{str(p):sha(p) for p in frozen_files}}
    write(run/'workflow-freeze.json',plan)
    write(run/'prepared.json',dict(utc=datetime.now(timezone.utc).isoformat(),counts=counts,
        checkpoint_commit_sha256=sha(checkpoint/'commit.json'),hashes=plan['source_hashes']))
    print(json.dumps(counts),flush=True)
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    print(json.dumps(prepare(parser.parse_args().output.resolve())),flush=True)


if __name__=='__main__':
    main()
