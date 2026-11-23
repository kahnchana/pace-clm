#!/usr/bin/env python3
"""Freeze a four-source 2/2/2/2 continuation without changing historical runs."""
import argparse
import shutil
import sqlite3
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pace_clm.rl_anchor import verify_committed_checkpoint
from pace_clm.rl_balanced160 import SOURCES, balanced_schedule, source
from prepare_jevany_estp128 import (
    ROOT, SG, PY, IT, read, rows, sha, write, dump, select_episodes, snapshot_trainer,
)
from prepare_jevany_to32 import extend_replay_bank
from stream_gemma.inference.runtime import video_metadata

PARENT = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128'
DEFAULT = ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160'


def trainer_command(run):
    plan = read(run/'workflow-freeze.json')
    return list(map(str, [PY, run/'code/run_jevany_estp_rloo.py', '--output', run,
        '--spans', run/'data/episodes.jsonl', '--checkpoint', read(PARENT/'freeze.json')['checkpoint'],
        '--anchor-manifest', run/'anchor-extended/replay.jsonl', '--learners', 7, '--starts', 2,
        '--rollouts', 4, '--microbatch', 24, '--physical-batch', 24, '--collection-batch', 24,
        '--generator-batch', 8, '--frame-cache', 2048, '--decision-budget', plan['decision_budget'],
        '--reward-version', 'pauc-f-aligned-gpt-v1', '--full-run', '--sampling-unit', 'episode']))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source snapshot contract changed: ' + old[:100])
    return text.replace(old, new, 1)


def snapshot(run):
    snapshot_trainer(run)
    path = run/'code/run_jevany_estp_rloo.py'
    trainer = path.read_text()
    # Keep historical trainer/learner sources immutable. Only this run's snapshot changes.
    begin = trainer.index('    from pace_clm.rl_estp import continuation_schedule')
    end = trainer.index('\n    prepared=', begin)
    trainer = trainer[:begin] + (
        '    from pace_clm.rl_balanced160 import balanced_schedule\n'
        f'    episode_batches=balanced_schedule(spans,json.loads(Path({str(PARENT/"data/schedule.json")!r}).read_text()),seed=a.seed)'
    ) + trainer[end:]
    trainer = replace_once(trainer,
        'from pace_clm.rl_mixed import training_schedule, validate_spec, attach_loss_scales',
        'from pace_clm.rl_mixed import training_schedule, validate_spec\n'
        'from pace_clm.rl_balanced160 import attach_source_loss_scales as attach_loss_scales')
    trainer = trainer.replace('from jevany_estp_control import', 'from jevany_balanced160_control import')
    trainer = trainer.replace('complete through step 128 irrespective of scores', 'complete through step 160 irrespective of scores')
    trainer = replace_once(trainer,
        "freeze['domain_objective']='0.5 mean ours + 0.25 mean WEB + 0.25 mean EGO/ESTP-IT family; constant 32 loss scale'",
        "freeze['domain_objective']='0.25 mean each of original, WEB, EGO and ESTP-IT; constant 32 loss scale'")
    trainer = replace_once(trainer, "    freeze['hashes'] =", 
        "    files += [ROOT/'src/pace_clm/rl_balanced160.py',ROOT/'scripts/jevany_balanced160_control.py']\n    freeze['hashes'] =")
    path.write_text(trainer)
    probe = (ROOT/'scripts/probe_jevany_mixed_parity.py').read_text()
    probe = replace_once(probe, "initial=Path(cfg['checkpoint'])", f"initial=Path({str(PARENT/'step-000128')!r})")
    probe = probe.replace('batch-000000/scored.json', 'batch-000128/scored.json')
    (run/'code/probe_estp_parity.py').write_text(probe)
    evaluator = (ROOT/'scripts/eval_jevany_estp128.py').read_text()
    evaluator = evaluator.replace('baseline-step32', 'baseline-step128').replace('paired_vs_step32', 'paired_vs_step128')
    evaluator = evaluator.replace('complete all 128 updates', 'complete all 160 updates')
    # This experiment uses the same development cohorts at 128/144/160. Endpoint
    # IT test and ESTP-Bench remain in the separately recovered parent final suite.
    (run/'code/eval_balanced160_queue.py').write_text(evaluator)


def prepare(run):
    if (run/'prepared.json').exists():
        for path, expected in read(run/'prepared.json')['hashes'].items():
            if sha(path) != expected:
                raise ValueError('frozen input changed: ' + path)
        return read(run/'prepared.json')['counts']
    if run.exists():
        raise FileExistsError('inspect incomplete preparation before retry: ' + str(run))
    if read(PARENT/'latest.json')['step'] != 128 or read(PARENT/'status.json')['steps'] != 128:
        raise ValueError('parent training not complete at step128')
    verify_committed_checkpoint(PARENT/'step-000128', 128)
    for path, expected in read(PARENT/'workflow-freeze.json')['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('parent frozen source changed: ' + path)
    print('Verified parent checkpoint and frozen sources.', flush=True)
    for name in ('data', 'code', 'cache', 'queue', 'evaluation'):
        (run/name).mkdir(parents=True)
    for f in (PARENT/'data').iterdir():
        if f.is_file():
            shutil.copy2(f, run/'data'/f.name)
    episodes = rows(PARENT/'data/episodes.jsonl')
    prefix = read(PARENT/'data/schedule.json')
    old_videos = {s['video_id'] for s in episodes if source(s) == 'estp_it'}
    splits = read(PARENT/'data/estp-video-splits.json')
    fresh_videos = [v for v in splits['train'] if v not in old_videos]
    new, rejected, tasks, eligible = select_episodes(read(IT/'estp.json'), fresh_videos, 64, training=True)
    if len({s['video_id'] for s in new}) != 64:
        raise ValueError('require 64 fresh IT source videos')
    paths = sorted({s['video_path'] for s in new})
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = dict(zip(paths, pool.map(video_metadata, paths)))
    for s in new:
        count, fps = metadata[s['video_path']]
        if abs(fps-2) > .01 or s['source_indices'][-1] >= count:
            raise ValueError('ESTP media contract failed: ' + s['span_id'])
        s.update(source_frame_count=count, measured_fps=fps)
    episodes += new
    schedule = balanced_schedule(episodes, prefix)
    selected = [i for b in schedule[128:] for i in b['episode_indices']]
    before = {i for b in prefix for i in b['episode_indices']}
    reused = Counter(source(episodes[i]) for i in selected if i in before)
    if dict(reused) != {'ego': 64}:
        raise ValueError('unexpected episode reuse: ' + str(reused))
    evaluation = rows(run/'data/evaluation-specs.jsonl')
    parent_cohorts = read(run/'data/cohorts.json')
    heldout = {q for k in ('ours_heldout','web_heldout','ego_heldout','estp_it_dev','estp_it_test','estp_bench')
               for q in parent_cohorts.get(k, [])}
    if any(episodes[i]['span_id'] in heldout for i in selected):
        raise ValueError('heldout identity in training schedule')
    if any(s['video_id'] not in splits['train'] for s in new):
        raise ValueError('IT source-video split leak')
    dump(run/'data/episodes.jsonl', episodes)
    dump(run/'data/estp-phase-train.jsonl', new)
    write(run/'data/parent-schedule.json', prefix)
    write(run/'data/schedule.json', schedule)
    all_sampled = {episodes[i]['span_id'] for b in schedule for i in b['episode_indices']}
    phase_sampled = {episodes[i]['span_id'] for i in selected}
    write(run/'data/sampled-identities.json', sorted(all_sampled))
    write(run/'data/phase-sampled-identities.json', sorted(phase_sampled))
    cohorts = read(run/'data/cohorts.json')
    for d in ('web', 'ego'):
        for label, ids in [('sampled', all_sampled), ('phase_sampled', phase_sampled)]:
            cohorts[d+'_'+label] = [q for q in cohorts[d+'_train'] if q in ids]
        cohorts[d+'_unsampled'] = [q for q in cohorts[d+'_train'] if q not in all_sampled]
        cohorts[d+'_phase_unsampled'] = [q for q in cohorts[d+'_train'] if q not in phase_sampled]
    write(run/'data/cohorts.json', {k:v for k,v in cohorts.items() if v})
    budget = schedule[-1]['exposure_after']
    counts = dict(updates=160, additional_updates=32, scheduled_decisions=budget,
        additional_decisions=budget-prefix[-1]['exposure_after'],
        additional_episodes=dict(Counter(source(episodes[i]) for i in selected)),
        reused_parent_episodes=dict(reused), fresh_it_videos=64, it_eligible_fresh_videos=eligible,
        it_rejected_records=rejected, it_selected_task_types=tasks,
        additional_domain_decisions={d: sum(b['domain_decisions'][d] for b in schedule[128:]) for d in ('ours','web','ego')})
    audit = read(run/'data/audit.json')
    audit.update(version='four-source-balanced-v4-to160', counts=counts,
        mixture='2 original / 2 WEB / 2 EGO / 2 ESTP-IT; source objective weights .25 each',
        reuse_policy='64 native EGO episodes reused once; 64 fresh episodes from each other source; no within-phase episode repeats',
        quality_policy='complete all 32 continuation updates; correctness failures require recovery')
    write(run/'data/audit.json', audit)
    extend_replay_bank(Path(read(PARENT/'freeze.json')['anchor_manifest']), run/'anchor-extended', schedule)
    for name in ('anchor-calibration.json','runtime-contract.json'):
        shutil.copy2(PARENT/name, run/name)
    for step in range(1,129):
        for name in (f'step-{step:06d}',f'batch-{step-1:06d}'):
            (run/name).symlink_to(PARENT/name, target_is_directory=True)
    write(run/'latest.json', dict(step=128, checkpoint=str(run/'step-000128')))
    snapshot(run)
    plan = dict(version='four-source-balanced-v4-to160', schedule=schedule, decision_budget=budget,
        parent_run=str(PARENT), resume_checkpoint=str(PARENT/'step-000128'), resume_step=128,
        target_step=160, full_steps=[144,160], fast_steps=[], sampling_unit='episode',
        source_weights={name:.25 for name in SOURCES}, rollouts_per_episode=4,
        training_node='ip-10-3-120-167', training_allocation=38177,
        evaluation_node='ip-10-3-70-177', evaluation_allocation=38226,
        final_evaluation='assigned evaluation node, per 2026-11-15 user instruction; development cohorts only',
        parent_final_recovery='original training node after step160 releases GPUs; preserve original endpoint jobs/shards',
        source_hashes={}, quality_stopping=False, coverage_gate=dict(enabled=False))
    write(run/'workflow-freeze.json', plan)
    subprocess.run(trainer_command(run)+['--updates','160','--freeze-only'], check=True, cwd=ROOT)
    for name in ('training.sqlite','evaluation.sqlite'):
        with sqlite3.connect(f'file:{PARENT/"cache"/name}?mode=ro', uri=True) as src:
            with sqlite3.connect(run/'cache'/name) as dst:
                src.backup(dst)
    print('Hashing 64 fresh IT media inputs.', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        media = list(pool.map(lambda p:dict(path=p,sha256=sha(p)),paths))
    dump(run/'data/video-hashes.jsonl', rows(PARENT/'data/video-hashes.jsonl')+media)
    frozen = [*list((run/'data').glob('*')), *list((run/'code').glob('*.py')),
        ROOT/'scripts/prepare_jevany_balanced160.py', ROOT/'scripts/run_jevany_balanced160.py',
        ROOT/'scripts/run_jevany_balanced160.sh', ROOT/'scripts/jevany_balanced160_control.py',
        ROOT/'scripts/with_jevany_nvme_cache.sh', ROOT/'src/pace_clm/rl_balanced160.py', run/'runtime-contract.json']
    plan['source_hashes'] = {**read(run/'freeze.json')['hashes'], **{str(p):sha(p) for p in frozen}}
    write(run/'workflow-freeze.json', plan)
    write(run/'prepared.json', dict(utc=datetime.now(timezone.utc).isoformat(), counts=counts,
        checkpoint_commit_sha256=sha(PARENT/'step-000128/commit.json'), hashes=plan['source_hashes']))
    return counts


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    print(prepare(parser.parse_args().output.resolve()),flush=True)
