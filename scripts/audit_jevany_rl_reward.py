#!/usr/bin/env python3
"""Persist a stratified E1 training-only sample and audit unchanged answers."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random

from jevany_rl_reward import ROOT, contract, score_traces, sha, write


def select(source, output, seed=42):
    target = output / 'selected.json'
    if target.exists():
        return json.loads(target.read_text())
    rng = random.Random(seed)
    buckets = defaultdict(list)
    provenance = []
    for batch in sorted(source.glob('batch-*')):
        number = int(batch.name.split('-')[1])
        phase = 'early' if number < 8 else 'middle' if 12 <= number < 20 else 'late' if number >= 24 else None
        if phase is None:
            continue
        path = batch / 'scored.json'
        traces = json.loads(path.read_text())
        groups = defaultdict(list)
        for trace in traces:
            groups[trace['spec']['span_id']].append(trace)
        for span_id, members in groups.items():
            if len(members) != 4:
                raise ValueError('incomplete four-rollout group')
            kind = ('none', 'single', 'multiple')[min(2, len(members[0]['spec']['gold_points']))]
            small = []
            for trace in members:
                small.append({'trajectory_id': trace['trajectory_id'], 'spec': trace['spec'],
                              'e1_reward': trace['reward'],
                              'steps': [{'action': s['action'], 'answer': s['answer'],
                                         'row': {'frame_index': s['row']['frame_index']}}
                                        for s in trace['steps']]})
            buckets[phase, kind].append({'batch': batch.name, 'span_id': span_id,
                                         'phase': phase, 'stratum': kind, 'traces': small})
        provenance.append({'path': str(path), 'sha256': sha(path)})
    chosen = []
    for index, key in enumerate(sorted(buckets)):
        rng.shuffle(buckets[key])
        count = 15 if index < 2 else 14
        if len(buckets[key]) < count:
            raise ValueError(f'insufficient audit groups: {key}')
        chosen.extend(buckets[key][:count])
    write(target, chosen)
    write(output / 'selection.json', {'seed': seed, 'groups': len(chosen),
                                     'source': str(source), 'provenance': provenance,
                                     'selection_sha256': sha(target), 'split': 'training only'})
    return chosen


def compare(old, new):
    counts = Counter()
    for i, j in combinations(range(4), 2):
        a, b = old[i] - old[j], new[i] - new[j]
        if abs(a) < 1e-10 and abs(b) < 1e-10:
            counts['both_tied'] += 1
        elif abs(a) < 1e-10:
            counts['old_tie_broken'] += 1
        elif abs(b) < 1e-10:
            counts['new_tie'] += 1
        elif a * b < 0:
            counts['strict_inversion'] += 1
        else:
            counts['strict_agreement'] += 1
    tops = lambda values: [i for i, x in enumerate(values) if abs(x - max(values)) < 1e-10]
    return dict(counts), tops(old), tops(new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/jevany-2b-rl/profile-7p1-v10')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/jevany-2b-rl/reward-audit-v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = select(args.source, args.output)
    traces = [t for group in selected for t in group['traces']]
    judge = score_traces(traces, args.output / 'gpt', args.output / 'gpt-cache.sqlite')
    results, strata = [], defaultdict(Counter)
    for group in selected:
        members = group['traces']
        old, new = [t['e1_reward'] for t in members], [t['reward'] for t in members]
        pairs, old_top, new_top = compare(old, new)
        actions = {tuple(s['action'] for s in t['steps']) for t in members}
        histories = {tuple((s['action'], s['answer']) for s in t['steps']) for t in members}
        fires = [sum(s['action'] == 2 for s in t['steps']) for t in members]
        coverage = [t['reward_metrics']['num_value_adding_fires'] for t in members]
        tied = max(new) - min(new) < 1e-10
        category = ('identical_histories' if len(histories) == 1 else
                    'different_histories_equal_outcomes' if len(set(zip(fires, coverage))) == 1 else
                    'coarse_judging_or_metric_tie') if tied else 'informative'
        result = {k: group[k] for k in ('batch', 'span_id', 'phase', 'stratum')}
        result.update(old_rewards=old, aligned_rewards=new, pairwise=pairs,
                      old_top_set=old_top, new_top_set=new_top,
                      top_set_changed=old_top != new_top, top_sets_disjoint=not set(old_top) & set(new_top),
                      action_histories=len(actions), emitted_histories=len(histories),
                      response_counts=fires, useful_coverage=coverage,
                      reward_range=max(new)-min(new), tie_category=category)
        results.append(result)
        for key in (group['stratum'], group['phase'], 'all'):
            strata[key].update(groups=1, top_set_changed=int(old_top != new_top),
                               informative=int(not tied), **{category: 1} if tied else {})
            strata[key].update(pairs)
    # A reward that is constant despite differing useful coverage/response counts
    # fails a basic reliability gate. Unjudged/incompatible rubric comparisons
    # are explicitly unavailable, never relabeled as formula-only evidence.
    differing = [r for r in results if len(set(zip(r['response_counts'], r['useful_coverage']))) > 1]
    informative = [r for r in differing if r['reward_range'] > 1e-10]
    event_informative = [r for r in informative if r['stratum'] != 'none']
    passed = len(informative) >= max(1, len(differing) * .5) and len(event_informative) >= 10
    summary = {'groups': len(results), 'trajectories': len(traces), 'judge': judge,
               'strata': dict(strata), 'behavior_differing_groups': len(differing),
               'behavior_differing_informative': len(informative),
               'event_informative': len(event_informative),
               'gate_passed': passed,
               'gate': 'at least half of differing outcome groups informative; at least ten event groups informative',
               'limitations': ['E1 local rubric and official GPT cumulative rubric are incompatible; local-to-official score conversion unavailable.',
                              'Formula-only and judge-only ranking isolation unavailable; observed differences combine formula, aggregation and judge.',
                              'Equal reward with different outcomes may reflect metric ties or coarse/noisy judging; cause is not proven.',
                              'This is a single GPT judgment per unique request, not a reliability estimate from repeated judgments.']}
    write(args.output / 'groups.json', results)
    write(args.output / 'summary.json', summary)
    write(args.output / 'scored.json', traces)
    write(args.output / 'reward-spec.json', contract())
    lines = ['# E2-R reward audit', '',
             f"Training-only sample: {len(results)} complete groups / {len(traces)} saved trajectories; seed 42.",
             f"Gate passed: **{passed}**. Differing-outcome groups with reward signal: {len(informative)}/{len(differing)}; event groups: {len(event_informative)}.",
             '', '## Stratified ranking and tie diagnostics', '', '```json',
             json.dumps(dict(strata), indent=2), '```', '', '## Interpretation', '',
             *summary['limitations'], '',
             'Freeze GPT-4.1 official cumulative-answer judging for training and evaluation. Do not substitute E1 local judgments.',
             'Preserve overlapping MarkII windows and the official inclusive window-end convention. All physical fires enter precision.',
             'PAUC starts at 0.5/2 even for empty windows; raw useful coverage is reported separately. Zero-fire precision and F are zero.',
             'No-event reward is -0.25 per fire: waiting earns zero; each extra fire is strictly worse.',
             'Mean clip F is a surrogate and is not global benchmark F. Historical E1 comparison changes both reward and judge.', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary), flush=True)
    if not passed:
        raise RuntimeError('reward audit gate failed; investigate scoring/exploration before training')


if __name__ == '__main__':
    main()
