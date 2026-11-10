#!/usr/bin/env python3
"""Learning curves, matched development comparison and paired WEB intervals."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from jevany_rl_reward import ROOT, SG, OFFICIAL, write
from zs_infer.pauc_f import fire_times, assign_value_adding_fires


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sufficient(directory):
    predictions = rows(directory/'predictions.jsonl')
    scores = {r['question_id']: r for r in read(directory/'pauc_llm_scores-gpt-4.1.json')}
    result = {}
    for prediction in predictions:
        item = scores[prediction['question_id']]
        fires = fire_times(prediction)
        useful = len(assign_value_adding_fires(fires, item['answer']))
        areas = []
        for turn in item['answer']:
            points = [[float(time)-turn['reply_timespan'][0], score]
                      for time, score in turn['judge_scores'].items()]
            area = OFFICIAL.area_under_line_ratio(points, turn['reply_timespan'][1]-turn['reply_timespan'][0], 2, .5, .5) if points else .25
            areas.append(area)
        result[prediction['question_id']] = [useful, len(fires), sum(areas), len(areas)]
    return result


def metric(statistics):
    useful, fires, area, turns = statistics.T
    precision = np.divide(useful, fires, out=np.zeros_like(useful), where=fires > 0)
    recall = np.divide(area, turns, out=np.zeros_like(area), where=turns > 0)
    return 100 * np.divide(2*precision*recall, precision+recall,
                           out=np.zeros_like(precision), where=precision+recall > 0)


def paired_bootstrap(candidate, baseline, n=5000):
    left, right = sufficient(candidate), sufficient(baseline)
    if left.keys() != right.keys():
        raise ValueError('paired video IDs differ')
    ids = sorted(left)
    a, b = np.array([left[i] for i in ids], dtype=float), np.array([right[i] for i in ids], dtype=float)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), (n, len(ids)))
    deltas = metric(a[draws].sum(1)) - metric(b[draws].sum(1))
    # Verify our sufficient-statistic aggregation against the official summaries.
    for directory, stats in ((candidate, a), (baseline, b)):
        actual = read(directory/'pauc_f_summary-gpt-4.1-v1.json')['pauc_f']
        if abs(float(metric(stats.sum(0)[None])[0]) - actual) > 1e-8:
            raise ValueError('bootstrap aggregation does not match official scorer')
    return {'paired_videos': len(ids), 'replicates': n, 'seed': 42,
            'difference_points': float(metric(a.sum(0)[None])[0]-metric(b.sum(0)[None])[0]),
            'percentile_95_ci': np.quantile(deltas, [.025, .975]).tolist(),
            'scope': 'video sampling uncertainty conditional on saved judgments; not judge/retraining uncertainty'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    curves = []
    for batch in sorted(run.glob('batch-*')):
        if not (batch/'report.json').exists():
            continue
        report = read(batch/'report.json')
        learners = list(report['learner'].values())
        traces = read(batch/'scored.json')
        groups, strata = defaultdict(list), defaultdict(Counter)
        action_counts = Counter(s['action'] for t in traces for s in t['steps'])
        for trace in traces:
            groups[trace['spec']['span_id']].append(trace)
        for members in groups.values():
            kind = ('none', 'single', 'multiple')[min(2, len(members[0]['spec']['gold_points']))]
            rewards = [t['reward'] for t in members]
            strata[kind].update(groups=1, equal_reward=int(max(rewards)-min(rewards)<1e-10))
        curve = {'round': report['step'], 'optimizer_steps': report['optimizer_steps'],
                 'seconds': report['seconds'], 'allocated_gpu_hours': report['gpu_hours'],
                 'mean_reward': report['mean_reward'], 'fires': report['fires'],
                 'useful_fires': sum(t['reward_metrics']['num_value_adding_fires'] for t in traces),
                 'wasted_fires': sum(t['reward_metrics']['num_wasted_fires'] for t in traces),
                 'entropy': sum(r['entropy_sum'] for r in learners)/report['decisions'],
                 'sft_kl': sum(r['kl_sum'] for r in learners)/report['decisions'],
                 'grad_norm': max(r['grad_norm'] for r in learners),
                 'max_parity_probability_error': max(r['parity_probability_max'] for r in learners),
                 'max_parity_logp_error': max(r['parity_max'] for r in learners),
                 'silence': action_counts[0], 'hold': action_counts[1], 'response': action_counts[2],
                 'equal_reward_groups': sum(v['equal_reward'] for v in strata.values()),
                 'strata': dict(strata)}
        curves.append(curve)
    write(run/'learning-curves.json', curves)
    if curves:
        with (run/'learning-curves.csv').open('w') as handle:
            fields = [k for k in curves[0] if k != 'strata']
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(curves)
    evaluations = []
    for name in ('dev-sft', 'dev-step8', 'dev-step16', 'dev-step32'):
        path = run/'evaluation'/name/'pauc_f_summary-gpt-4.1-v1.json'
        if path.exists():
            evaluations.append({'checkpoint': name, **read(path)})
    table = ['| Checkpoint | PAUC-F | Precision | Temporal recall | Useful / wasted fires |',
             '|---|---:|---:|---:|---:|']
    for score in evaluations:
        table.append(f"| {score['checkpoint']} | {score['pauc_f']:.4f} | {score['fire_precision']:.2f} | {score['pauc_recall_component']:.2f} | {score['num_value_adding_fires']} / {score['num_wasted_fires']} |")
    bootstrap = None
    if (run/'selected.json').exists():
        step = read(run/'selected.json')['round']
        candidate = run/'evaluation'/f'web-selected-step{step}'
        if (candidate/'pauc_f_summary-gpt-4.1-v1.json').exists():
            baseline = ROOT/'outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web'
            # Match scoring/generation settings; SFT control is a retained
            # historical evaluation, never described as a fresh paired rerun.
            a, b = read(candidate/'manifest.json'), read(baseline/'manifest.json')
            for key in ('response_margin', 'max_new_tokens', 'window_frames',
                        'memory_mode', 'memory_instruction', 'memory_select',
                        'history', 'original_generator_adapter_sha256'):
                if a[key] != b[key]:
                    raise ValueError(f'unmatched WEB baseline protocol: {key}')
            bootstrap = paired_bootstrap(candidate, baseline)
            write(run/'paired-web-bootstrap.json', bootstrap)
    status = read(run/'workflow-status.json')
    summary = {'state': status, 'completed_rounds': len(curves),
               'optimizer_steps': len(curves), 'sampled_decisions': len(curves)*4096,
               'complete_round_node_hours': sum(c['seconds'] for c in curves)/3600,
               'complete_round_allocated_gpu_hours': sum(c['allocated_gpu_hours'] for c in curves),
               'evaluations': evaluations, 'paired_web_bootstrap': bootstrap,
               'historical_e1_web_pauc_f': 53.0861, 'historical_sft_web_pauc_f': 54.6651,
               'comparison_limits': 'E1 vs E2-R changes reward formula, answer aggregation and judge. One seed. Reused WEB development. E2-G not run.'}
    write(run/'result.json', summary)
    lines = ['# E2-R: reward-aligned RLOO', '',
             f"Completed rounds / optimizer steps: {len(curves)} / {len(curves)}; sampled decisions: {len(curves)*4096:,}.",
             f"Complete-round training cost: {summary['complete_round_node_hours']:.2f} node-hours / {summary['complete_round_allocated_gpu_hours']:.2f} allocated GPU-hours. Initialization, audit, evaluation and inter-round restarts are additional.",
             '', '128-video reused WEB development, video-disjoint from the checked SFT/RL training cohort. Greedy raw learned actions; response margin zero; actual generated history; GPT-4.1 official judging.',
             '', *table, '',
             'Reward: official clip PAUC-F / 100 on event spans; -0.25 per physical fire on no-event spans. GPT-4.1 cumulative-answer rubric. Fresh SFT initialization; one RLOO update per round.',
             '', summary['comparison_limits'], '',
             f"Workflow status at report generation: {status['state']}.",
             'Historical E1 WEB: 53.0861; historical SFT WEB: 54.6651. Development scores above use a different video subset and must not be compared directly with these full-WEB values.', '']
    if bootstrap:
        lines += ['Selected checkpoint full WEB, compared with retained historical SFT control:',
                  '```json', json.dumps(bootstrap, indent=2), '```', '']
    elif len(curves) == 32 and not (run/'selected.json').exists():
        lines += ['No checkpoint exceeded development SFT PAUC-F while retaining at least 90% useful coverage. No full-WEB sweep or automatic extension was launched.', '']
    (run/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
