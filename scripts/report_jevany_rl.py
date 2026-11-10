#!/usr/bin/env python3
"""Aggregate complete-update throughput and phase-filtered GPU telemetry."""
import argparse
from collections import Counter
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def report(path, warmup=1):
    freeze = json.loads((path/'freeze.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(path.glob('batch-*/report.json'))]
    measured = rows[warmup:]
    if not measured:
        raise ValueError('no complete warm updates after warmup')
    samples = []
    with (path/'gpu_samples.csv').open() as handle:
        for fields in csv.reader(handle):
            if len(fields) != 4:
                continue
            try:
                ts = datetime.strptime(fields[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                samples.append((ts, int(fields[1]), float(fields[2]), float(fields[3])))
            except ValueError:
                continue
    phases = {'complete':{}, 'collection':{}, 'judge':{}, 'learner':{}}
    for phase in phases:
        selected = []
        for row in measured:
            start, stop = {
                'complete': (row['started_epoch'], row['ended_epoch']),
                'collection': (row['started_epoch'], row['collection_ended_epoch']),
                'judge': (row['collection_ended_epoch'], row['judge_ended_epoch']),
                'learner': (row['judge_ended_epoch'], row['learner_ended_epoch']),
            }[phase]
            selected.extend(s for s in samples if start <= s[0] <= stop)
        for gpu in range(8):
            subset = [s for s in selected if s[1] == gpu]
            if subset:
                phases[phase][gpu] = {'mean_util': statistics.mean(s[2] for s in subset),
                                     'idle_fraction': statistics.mean(s[2] <= 5 for s in subset),
                                     'peak_device_gib': max(s[3] for s in subset)/1024,
                                     'samples': len(subset)}
    seconds = sum(r['seconds'] for r in measured)
    decisions = sum(r['decisions'] for r in measured)
    output = {'run':str(path), 'learners':freeze['learners'], 'generators':8-freeze['learners'],
              'microbatch':rows[-1]['microbatch'], 'physical_batch':freeze['physical_batch'],
              'starts':freeze['starts'],'rollouts_per_start':freeze['rollouts'], 'warmup_updates':warmup,
              'completed_updates':len(rows),
              'measured_updates':len(measured), 'update_seconds':[r['seconds'] for r in measured],
              'mean_update_seconds':seconds/len(measured), 'decisions_per_second':decisions/seconds,
              'allocated_gpu_hours':8*seconds/3600,
              'gpu_hours_per_1000_decisions':8*seconds/3600*1000/decisions,
              'startup_seconds_from_first_gpu_sample':rows[0]['started_epoch']-min(s[0] for s in samples) if samples else None,
              'mean_collection_seconds':statistics.mean(r['collection_ended_epoch']-r['started_epoch'] for r in measured),
              'mean_learner_seconds':statistics.mean(r['learner_ended_epoch']-r['judge_ended_epoch'] for r in measured),
              'phase_gpu_samples':phases,
              'max_replay_logp_error':max(v['parity_max'] for r in rows for v in r['learner'].values()),
              'max_replay_probability_error':max(v['parity_probability_max'] for r in rows for v in r['learner'].values()),
              'max_replay_kl':max(v['parity_kl_max'] for r in rows for v in r['learner'].values()),
              'max_replay_mean_kl':max(v['parity_kl_mean'] for r in rows for v in r['learner'].values()),
              'max_selected_logp_error':max(v['parity_selected_logp_max'] for r in rows for v in r['learner'].values()),
              'max_actor_allocated_gib':max(v['allocated_gib'] for r in measured for v in r['learner'].values()),
              'max_actor_reserved_gib':max(v['reserved_gib'] for r in measured for v in r['learner'].values()),
              'min_lora_grad_norm':min(v['lora_grad_norm'] for r in rows for v in r['learner'].values()),
              'min_head_grad_norm':min(v['head_grad_norm'] for r in rows for v in r['learner'].values()),
              'mean_judge_seconds':statistics.mean(r['judge']['seconds'] for r in measured),
              'mean_save_seconds':statistics.mean(r['save_seconds'] for r in measured)}
    output['token_efficiency'] = {}
    for name in ('collection','learner'):
        preparation = [v['preparation'] for r in measured for v in r[name].values()]
        real = sum(v.get('real_tokens',0) for v in preparation)
        physical = sum(v.get('physical_tokens',0) for v in preparation)
        real_rows = sum(sum(v.get('real_batch_sizes',[])) for v in preparation)
        physical_rows = sum(sum(shape[0] for shape in v.get('microbatch_shapes',[])) for v in preparation)
        output['token_efficiency'][name] = {
            'physical_shape_histogram':dict(Counter(f'{shape[0]}x{shape[1]}' for v in preparation for shape in v.get('microbatch_shapes',[]))),
            'real_tokens':real,'physical_tokens':physical,
            'real_token_fraction':real/physical if physical else None,
            'real_row_fraction':real_rows/physical_rows if physical_rows else None}
    generator_batches, request_times = {}, []
    for row in measured:
        traces=json.loads((path/f'batch-{row["step"]-1:06d}'/'rollouts.json').read_text())
        for trace in traces:
            for step in trace['steps']:
                stats=step.get('generator_stats')
                if stats:
                    if stats['batch']>0:generator_batches[(stats['gpu'],stats['batch_id'])]=stats
                    request_times.append(stats['request_seconds'])
    output['generator_broker']={k:sum(r.get('generator_broker',{}).get(k,0) for r in measured) for k in ('logical_requests','unique_computations','cache_hits','inflight_duplicates')}
    broker=output['generator_broker']
    broker['reuse_fraction']=1-broker['unique_computations']/broker['logical_requests'] if broker['logical_requests'] else 0
    restore_path=path/'resume_verification.json'
    output['resume_verification']=json.loads(restore_path.read_text()) if restore_path.exists() else None
    output['generator_mean_batch']=statistics.mean(v['batch'] for v in generator_batches.values()) if generator_batches else 0
    output['generator_mean_request_seconds']=statistics.mean(request_times) if request_times else 0
    output['generator_max_request_seconds']=max(request_times,default=0)
    (path/'profile_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    return {k:v for k,v in output.items() if k != 'phase_gpu_samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--warmup',type=int,default=1)
    args=parser.parse_args()
    print(json.dumps([report(p,args.warmup) for p in args.runs],indent=2))
