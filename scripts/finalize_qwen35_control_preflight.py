#!/usr/bin/env python3
"""Check the GPU preflight artifacts and freeze an estimated twelve-hour step budget."""
import hashlib
import ast
import json
import importlib.metadata
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file

from run_qwen35_control import BASE, ROOT, write_json
from prepare_qwen35_control import aliases, identities, SG


def compare(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: shape/dtype differ')
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7, msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i,(lval,rval) in enumerate(zip(left,right)):
            compare(lval, rval, f'{path}.{i}')
    else:
        assert left == right, path


def estimate_steps(seconds_per_step, rows):
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError('invalid measured step duration')
    total = max(4, round(43200 / seconds_per_step / 4) * 4)
    if total * 128 > rows:
        raise ValueError('estimated run would exhaust the eligible pool')
    return total, [total//4*i for i in range(1,5)]


def main():
    out = BASE / 'control-seed42'
    checkpoint = lambda name: out / 'preflight' / name / 'step-000002'
    left, right = checkpoint('continuous'), checkpoint('resumed')
    for rank in range(8):
        restored = json.loads((out/f'preflight/restore/verified.rank{rank}.json').read_text())
        assert restored == {'exact_restore':True,'step':1,'seen_per_rank':16}
    def differences(a,b):
        head_a = torch.load(a/'pointer_head.pt',map_location='cpu',weights_only=True)
        head_b = torch.load(b/'pointer_head.pt',map_location='cpu',weights_only=True)
        adapter_a = load_file(a/'controller_adapter/adapter_model.safetensors')
        adapter_b = load_file(b/'controller_adapter/adapter_model.safetensors')
        return {'head_max_absolute':max((head_a[k]-head_b[k]).abs().max().item() for k in head_a),
                'adapter_max_absolute':max((adapter_a[k]-adapter_b[k]).abs().max().item() for k in adapter_a)}
    baseline = differences(left,checkpoint('continuous-attempt1'))
    resumed_difference = differences(left,right)
    # Independent uninterrupted repetitions already differ; checkpoint restoration
    # is verified exactly above rather than relaxing its tolerance to hide drift.
    for key in baseline:
        if resumed_difference[key] > max(1e-7,baseline[key]*1.5):
            raise ValueError('resume divergence exceeds the independent-run numerical baseline')
    for rank in range(8):
        compare(torch.load(left/f'rng.rank{rank}.pt',weights_only=False),
                torch.load(right/f'rng.rank{rank}.pt',weights_only=False), f'rng{rank}')
    resume_before = torch.load(out/'preflight/resumed/step-000001/training_state.pt',weights_only=False,map_location='cpu')
    resume_after = torch.load(right/'training_state.pt',weights_only=False,map_location='cpu')
    assert resume_after['training_seconds'] > resume_before['training_seconds'] > 0
    assert resume_after['step'] == 2 and resume_after['seen_per_rank'] == 32
    assert all(float(v['step']) == 2 for v in resume_after['optimizer']['state'].values())
    cached = out/'preflight/representative_cached'
    if not (cached/'summary.rank0.json').exists():
        with (out/'logs/preflight_representative_cached.log').open('w') as log:
            subprocess.run(['bash',str(ROOT/'scripts/run_qwen35_control_command.sh'),'train',
                '--manifest',str(BASE/'shared/manifests/probe.jsonl'),'--output',str(cached),
                '--initialize-from',str(BASE/'shared/initial-seed42/step-000000'),
                '--stop-after-steps','24'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    summaries = []
    for name in ('continuous','resumed','representative','representative_cached','stress'):
        for rank in range(8):
            summary = json.loads((out/f'preflight/{name}/summary.rank{rank}.json').read_text())
            if summary['peak_reserved_gib'] > summary['device_memory_gib'] * 0.9:
                raise ValueError(f'{name} rank {rank} has less than 10% memory headroom')
            summaries.append({'name':name,'rank':rank,**summary})
    log = out / 'logs/preflight_representative_cached.log'
    updates = []
    for line in log.read_text().splitlines():
        if line.startswith('{'):
            value = json.loads(line)
            if 'step' in value and 'training_seconds' in value:
                updates.append(value)
    if len(updates) != 24:
        raise ValueError('expected 24 representative updates')
    raw_seconds_per_step = (updates[-1]['training_seconds']-updates[1]['training_seconds'])/22
    audit = json.loads((BASE/'shared/manifests/audit.json').read_text())
    # The probe deliberately samples short/middle/long contexts equally. Weight
    # its measured costs by the frozen pool's actual context distribution rather
    # than treating the oversampled 32-frame stress cases as one third of training.
    groups = {}
    for i, update in enumerate(updates):
        if i < 6:
            continue
        elapsed = update['training_seconds']-updates[i-1]['training_seconds']
        frames = sum(update['rank0_context_frames'])/16
        group = groups.setdefault(i % 3, {'frames':[],'seconds':[]})
        group['frames'].append(frames)
        group['seconds'].append(elapsed)
    anchors = sorted((np.mean(v['frames']),np.mean(v['seconds'])) for v in groups.values())
    seconds_per_step = sum(np.interp(ast.literal_eval(key)[0], [x[0] for x in anchors], [x[1] for x in anchors])*count
                           for key,count in audit['contexts_and_memory'].items())/audit['presentations']
    # Supplemental zero-overlap proof for TV/VAD when preparation began before
    # these paths were added to exclusions. Verify the full metadata provenance.
    metadata_path = SG/'datasets/markii/metadata.json'
    metadata_bytes = metadata_path.read_bytes()
    source_summary = json.loads((ROOT/'outputs/jevany-qwen35-2b-full/manifests/source.summary.json').read_text())
    j1s_summary = json.loads((ROOT/'outputs/jevany-qwen35-2b-full/manifests/j1s.summary.json').read_text())
    assert hashlib.sha256(metadata_bytes).hexdigest() == source_summary['metadata_sha256']
    assert j1s_summary['source_sha256'] == source_summary['manifest_sha256']
    all_source_aliases = set().union(*(aliases(key) for key in json.loads(metadata_bytes)))
    supplemental = []
    for kind in ('TV','VAD'):
        path = Path('/fsx/sfr/data/multimodal/online_video_language/ProactiveVideoQA')/kind/'anno.json'
        data = path.read_bytes()
        ids = set().union(*(aliases(v) for v in identities(json.loads(data))))
        if ids & all_source_aliases:
            raise ValueError(f'{kind} overlaps source metadata; rebuild exclusions before launch')
        supplemental.append({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'source_metadata_overlap':0})
    write_json(BASE/'shared/manifests/supplemental_exclusions.json', {'metadata_sha256':source_summary['metadata_sha256'],'benchmarks':supplemental})
    total, milestones = estimate_steps(seconds_per_step,audit['presentations'])
    paired = BASE/'plans/audit.json'
    paired_initial = BASE/'shared-initial-seed42/step-000000'
    paired_identity = None
    if paired.exists() and paired_initial.exists():
        other = json.loads(paired.read_text())
        assert other['source_sha256'] == audit['source_sha256']
        assert other['eligible_rows'] == audit['presentations']
        for label, fraction in other['target'].items():
            assert abs(audit['class_proportions'][label]-fraction) < 1e-12
        for item in audit['exclusions']:
            assert other['exclusion_files'][item['path']] == item['sha256']
        for filename in ('controller_adapter/adapter_model.safetensors','pointer_head.pt'):
            assert hashlib.sha256((paired_initial/filename).read_bytes()).digest() == hashlib.sha256((BASE/'shared/initial-seed42/step-000000'/filename).read_bytes()).digest()
        index = np.load(BASE/'plans/representative.npy')
        index = index[np.argsort(index[:,0])]
        assert len(index) == audit['presentations'] and np.all(np.diff(index[:,0]) > 0)
        cursor, offset = 0, 0
        pool_hash = hashlib.sha256()
        with Path(other['source']).open('rb') as handle:
            for line in handle:
                if cursor < len(index) and offset == int(index[cursor,0]):
                    assert len(line) == int(index[cursor,1])
                    pool_hash.update(line)
                    cursor += 1
                offset += len(line)
        assert cursor == len(index) and pool_hash.hexdigest() == audit['eligible_sha256']
        paired_identity = 'exact full A/B row multiset equality verified by source-ordered byte hash of representative offsets; byte-identical initial LoRA/head'
    counts, cumulative, current_update = Counter(), Counter(), Counter()
    windows, absence, longest = [], Counter(), Counter()
    update_min, update_max = Counter({k:128 for k in ('Hold','Response','Silence')}), Counter()
    label_pattern = re.compile(rb'"label": "(Hold|Response|Silence)"')
    with (BASE/'shared/manifests/control.jsonl').open('rb') as handle:
        for number, line in enumerate(handle,1):
            label = label_pattern.search(line)[1].decode()
            counts[label] += 1
            cumulative[label] += 1
            current_update[label] += 1
            if number % 128 == 0:
                for key in ('Hold','Response','Silence'):
                    update_min[key] = min(update_min[key],current_update[key])
                    update_max[key] = max(update_max[key],current_update[key])
                    absence[key] = 0 if current_update[key] else absence[key]+1
                    longest[key] = max(longest[key],absence[key])
                current_update.clear()
            if number % 4096 == 0:
                windows.append({'end_row':number,'labels':dict(counts),'cumulative_labels':dict(cumulative)})
                counts.clear()
    write_json(BASE/'shared/manifests/control_exposure_audit.json', {'windows_4096':windows,
        'per_update_min':dict(update_min),'per_update_max':dict(update_max),
        'longest_updates_without_class':dict(longest),'terminal_window_labels':dict(counts)})
    report = {'passed':True, 'resume_comparison':'exact loaded LoRA/head/optimizer/progress/RNG state verified on all eight ranks; subsequent update advances state correctly',
              'numerical_reproducibility_limitation': 'independent uninterrupted repetitions differ in adapter updates despite matching RNG; resumed-update differences are within that measured numerical baseline',
              'uninterrupted_repeat_difference':baseline,'resumed_update_difference':resumed_difference,
              'training_elapsed_persists':True, 'measured_seconds_per_step':seconds_per_step,
              'raw_probe_seconds_per_step':raw_seconds_per_step,
              'context_cost_anchors':anchors,
              'step_estimation':'context-stratified probe costs weighted by frozen full-pool context frequencies; approximately twelve hours, with cache/order variation',
              'paired_artifact_checks':paired_identity,
              'presentations_per_second':128/seconds_per_step,'summaries':summaries}
    gpu_peaks = {}
    for line in (out/'logs/preflight_gpu.csv').read_text().splitlines():
        _, rank, used, capacity = [v.strip() for v in line.split(',')]
        fraction = float(used)/float(capacity)
        gpu_peaks[rank] = max(gpu_peaks.get(rank,0), fraction)
    if len(gpu_peaks) != 8 or max(gpu_peaks.values()) > 0.9:
        raise ValueError('measured total device memory does not retain 10% headroom')
    report['nvml_peak_memory_fraction'] = gpu_peaks
    report['package_versions'] = {}
    for package in ('torch','transformers','peft','triton','fla-core','flash-linear-attention'):
        try:
            report['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['package_versions'][package] = None
    write_json(out/'preflight/report.json',report)
    paths = [ROOT/'scripts'/name for name in ('train_jevany_2b_batched.py','train_jevany_streaming.py','run_qwen35_control.py',
             'run_qwen35_control_command.sh','eval_jevany_streaming.py','prepare_qwen35_control.py','finalize_qwen35_control_preflight.py',
             'score_exp10_web_ego.py','merge_jevany_benchmark.py','launch_qwen35_control.py')]
    from pace_clm.stream_gemma_adapter import validate_promoted_contract
    generator = validate_promoted_contract(SG)
    generator_path = Path(generator['adapter'])/'adapter_model.safetensors'
    generator_hash = hashlib.sha256()
    with generator_path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            generator_hash.update(block)
    plan = {'preflight_passed':True, 'budget_policy':'fixed steps estimated from preflight; user amendment 2026-11-05; no time-based stop',
            'estimated_training_hours':total*seconds_per_step/3600,'measured_seconds_per_step':seconds_per_step,
            'total_steps':total,'milestones':milestones,'global_batch':128,'seed':42,
            'manifest':str(BASE/'shared/manifests/control.jsonl'),'manifest_sha256':audit['control_sha256'],
            'eligible_sha256':audit['eligible_sha256'], 'initial_checkpoint':str(BASE/'shared/initial-seed42/step-000000'),
            'backend':'nccl', 'physical_chunk_per_gpu':8,
            'web_gold_sha256': next(item['sha256'] for item in audit['exclusions'] if item['path'].endswith('proactive-videoqa-web/anno.json')),
            'generator_adapter_sha256':generator_hash.hexdigest(),
            'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    if (out/'launch_plan.json').exists():
        raise FileExistsError('launch plan is already frozen')
    write_json(out/'launch_plan.json',plan)
    print(json.dumps(plan,indent=2))


if __name__ == '__main__':
    main()
