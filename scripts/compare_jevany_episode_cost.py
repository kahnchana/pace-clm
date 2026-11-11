#!/usr/bin/env python3
"""Predeclare nearest saved cost controls, then report them after the pilot."""
import argparse
import json
from pathlib import Path
import time

from jevany_rl_reward import ROOT, sha, write
from report_jevany_e2 import paired_bootstrap


def read(path):return json.loads(Path(path).read_text())


def prepare(run):
    anchor=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1'
    reports=sorted(anchor.glob('batch-*/report.json'))
    controls=[]
    for step in (8,16,24,32):
        source=anchor/f'evaluation/web-step{step}'
        selected=reports[:step]
        if len(selected)!=step:raise ValueError('incomplete historical cost records')
        controls.append({'step':step,'source':str(source),'sampled_decisions':sum(read(p)['decisions'] for p in selected),
                         'core_gpu_hours':sum(read(p)['gpu_hours'] for p in selected),
                         'hashes':{str(p):sha(p) for p in [*selected,source/'predictions.jsonl',source/'pauc_llm_scores-gpt-4.1.json']}})
    spec={'workflow_freeze_sha256':sha(run/'workflow-freeze.json'),'script_sha256':sha(Path(__file__)),
          'controls':controls,'selection':'nearest historical core allocated GPU-hours among the four saved full-WEB anchored checkpoints; ties favor earlier checkpoint',
          'limits':'Approximate matching to saved checkpoints; signed cost differences disclosed. Core costs exclude initialization, audit, evaluation and restarts.'}
    target=run/'cost-comparison-freeze.json'
    if target.exists() and read(target)!=spec:raise ValueError('cost comparison contract changed')
    if not target.exists():write(target,spec)
    return spec


def report(run,spec):
    for control in spec['controls']:
        for path,digest in control['hashes'].items():
            if sha(path)!=digest:raise ValueError('historical cost control changed: '+path)
    curves=read(run/'learning-curves.json') if (run/'learning-curves.json').exists() else []
    comparisons=[]
    for evaluation in sorted((run/'evaluation').glob('web-step*')):
        if not (evaluation/'pauc_f_summary-gpt-4.1-v1.json').exists():continue
        step=int(evaluation.name.removeprefix('web-step'))
        actual=next((r for r in curves if r['optimizer_step']==step),None)
        if actual is None:continue
        control=min(spec['controls'],key=lambda c:(abs(c['core_gpu_hours']-actual['core_gpu_hours']),c['step']))
        interval=paired_bootstrap(evaluation,Path(control['source']))
        comparisons.append({'candidate_step':step,'candidate_core_gpu_hours':actual['core_gpu_hours'],
                            'candidate_sampled_decisions':actual['sampled_decisions'],
                            'reference_step':control['step'],'reference_core_gpu_hours':control['core_gpu_hours'],
                            'reference_sampled_decisions':control['sampled_decisions'],
                            'gpu_hour_difference':actual['core_gpu_hours']-control['core_gpu_hours'],
                            'paired_bootstrap':interval})
    write(run/'gpu-hour-comparisons.json',{'spec_sha256':sha(run/'cost-comparison-freeze.json'),
          'comparisons':comparisons,'limits':spec['limits']})
    print(json.dumps({'gpu_hour_comparisons':len(comparisons)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();spec=prepare(args.run)
    if args.prepare_only:print('cost comparison controls frozen',flush=True);return
    if args.watch:
        while True:
            path=args.run/'workflow-status.json'
            state=read(path)['state'] if path.exists() else None
            if state in ('complete','stopped_by_coverage_gate','failed'):
                # Terminal coverage/complete status precedes the final report call.
                # A completed checkpoint evaluation is reported during the workflow.
                result=args.run/'result.json'
                if state=='failed' or (result.exists() and read(result)['workflow']['state']==state):break
            time.sleep(30)
    report(args.run,spec)


if __name__=='__main__':main()
