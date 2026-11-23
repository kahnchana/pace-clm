#!/usr/bin/env python3
"""Read-only compact status for the long-running two-node SFT workflow."""
import argparse
import json
from pathlib import Path


def status(run):
    def read(name):
        path=run/name
        return json.loads(path.read_text()) if path.exists() else {}
    training=read('training-status.json');monitor=read('monitor-status.json')
    result={k:v for k,v in training.items() if k in (
        'step','training_seconds','source_presentations','estimated_total_steps','last_step_seconds')}
    result['evaluation']=read('eval-supervisor-status.json')
    result['completed_evaluations']=monitor.get('completed_evaluations',[])
    result['training_done']=(run/'training-complete.json').exists()
    result['done']=(run/'complete.json').exists()
    result['attention']=[name for name,value in monitor.get('processes',{}).items()
        if not value.get('done') and not value.get('launcher_alive') and value.get('restarts',0)>=3]
    if result['done']:result['report']=(run/'report.md').read_text()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(status(parser.parse_args().run.resolve())))
