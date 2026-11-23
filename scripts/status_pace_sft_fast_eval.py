#!/usr/bin/env python3
"""Read-only lightweight progress for a pinned faster evaluation."""
import argparse
import json
import re
import time
from pathlib import Path


def status(out):
    def read(name):
        p=out/name
        return json.loads(p.read_text()) if p.exists() else {}
    phase=read('status.json');start=read('run-start.json');value=dict(step=read('dispatch.json').get('step'),phase=phase.get('phase','starting'),elapsed_minutes=(time.time()-start.get('started_epoch',time.time()))/60)
    log=out/'inference.log';text=log.read_text(errors='replace') if log.exists() else ''
    ids=set(re.findall(r'\[rank \d+\] \d+/\d+ (\S+) fires=\d+ elapsed=',text));value['completed_episodes']={s:sum(q.startswith(prefix) for q in ids) for s,prefix in [('web','web:'),('ego','ego:'),('estp200','estp-eval:')]}
    value['loaded_generator_workers']=len(set(re.findall(r'"direct_gpu_moe_load_seconds"[^\n]+"local_rank": (\d+)',text)))
    value['scored_suites']=[s for s in ['web','ego','estp-bench-explicit-200'] if (out/s/'result.json').exists()]
    value['attention']=read('needs-attention.json');value['complete']=(out/'complete.json').exists()
    if value['complete']:value['summary']=read('complete.json')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args();print(json.dumps(status(a.output_dir)))
