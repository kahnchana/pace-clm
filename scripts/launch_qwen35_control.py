#!/usr/bin/env python3
"""Launch the preflighted control with a private snapshot of its Python code."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

from run_qwen35_control import BASE, ROOT, write_json


def main():
    if socket.gethostname() != 'ip-10-3-120-167':
        raise RuntimeError('this launcher is for the control node')
    out = BASE/'control-seed42'
    plan = json.loads((out/'launch_plan.json').read_text())
    if not plan['preflight_passed']:
        raise RuntimeError('preflight has not passed')
    if (out/'orchestrator.pid').exists():
        raise FileExistsError('control was already launched; inspect status before restarting')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:
        raise RuntimeError('GPUs are occupied')
    runtime = out/'runtime'
    code = runtime/'pace-clm'
    code.mkdir(parents=True)
    shutil.copytree(ROOT/'scripts',code/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    shutil.copytree(ROOT/'configs',code/'configs')
    (code/'outputs').symlink_to(ROOT/'outputs',target_is_directory=True)
    (runtime/'stream-gemma').symlink_to(ROOT.parent/'stream-gemma',target_is_directory=True)
    for origin, target in ((ROOT/'.env',code/'.env'),(ROOT.parent/'.env',runtime/'.env')):
        if origin.exists():
            target.symlink_to(origin)
    hashes = {str(path.relative_to(code)):hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ('scripts','src','configs') for path in (code/folder).rglob('*') if path.is_file()}
    write_json(out/'code_snapshot.json',{'code_root':str(code),'sha256':hashes,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    env = dict(os.environ,PACE_CONTROL_CODE_ROOT=str(code))
    with (out/'logs/orchestrator.log').open('a') as log:
        process = subprocess.Popen(['bash',str(code/'scripts/run_qwen35_control_command.sh'),'python',
            str(code/'scripts/run_qwen35_control.py'),'--output',str(out)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'orchestrator.pid').write_text(str(process.pid)+'\n')
    print(json.dumps({'pid':process.pid,'output':str(out),'plan':plan},indent=2))


if __name__ == '__main__':
    main()
