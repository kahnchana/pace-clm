"""Durable checkpoint jobs and bounded lead for asynchronous evaluation."""
import json
from pathlib import Path
import time
from jevany_rl_reward import write


def read(path):return json.loads(Path(path).read_text())


def enqueue(run,name,checkpoint,kind,step=0,exposure=0,priority=10):
    run=Path(run);directory=run/'queue';directory.mkdir(exist_ok=True)
    job={'name':name,'checkpoint':str(checkpoint),'kind':kind,'step':step,'exposure':exposure,'priority':priority}
    path=directory/(name+'.json')
    if path.exists() and read(path)!=job:raise ValueError('evaluation job identity changed')
    if not path.exists():write(path,job)


def publish_checkpoint(run,step,checkpoint,report):
    run=Path(run);plan=read(run/'workflow-freeze.json');exposure=report['episode_schedule']['exposure_after']
    if step==1:enqueue(run,'gate-step1',checkpoint,'gate',step,exposure,1)
    if step in plan['full_steps']:enqueue(run,f'full-step{step}',checkpoint,'full',step,exposure,2)
    elif step in plan['fast_steps']:enqueue(run,f'fast-step{step}',checkpoint,'fast',step,exposure,5)


def before_batch(run,version):
    run=Path(run)
    while True:
        if (run/'stop-request.json').exists():return False
        state=run/'evaluation-status.json'
        if state.exists() and read(state)['state']=='failed':raise RuntimeError('asynchronous evaluator failed; training halted')
        if version==0:return True
        # Do not continue the pilot before its update/restore and remote gate.
        if not (run/'gate-passed.json').exists():time.sleep(5);continue
        plan=read(run/'workflow-freeze.json')
        exposure=plan['schedule'][version-1]['exposure_after']
        pending=[read(p) for p in (run/'queue').glob('full-step*.json')
                 if not (run/'evaluation'/p.stem/'result.json').exists()]
        if pending and exposure-min(j['exposure'] for j in pending)>=10000:
            write(run/'training-wait.json',{'reason':'bounded evaluation lag','exposure':exposure})
            time.sleep(10);continue
        return True
