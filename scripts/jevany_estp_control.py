"""Checkpoint publication for the score-independent step-128 continuation."""
import time
from pathlib import Path

from jevany_mixed_control import enqueue, read
from jevany_rl_reward import write


def publish_checkpoint(run, step, checkpoint, report):
    run = Path(run)
    plan = read(run / 'workflow-freeze.json')
    exposure = report['episode_schedule']['exposure_after']
    if step == 33:
        enqueue(run, 'parity-step33', checkpoint, 'parity', step, exposure, 0)
    if step == 128:
        enqueue(run, 'full-step128', checkpoint, 'final', step, exposure, 2)
    elif step in plan['full_steps']:
        enqueue(run, f'full-step{step}', checkpoint, 'full', step, exposure, 2)


def before_batch(run, version):
    run = Path(run)
    while version > 32 and not (run / 'continuation-gate-passed.json').exists():
        write(run / 'training-wait.json', {'reason': 'step-33 correctness gate', 'version': version})
        time.sleep(10)
    return True
