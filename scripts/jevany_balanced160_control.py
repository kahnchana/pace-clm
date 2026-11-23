"""Publication and first-update correctness gate for the balanced continuation."""
import time
from pathlib import Path

from jevany_mixed_control import enqueue
from prepare_jevany_estp128 import read


def publish_checkpoint(run, step, checkpoint, report):
    run = Path(run)
    exposure = report['episode_schedule']['exposure_after']
    if step == 129:
        enqueue(run, 'parity-step129', checkpoint, 'parity', step, exposure, 0)
    if step in read(run/'workflow-freeze.json')['full_steps']:
        enqueue(run, f'full-step{step}', checkpoint, 'full', step, exposure, 2)


def before_batch(run, version):
    run = Path(run)
    while version > 128 and not (run/'continuation-gate-passed.json').exists():
        time.sleep(10)
    return True
