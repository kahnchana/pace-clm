"""Scheduler checks use temporary artifacts and never launch GPU jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('latest_eval', Path(__file__).resolve().parents[1] / 'scripts/watch_pace_sft_latest_eval.py')
eval_watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_watch)


class LatestEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.template = dict(source_hashes={}, population={'web': 500, 'ego': 326, 'estp_bench_explicit': 200}, implementation='validated', hard_time_limit=False)

    def commit(self, step):
        cp = self.run / 'train' / f'step-{step:06d}'
        cp.mkdir(parents=True)
        eval_watch.write(cp / 'commit.json', dict(step=step))
        return cp

    def finish_eval(self, selection, **changes):
        out = self.run / 'evaluation' / f"fast-step-{selection['step']:06d}"
        dispatch = dict(self.template, **selection)
        dispatch.update(changes)
        eval_watch.write(out / 'dispatch.json', dispatch)
        eval_watch.write(out / 'complete.json', dict(step=selection['step'], scores=dict(web=50, ego=30, estp_bench_explicit_200=20)))

    def test_newest_committed_only_skips_backlog_and_partial_checkpoint(self):
        self.commit(4550)
        self.commit(5223)
        (self.run / 'train/step-006000').mkdir()
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['step'], 5223)

    def test_already_evaluated_waits_then_selects_new_checkpoint(self):
        self.commit(4550)
        self.finish_eval(eval_watch.latest(self.run))
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['state'], 'waiting_for_new_checkpoint')
        self.commit(5223)
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['step'], 5223)
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['state'], 'evaluate')

    def test_different_protocol_does_not_count_as_completed(self):
        self.commit(4550)
        self.finish_eval(eval_watch.latest(self.run), population={'estp': 851})
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['state'], 'evaluate')

    def test_training_complete_stops_only_after_final_evaluation(self):
        self.commit(11375)
        eval_watch.write(self.run / 'training-complete.json', dict(step=11375))
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['state'], 'evaluate')
        self.finish_eval(eval_watch.latest(self.run))
        self.assertEqual(eval_watch.decision(self.run, self.template, self.base)['state'], 'complete')

    def test_control_watcher_is_not_mistaken_for_gpu_evaluation(self):
        control = self.run / 'evaluation/continuous-latest'
        eval_watch.write(control / 'launch.json', dict(pid=123))
        with patch.object(eval_watch, 'BASE', self.base), patch.object(eval_watch, 'alive', return_value=True):
            self.assertIsNone(eval_watch.other_gpu_job())
            suite = self.run / 'evaluation/fast-step-005223'
            eval_watch.write(suite / 'launch.json', dict(pid=456))
            eval_watch.write(suite / 'dispatch.json', dict(step=5223))
            self.assertEqual(eval_watch.other_gpu_job(), str(suite))

    def test_retry_budget_survives_scheduler_restart(self):
        out = self.run / 'evaluation/fast-step-005223'
        eval_watch.write(out / 'continuous-attempts.json', dict(attempts=4))
        with patch.object(eval_watch.subprocess, 'run') as probe, patch.object(eval_watch.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'attempts exhausted'):
                eval_watch.supervise(out, self.run, dict(step=5223), 4)
            probe.assert_not_called()
            launch.assert_not_called()

    def test_live_pinned_job_is_waited_for_without_launching_new_job(self):
        out = self.run / 'evaluation/fast-step-005223'
        eval_watch.write(out / 'launch.json', dict(pid=123))
        with patch.object(eval_watch, 'alive', return_value=True), patch.object(eval_watch, 'pause'), patch.object(eval_watch.subprocess, 'Popen') as launch:
            self.assertFalse(eval_watch.supervise(out, self.run, dict(step=5223, checkpoint='pinned'), 4))
            launch.assert_not_called()
            self.assertEqual(eval_watch.read(self.run / 'status.json')['checkpoint'], 'pinned')


if __name__ == '__main__':
    unittest.main()
