#!/usr/bin/env python3
"""Three complete-episode correctness cases; not a checkpoint quality sweep."""
import argparse
from pathlib import Path
from prepare_pace_estp_sft import ROOT,read,write
from run_pace_estp_sft import PY,checked,rows
from jevany_mixed_reward import score_predictions


def main(run):
    out=run/'smoke';checkpoint=run/'train/step-000001'
    if not (out/'rollout/predictions.jsonl').exists():
        checked([PY,ROOT/'scripts/eval_jevany_streaming.py','--controller',checkpoint,
            '--validation-manifest',run/'data/evaluation-manifest.jsonl','--selection-ids',out/'selection.json',
            '--output-dir',out/'rollout','--response-margin','0','--answer-model','promoted',
            '--max-new-tokens','96','--record-action-logits'],out/'inference.log')
        checked([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',out/'rollout','--world-size','1'],out/'merge.log')
    predictions=rows(out/'rollout/predictions.jsonl')
    specs={s['span_id']:s for s in rows(run/'data/evaluation-specs.jsonl')}
    assert {p['question_id'] for p in predictions}==set(read(out/'selection.json'))
    for p in predictions:
        trace=p['jevany_action_trace']
        assert len(trace)==specs[p['question_id']]['rollout_frames']
        assert all(t['action']==('Silence','Hold','Response')[max(range(3),key=lambda i:t['logits'][i])] for t in trace)
    gold=[dict(question_id=p['question_id'],conversation=[dict(role='user',content=specs[p['question_id']]['question'])],
        answer=specs[p['question_id']]['gold_answers']) for p in predictions]
    metrics,usage=score_predictions(predictions,gold,out/'scoring',run/'cache/smoke.sqlite')
    write(out/'result.json',dict(passed=True,episodes=len(predictions),raw_argmax_verified=True,
        complete_native_timelines=True,per_episode=metrics,judge_usage=usage,
        purpose='pipeline correctness only, step one; not checkpoint selection'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
