#!/usr/bin/env python3
"""Prepare saved RL ESTP streams for the unchanged stream-gemma explicit scorer.

Default is CPU-only preflight: no inference and no judge calls. --judge requires
an explicit user authorization for this additional post-hoc scoring pass.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from prepare_jevany_estp128 import ROOT, SG, read, rows, sha, write, dump
from zs_infer import estp_pauc_f as estp


def prepare(source_eval, output):
    result=read(source_eval/'result.json')
    if not result.get('estp_bench_official_estp_f1',{}).get('headline_eligible'):
        raise ValueError('requires a completed full ESTP-Bench SQ source result')
    source=source_eval/'estp_bench/predictions.jsonl'
    manifest=estp.build_manifest(estp.DEFAULT_ANNOTATION,estp.DEFAULT_VIDEO_ROOT,horizon='clip')
    records=manifest['records']
    expected={r['question_id'] for r in records}
    by_id={}
    for prediction in rows(source):
        qid=f"{prediction['video_id']}/{prediction['clip_id']}/{prediction['qa_index']}"
        if qid in by_id:
            raise ValueError('duplicate source question identity')
        by_id[qid]=prediction
    if not expected<=by_id.keys():
        raise ValueError('missing explicit prediction cases')
    predictions=[]
    for record in records:
        original=by_id[record['question_id']]
        # Saved ESTP-F1 streams already contain absolute annotation timestamps.
        # Do not add the clip origin twice or shift/widen gold windows.
        fires=[dict(role='assistant',content=t['content'],absolute_time=float(t['time']),
                    time=float(t['time'])-record['eval_start_time'],fire_id=i)
               for i,t in enumerate(original['estp_turns']) if t.get('role')=='assistant']
        if any(not f['content'].strip() for f in fires):
            raise ValueError('empty saved answer')
        predictions.append(dict(question_id=record['question_id'],task_type=record['task_type'],
            eval_start_time=record['eval_start_time'],eval_end_time=record['eval_end_time'],
            model_response_list=fires))
    fingerprint=dict(source_result=str(source_eval/'result.json'),source_result_sha256=sha(source_eval/'result.json'),
        source_predictions=str(source),source_predictions_sha256=sha(source),
        estp_pauc_scorer_sha256=sha(Path(estp.__file__)),official_scorer_sha256=sha(estp.OFFICIAL_SCORER),
        annotation_sha256=manifest['summary']['annotation_sha256'],
        adapter_sha256=sha(Path(__file__)),protocol=estp.SCORER_VERSION,
        horizon='clip',match_window='strict',headline_omega=.5,
        aggregation='unweighted eight-task macro-average',source_job=result['job'])
    output.mkdir(parents=True,exist_ok=True)
    frozen=output/'source-freeze.json'
    if frozen.exists() and read(frozen)!=fingerprint:
        raise ValueError('existing scoring inputs/protocol changed')
    write(frozen,fingerprint)
    estp.write_manifest(manifest,output)
    dump(output/'predictions.jsonl',predictions)
    official_predictions,offsets=estp.official_predictions(predictions)
    assert sum(len(p['model_response_list']) for p in official_predictions)==sum(len(p['model_response_list']) for p in predictions)
    write(output/'estp_pauc_gold.json',estp.official_gold_from_manifest(records))
    dump(output/'predictions_estp_pauc.jsonl',official_predictions)
    if offsets:write(output/'timestamp_offsets.json',offsets)
    official=estp.load_official_scorer()
    meta=estp.build_judge_requests(official,output/'predictions_estp_pauc.jsonl',
        output/'estp_pauc_gold.json',output/'pauc_batch_input-gpt-4.1.jsonl',judge_model='gpt-4.1')
    summary={k:v for k,v in meta.items() if k!='alias'}
    summary.update(cases=len(records),physical_fires=sum(len(p['model_response_list']) for p in predictions),
        gold_turns=manifest['summary']['gold_turns'],
        positive_duration_turns=manifest['summary']['positive_duration_pauc_turns'],
        zero_duration_turns_excluded=manifest['summary']['zero_duration_turns_excluded'],
        new_inference=False,judge_calls_submitted=False,
        estp_f1_judgments_reusable=False,
        rubric_note='PAUC uses cumulative-answer judgments with official 1/2/3 responses mapped to 0/1/2; ESTP-F1 uses a separate 1–5 single-answer rubric.')
    write(output/'preflight.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-eval',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--judge',action='store_true')
    args=parser.parse_args();source=args.source_eval.resolve();output=(args.output or source/'estp_pauc_explicit').resolve()
    print(json.dumps(prepare(source,output),indent=2),flush=True)
    if args.judge:
        from judge_exp10_utility import load_gateway_key
        load_gateway_key()
        result=estp.cmd_judge(SimpleNamespace(manifest=output/'explicit_manifest.jsonl',
            predictions=output/'predictions.jsonl',official_scorer=estp.OFFICIAL_SCORER,pred_dir=output,
            judge_model='gpt-4.1',preflight_only=False,concurrency=16,max_attempts=5,request_timeout=120.))
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
