#!/usr/bin/env python3
"""Freeze whole trusted training intervals; never infer negative labels in tails."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from prepare_jevany_rl import ROOT, SG, resolved
from prepare_jevany_decisions import make_args, sha256
from prepare_qwen35_control import aliases, exclusions
from stream_gemma.data import _conversation_and_range, _parse_record, _paint_kwargs
from stream_gemma.episode import flatten_stream_conversation, paint_v24_states


def episode(question, text, meta, path):
    """Preserve legacy event extraction, but bound all decisions to explicit labels."""
    if not text:
        raise ValueError('empty_annotations')
    cursor = 0.
    for start, end, *_ in text:
        if abs(float(start)-cursor)>1e-6 or float(end)<=float(start):
            raise ValueError('annotation_gap_or_overlap')
        cursor = float(end)
    stop = math.floor(min(cursor, float(meta['duration']), int(meta['num_frames']))+1e-9)
    if stop<1:
        raise ValueError('empty_interval')
    bounded = {**meta, 'duration': min(cursor, float(meta['duration']))}
    built = _conversation_and_range(question=question, text_stream=text, metadata=bounded,
                                   source_fps=1, keep_trailing_silence=True)
    points = []
    raw_events = any(str(t[2]).strip() for t in text if len(t)>2)
    if built:
        conversation, interval = built
        painted = paint_v24_states(flatten_stream_conversation(conversation),
                                   **_paint_kwargs(make_args()))
        for event in painted.kept_events:
            time = interval.start+event.start
            if time<0 or time+4>stop:
                raise ValueError('incomplete_event_window')
            points.append({'frame_index':time, 'time_seconds':float(time), 'content':event.content})
    if raw_events and not points:
        raise ValueError('captioned_record_without_retained_events')
    identity = hashlib.sha256(f'{resolved(path)}\n{question}'.encode()).hexdigest()
    return {'span_id':identity[:24], 'video_id':Path(path).stem,
            'video_path':resolved(path), 'question':question, 'decode_start':0,
            'rollout_start':0, 'rollout_stop':stop, 'rollout_frames':stop,
            'prefix_turns':[], 'initial_memory':None, 'gold_points':points,
            'annotation_end_seconds':cursor, 'physical_duration_seconds':float(meta['duration']),
            'quiet_decisions':sum(not any(g['time_seconds']<=i<=g['time_seconds']+4
                                            for g in points) for i in range(stop))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'episodes.jsonl').exists():raise FileExistsError(args.output)
    source=ROOT/'outputs/jevany-j1s/manifests/train.jsonl'
    cohort={resolved(json.loads(line)['video_path']) for line in source.open()}
    names={Path(p).name for p in cohort}
    denied, provenance=exclusions()
    annotations=SG/'datasets/markii/train.jsonl'
    metadata_path=SG/'datasets/markii/metadata.json'
    metadata=json.loads(metadata_path.read_text())
    rows={};removed=Counter(); rejected=[]
    for index,line in enumerate(annotations.open()):
        relative,question,text=_parse_record(json.loads(line))
        meta=metadata.get(str(Path(relative).with_suffix('')))
        if not meta:continue
        path=meta.get('path') or str(SG/'datasets/markii/videos'/relative)
        if Path(path).name not in names or resolved(path) not in cohort:continue
        reason=None
        if aliases(path)&denied:reason='evaluation_exclusion'
        else:
            try:row=episode(question,text,meta,path)
            except ValueError as exc:reason=str(exc)
        if reason:
            removed[reason]+=1
            rejected.append({'record':index,'video_path':path,'question':question,'reason':reason})
            continue
        key=row['span_id']
        if key in rows:
            if rows[key]!=row:raise ValueError('conflicting duplicate video/request annotations')
            removed['identical_duplicate']+=1
        else:rows[key]=row
    selected=sorted(rows.values(),key=lambda r:r['span_id'])
    if not selected:raise ValueError('no eligible episodes')
    target=args.output/'episodes.jsonl'
    target.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
    (args.output/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    lengths=sorted(r['rollout_frames'] for r in selected)
    quiet=[r['rollout_frames'] for r in selected if not r['gold_points']]
    audit={'episodes':len(selected),'videos':len({r['video_path'] for r in selected}),
           'length_seconds':{'min':min(lengths),'median':lengths[len(lengths)//2],
                             'p95':lengths[int(.95*(len(lengths)-1))],'max':max(lengths)},
           'total_decisions':sum(lengths),'gold_events':sum(len(r['gold_points']) for r in selected),
           'quiet_decisions':sum(r['quiet_decisions'] for r in selected),
           'true_event_free_episodes':len(quiet),'event_free_lengths':sorted(quiet),
           'event_free_worst_rewards':[-.25*n for n in sorted(quiet)],
           'removed':dict(removed),'exclusion_sources':provenance,
           'hashes':{str(p):sha256(p) for p in (source,annotations,metadata_path,target)},
           'eligibility':'complete contiguous annotated interval from zero; complete four-second event windows; no length or density filtering',
           'event_free_contract':'unchanged -0.25 * fires; worst-case scale recorded explicitly',
           'unannotated_tails_are_negatives':False}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('hashes','exclusion_sources','event_free_lengths','event_free_worst_rewards')}),flush=True)


if __name__=='__main__':main()
