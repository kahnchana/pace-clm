#!/usr/bin/env python3
"""Development-only resident judge harness; never collects or takes an update."""
import importlib
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from stream_gemma.inference.qwen_stream import QwenRuntime


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--require-native',action='store_true');parser.add_argument('--output',type=Path,default=root/'outputs/jevany-2b-rl/judge-lab');args=parser.parse_args()
    out=args.output
    out.mkdir(exist_ok=True)
    native_counts={}
    if args.require_native:
        from jevany_rl_rpc import require_native_grouped_mm
        native_counts=require_native_grouped_mm()
    runtime=QwenRuntime.load(checkpoint=str(root.parent/'stream-gemma/outputs/lora-qwen-w1-long-neg'))
    if args.require_native:
        from pace_clm.rl_inputs import install_cached_vision_inputs
        install_cached_vision_inputs(runtime)
    (out/'ready.json').write_text(json.dumps({'ready':True,'optimizer_steps':0})+'\n')
    seen=None
    while True:
        path=out/'request.json'
        if not path.exists():time.sleep(1);continue
        request=json.loads(path.read_text())
        if request['sequence']==seen:time.sleep(1);continue
        seen=request['sequence']
        if request['action']=='stop':break
        try:
            if request['action']=='generator':
                import jevany_rl_generator_probe
                importlib.reload(jevany_rl_generator_probe)
                result=jevany_rl_generator_probe.run(runtime,out,request)
                (out/f'result-{seen}.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result),flush=True)
                torch.cuda.empty_cache()
                continue
            import jevany_rl_judge
            importlib.reload(jevany_rl_judge)
            judge=jevany_rl_judge.TransformersLocalJudge(runtime.device,runtime=runtime)
            started=time.monotonic()
            calibration=judge.calibrate(out/f'calibration-{seen}.json')
            result={'passed':True,'version':jevany_rl_judge.VERSION,'calibration':[r['score'] for r in calibration]}
            if request.get('generator_comparison'):
                from stream_gemma.grpo.judge import JudgeRequest
                compared=json.loads(Path(request['generator_comparison']).read_text())
                pairs=[JudgeRequest('semantic',r['row']['question'],r['batched_text'],r['single_text']) for r in compared if r['batched_text'].strip()!=r['single_text'].strip()]
                result['generator_semantic_scores']=[r['score'] for r in judge.score_many(pairs)]
            if request.get('traces'):
                from run_jevany_rl import judge_requests,apply_scores
                traces=json.loads(Path(request['traces']).read_text())
                payload,indexes=judge_requests(traces)
                scored=judge.score_cached(payload,out/'cache.sqlite',out/f'scores-{seen}.jsonl')
                result['judgments']=apply_scores(traces,indexes,scored)
                result['rewards']=[t['reward'] for t in traces]
            result['seconds']=time.monotonic()-started
        except Exception:
            result={'passed':False,'error':traceback.format_exc()}
        result.update(native_counts)
        (out/f'result-{seen}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        torch.cuda.empty_cache()


if __name__=='__main__':main()
