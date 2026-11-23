#!/usr/bin/env python3
"""No-update A/B/A profile against complete saved native trajectories."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import torch
import eval_pace_sft_fast_native as evaluator
from pace_clm.fast_eval_runtime import DirectGPUQwenRuntime
from prepare_pace_estp_sft import ROOT,read,write,sha

OUT=ROOT/'outputs/jevany-qwen35-2b-pace-estp-sft/eval-speed-design-v1'
CHECKPOINT=OUT.parent/'base-50-50-b384-prefetch-v2-16h/train/step-000143'


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    if socket.gethostname()!='ip-10-3-120-167' or os.environ.get('SLURM_JOB_ID')!='38177':raise RuntimeError('evaluation allocation only')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    sources=[ROOT/'scripts/eval_pace_sft_fast_native.py',ROOT/'src/pace_clm/fast_eval_runtime.py',Path(__file__).resolve()]
    write(OUT/'pilot/source-freeze.json',{str(p):sha(p) for p in sources})
    controllers={};generators={};original_methods={};original_dense=evaluator.load_dense_controller;original_generator=DirectGPUQwenRuntime.load
    phase={'name':'initializing'};stop=threading.Event()
    def sample():
        with (OUT/'pilot/gpu-util.jsonl').open('a',buffering=1) as handle:
            while not stop.is_set():
                r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if r.returncode==0:handle.write(json.dumps({'epoch':time.time(),'phase':phase['name'],'values':[float(s) for s in r.stdout.strip().split(',')]})+'\n')
                stop.wait(2)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    def dense(**kw):
        key=json.dumps(kw,sort_keys=True)
        if key not in controllers:
            value=original_dense(**kw);runtime=value[0];controllers[key]=value
            original_methods[key]=(runtime.assemble_turn,runtime._cached_image_features)
        runtime=controllers[key][0];runtime.assemble_turn,runtime._cached_image_features=original_methods[key];runtime.rl_cached_vision_inputs=False
        return controllers[key]
    def generator(cls,**kw):
        key=json.dumps(kw,sort_keys=True)
        if key not in generators:
            before=phase['name'];phase['name']='direct_gpu_loading';generators[key]=original_generator(**kw);phase['name']=before
        return generators[key]
    evaluator.load_dense_controller=dense;DirectGPUQwenRuntime.load=classmethod(generator)
    reference={p['question_id']:p for p in rows(OUT/'pilot/reference.jsonl')};results={}
    try:
        for name,cached in [('native_a',False),('cached_b',True),('native_a2',False)]:
            phase['name']=name;directory=OUT/'pilot'/name
            sys.argv=['eval_pace_sft_fast_native.py','--controller',str(CHECKPOINT),'--validation-manifest',str(OUT/'pilot/manifest.jsonl'),
                '--selection-ids',str(OUT/'pilot/selection.json'),'--output-dir',str(directory),'--response-margin','0',
                '--answer-model','promoted','--max-new-tokens','96','--record-action-logits','--direct-gpu-load']+(['--cached-controller-inputs'] if cached else [])
            began=time.monotonic();evaluator.main();duration=time.monotonic()-began
            predictions=rows(directory/'predictions.rank00.jsonl');cases=[]
            if {p['question_id'] for p in predictions}!=set(reference):raise RuntimeError('pilot population changed')
            for p in predictions:
                old=reference[p['question_id']];a=p['jevany_action_trace'];b=old['jevany_action_trace']
                if len(a)!=len(b):raise RuntimeError('pilot trace length mismatch')
                errors=[float((torch.tensor(x['logits']).softmax(-1)-torch.tensor(y['logits']).softmax(-1)).abs().max()) for x,y in zip(a,b)]
                actions=sum(x['action']!=y['action'] for x,y in zip(a,b));answers=p['model_response_list']==old['model_response_list']
                info=dict(question_id=p['question_id'],decisions=len(a),action_mismatches=actions,answers_identical=answers,max_probability_error=max(errors),
                    episode_seconds=p['jevany_episode_seconds'],controller_seconds=p['jevany_decision_seconds'],generator_seconds=p['jevany_generation_seconds'])
                cases.append(info)
                if actions or not answers or max(errors)>.005:
                    write(OUT/'pilot/parity-failed.json',dict(phase=name,case=info));raise RuntimeError('native policy/generator parity failed: '+json.dumps(info))
            results[name]=dict(wall_seconds=duration,cases=cases,episode_seconds=sum(p['jevany_episode_seconds'] for p in predictions),
                controller_seconds=sum(p['jevany_decision_seconds'] for p in predictions),generator_seconds=sum(p['jevany_generation_seconds'] for p in predictions))
            write(OUT/'pilot/progress.json',dict(phase=name,passed=True,results=results));print(json.dumps({'phase_complete':name,'results':results[name]}),flush=True)
        result=dict(passed=True,phases=results,direct_gpu_load_seconds=[r.direct_load_seconds for r in generators.values()],
            cached_speedup_vs_warm_native=results['native_a2']['episode_seconds']/results['cached_b']['episode_seconds'],
            cached_controller_speedup_vs_warm_native=results['native_a2']['controller_seconds']/results['cached_b']['controller_seconds'])
        write(OUT/'pilot/result.json',result);print(json.dumps(result),flush=True)
    finally:stop.set();sampler.join(timeout=5)


if __name__=='__main__':main()
