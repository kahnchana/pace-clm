#!/usr/bin/env python3
"""Independent causal streams, batched controller forwards, singleton MoE answers.

    The action is always the learned argmax, except explicitly sampled readiness
    diagnostics. Generator requests retain the existing singleton greedy path.
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from benchmark_qwen35_2b_training import batched_logits, collate, encode_batch_vision
from generate_selected_responses import forced_response
from jevany_mixed_inputs import EpisodeFrames, predecode_batch
from pace_clm.jevany_streaming import ACTIONS,PointerHead,controller_logits,decision_messages,option_text
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl_mixed import observation
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.sft_video import SequentialVideoCache
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from prepare_pace_estp_sft import ROOT,read,write,sha
from stream_gemma.inference.qwen_stream import QwenRuntime
from stream_gemma.v24_memory import normalize_memory_content


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def evaluate(checkpoint,spec_file,output,sampled=False,batch_size=8):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);torch.set_num_threads(1)
    specs=sorted(rows(spec_file),key=lambda s:s['span_id']);output.mkdir(parents=True,exist_ok=True)
    protocol=dict(split='joint-sft-development',question_ids=[s['span_id'] for s in specs],
        controller_sha256=sha(checkpoint/'controller_adapter/adapter_model.safetensors'),
        pointer_sha256=sha(checkpoint/'pointer_head.pt'),specs_sha256=sha(spec_file),response_margin=0,
        controller_policy='sampled raw categorical' if sampled else 'raw learned argmax',
        batch_size=batch_size,history='self-generated',window_frames=32,max_new_tokens=96,
        answer_model='promoted singleton greedy',source_sha256=sha(Path(__file__)))
    protocol['decoder_sha256']=sha(ROOT/'src/pace_clm/sft_video.py')
    protocol['cached_inputs_sha256']=sha(ROOT/'src/pace_clm/rl_inputs.py')
    if (output/'manifest.json').exists() and read(output/'manifest.json')!=protocol:raise ValueError('protocol changed')
    if rank==0:write(output/'manifest.json',protocol)
    shard=output/f'predictions.rank{rank:02d}.jsonl'
    completed={p['question_id'] for p in rows(shard)} if shard.exists() else set()
    assigned=[s for i,s in enumerate(specs) if i%world==rank and s['span_id'] not in completed]
    if not assigned:
        shard.touch(exist_ok=True);return
    runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'))
    install_cached_vision_inputs(runtime)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True));head.eval()
    contract=validate_promoted_contract(ROOT.parent/'stream-gemma')
    generator=QwenRuntime.load(checkpoint=contract['adapter'],model_name=contract['model_name'],
        model_revision=contract['model_revision'],frame_num_tokens=144,generation_prefix_mode='direct')
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
    gen_frames=EpisodeFrames(generator,response_blind=False,cache_frames=128)
    decoder=SequentialVideoCache()
    pending=iter(assigned);active=[];started=time.monotonic();last_progress=started;checked_batches=0;parity=[]
    def refill():
        while len(active)<batch_size:
            try:s=next(pending)
            except StopIteration:break
            rng=torch.Generator(device=runtime.device)
            rng.manual_seed(int(hashlib.sha256(('42:'+s['span_id']).encode()).hexdigest()[:16],16))
            active.append(dict(spec=s,frame=0,history=[],trace=[],responses=[],rng=rng))
    refill()
    with shard.open('a') as handle:
        try:
            while active:
                observations=[observation(s['spec'],s['history'],s['frame']) for s in active]
                cached={key:image for key,(image,_) in frames.frames.items()}
                decoded=decoder.prepare_rows(runtime,observations,cached)
                samples=[frames.prepare(row,prefetched=decoded,cpu_only=True) for row in observations]
                encode_batch_vision(runtime,samples,8)
                packed,visions,lengths=collate(samples,runtime)
                packed={k:v.to(runtime.device) for k,v in packed.items()}
                with torch.no_grad():logits,_=batched_logits(runtime,head,samples,(packed,visions,lengths))
                # Same unpadded observations and cached pixels, independently run
                # through the standard singleton readout before accepting batching.
                if checked_batches<4:
                    batch_errors=[]
                    for index,(inputs,vision,options,decision) in enumerate(samples):
                        inputs={k:v.to(runtime.device) for k,v in inputs.items()}
                        with torch.no_grad():single=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False)
                        error=float((single.float().softmax(-1)-logits[index].float().softmax(-1)).abs().max())
                        if error>.005 or int(single.argmax())!=int(logits[index].argmax()):
                            raise RuntimeError(f'batch/singleton policy mismatch: {error}')
                        batch_errors.append(error)
                    parity.append(batch_errors);checked_batches+=1
                    write(output/f'batch-parity.rank{rank:02d}.json',dict(passed=True,batches=parity))
                del packed,visions
                for index,(state,row) in enumerate(zip(active,observations,strict=True)):
                    values=logits[index].float()
                    action=int(torch.multinomial(values.softmax(-1),1,generator=state['rng'])) if sampled else int(values.argmax())
                    text='';frame=state['frame'];tag=('</Silence>','</Standby>','</Response>')[action]
                    if action==2:
                        _,gvision,_,_=gen_frames.prepare(row)
                        images=gen_frames.images_for(row)
                        messages=decision_messages(row,images)
                        messages[-1]['content']=[part for part in messages[-1]['content'] if part.get('text')!=option_text()]
                        text=forced_response(generator,messages,gvision,96)
                        if not text.strip():raise RuntimeError('empty frozen-generator answer')
                        state['responses'].append(dict(role='assistant',time=state['spec']['timestamps'][frame],content=text))
                    state['trace'].append(dict(action=ACTIONS[action],logits=values.cpu().tolist(),frame_index=frame,
                        input_frames=len(row['turns']),input_tokens=lengths[index]))
                    turn=dict(frame_index=frame,ts=row['turns'][-1]['ts'],emit=tag+(' '+text if text else ''),
                              state=('Silence','Standby','Response')[action])
                    if action==2:turn['memory_content']=normalize_memory_content(text)
                    state['history'].append(turn);state['frame']+=1
                remaining=[]
                for state in active:
                    if state['frame']<state['spec']['rollout_stop']:remaining.append(state);continue
                    result=dict(question_id=state['spec']['span_id'],video_id=state['spec']['video_id'],
                        model_response_list=state['responses'],jevany_action_trace=state['trace'],
                        jevany_actions=dict(Counter(t['action'] for t in state['trace'])),
                        v24_parse={'malformed':0},jevany_peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
                    handle.write(json.dumps(result,sort_keys=True)+'\n');handle.flush()
                    completed.add(state['spec']['span_id'])
                    print(json.dumps(dict(rank=rank,complete=len(completed),assigned=len(assigned),
                        question_id=state['spec']['span_id'],fires=len(state['responses']),seconds=time.monotonic()-started)),flush=True)
                active=remaining;refill()
                if time.monotonic()-last_progress>30:
                    write(output/f'progress.rank{rank:02d}.json',dict(complete=len(completed),assigned=len(assigned),
                        active_frames=[s['frame'] for s in active],seconds=time.monotonic()-started,
                        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3));last_progress=time.monotonic()
        except BaseException:
            write(output/f'incomplete.rank{rank:02d}.json',dict(states=[{k:v for k,v in s.items() if k!='rng'} for s in active]))
            raise
    decoder.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--specs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--sample',action='store_true');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();evaluate(a.checkpoint,a.specs,a.output_dir,a.sample,a.batch_size)
