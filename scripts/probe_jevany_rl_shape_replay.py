#!/usr/bin/env python3
"""Check exact likelihood replay when collection uses several physical shapes."""
import copy,json,math,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from benchmark_qwen35_2b_training import batched_logits
from jevany_rl_workers import prepare_chunks
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import PointerHead
from pace_clm.qwen_controller_runtime import load_2b_controller
from pace_clm.rl_inputs import install_cached_vision_inputs


def main():
 torch.set_num_threads(1)
 root=Path('outputs/jevany-2b-rl');checkpoint=Path('outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243')
 traces=json.loads((root/'profile-6p2-v7/batch-000000/scored.json').read_text())
 steps=[s for t in traces for s in t['steps']];steps.sort(key=lambda s:-sum(-math.exp(v)*v for v in s['old_logp']))
 templates=[s['row'] for s in steps[:8]];length=max(r['padded_length'] for r in templates)
 runtime,_=load_2b_controller(checkpoint=str(checkpoint/'controller_adapter'));install_cached_vision_inputs(runtime)
 head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
 head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
 frames=EpisodeFrames(runtime,response_blind=False,cache_frames=1024,defer_vision=True);frames.physical_batch=24
 rows=[];old={};shapes=[]
 with ThreadPoolExecutor(max_workers=1) as pool,torch.no_grad():
  for count in (1,5,9,17,22):
   chunk=[]
   for i in range(count):
    row=copy.deepcopy(templates[i%len(templates)]);row['padded_length']=length;row.pop('physical_batch',None);row['probe_index']=len(rows)+i;chunk.append(row)
   for _,samples,packed in prepare_chunks(chunk,frames,runtime,24,pool):
    values=batched_logits(runtime,head,samples,packed)[0].float().log_softmax(-1).cpu()
    physical=packed[0]['input_ids'].shape[0];shapes.append(physical)
    for row,value in zip(chunk,values,strict=True):row['physical_batch']=physical;old[row['probe_index']]=value
   rows.extend(chunk)
  rows.sort(key=lambda r:(r['padded_length'],r['physical_batch'],-r['probe_index']))
  maximum=0.
  for start,samples,packed in prepare_chunks(rows,frames,runtime,24,pool):
   values=batched_logits(runtime,head,samples,packed)[0].float().log_softmax(-1).cpu()
   for row,value in zip(rows[start:start+len(samples)],values,strict=True):maximum=max(maximum,float((old[row['probe_index']]-value).abs().max()))
 result={'physical_shapes':shapes,'observations':len(rows),'max_logp_error':maximum,'optimizer_steps':0,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
 (root/'shape-replay-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert maximum==0.


if __name__=='__main__':main()
