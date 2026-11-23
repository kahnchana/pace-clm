#!/usr/bin/env python3
"""Prefetched, source-balanced SFT; profile scratch weights before a fresh run."""
import argparse
import faulthandler
import json
import os
import random
import signal
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmark_qwen35_2b_training import batched_logits,collate,encode_batch_vision,predecode_batch
from pace_clm.jevany_streaming import ACTIONS,PointerHead,controller_logits
from pace_clm.qwen_controller_runtime import MODEL_NAME,MODEL_REVISION,load_2b_controller
from pace_clm.rl_inputs import install_cached_vision_inputs
from pace_clm.sft_pace_estp import CyclingPermutation,supervised_row
from prepare_pace_estp_sft import read,write,sha,dump
from train_jevany_streaming import EpisodeFrames,save_checkpoint,sync_gradients


class MixedRows:
    def __init__(self,data):
        self.offsets=np.load(data/'pace-offsets.npy',mmap_mode='r')
        self.pairs=np.load(data/'estp-training-pairs.npy',mmap_mode='r')
        self.episodes=[json.loads(s) for s in (data/'estp-train.jsonl').read_text().splitlines()]
        self.handle=open(read(data/'audit.json')['pace_manifest'],'rb')
        self.samplers=[CyclingPermutation(len(self.offsets),42),CyclingPermutation(len(self.pairs),420042)]

    def row(self,source,position):
        index=self.samplers[source].at(position)
        if source==0:
            offset,length=map(int,self.offsets[index]);self.handle.seek(offset)
            row=json.loads(self.handle.read(length));row['source']='pace'
        else:
            episode,frame=map(int,self.pairs[index]);row=supervised_row(self.episodes[episode],frame)
        return row

    def batch(self,step,rank):
        return [self.row(source,step*192+rank*24+i) for i in range(24) for source in (0,1)]

    def stress(self,rank):
        def cost(row):
            return len(row['turns'])*144+sum(len(t.get('emit',''))//3 for t in row['turns'])+len(row['question'])//3
        result=[]
        for source in (0,1):
            pool=[self.row(source,rank*256+i) for i in range(256)]
            result+=sorted(pool,key=cost,reverse=True)[:24]
        return result


def main(args):
    run=args.run.resolve();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if socket.gethostname()!='ip-10-3-70-177' or world!=8:raise RuntimeError('assigned eight-GPU training node only')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    stack=(run/f'stack.rank{rank}.log').open('a');faulthandler.register(signal.SIGUSR1,file=stack,all_threads=True)
    dist.init_process_group('gloo')
    test=torch.ones(1024);dist.all_reduce(test)
    if not bool((test==8).all()):raise RuntimeError('Gloo preflight failed')
    cfg=read(run/'freeze.json')
    for path,digest in cfg['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen source changed: '+path)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    out=run/'train';out.mkdir(exist_ok=True)
    previous=Path(read(out/'latest.json')['checkpoint']) if args.resume else None
    if not args.resume and (out/'latest.json').exists():raise FileExistsError('training state exists')
    runtime,_=load_2b_controller(checkpoint=str(previous/'controller_adapter') if previous else None,fresh_lora=previous is None)
    install_cached_vision_inputs(runtime)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    for name,p in runtime.model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:p.requires_grad_(True)
    runtime.model.train();runtime._conditional_model().model.visual.eval()
    runtime.model.gradient_checkpointing_enable();runtime.model.enable_input_require_grads()
    backbone=[p for p in runtime.model.parameters() if p.requires_grad];parameters=list(head.parameters())+backbone
    if sum(p.numel() for p in backbone)!=62521344 or sum(p.numel() for p in head.parameters())!=1049088:
        raise RuntimeError('trainable parameter contract changed')
    optimizer=torch.optim.AdamW([{'params':list(head.parameters()),'lr':cfg['head_lr']},
                                {'params':backbone,'lr':cfg['lora_lr']}],weight_decay=.01)
    step=0;seconds=0.;labels=Counter();last_save=0.
    if previous:
        commit=read(previous/'commit.json')
        for name,digest in commit['files'].items():
            if sha(previous/name)!=digest:raise ValueError('checkpoint checksum mismatch')
        state=torch.load(previous/'training_state.pt',map_location='cpu',weights_only=False)
        if (state['world'],state['global_batch'],state['model_revision'])!=(8,384,MODEL_REVISION):raise ValueError('resume identity mismatch')
        head.load_state_dict(torch.load(previous/'pointer_head.pt',map_location='cpu',weights_only=True));optimizer.load_state_dict(state['optimizer'])
        step=state['step'];seconds=state['training_seconds'];labels.update(state['global_labels']);last_save=seconds
        rng=torch.load(previous/f'rng.rank{rank}.pt',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
    if rank==0:write(out/'trainable_counts.json',dict(lora=sum(p.numel() for p in backbone),pointer=sum(p.numel() for p in head.parameters()),global_batch=384,physical_batch=args.physical_batch))
    dataset=MixedRows(run/'data')
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=1024,defer_vision=True)
    decode_pool=ThreadPoolExecutor(max_workers=4);prefetch=ThreadPoolExecutor(max_workers=1)

    def prepare(batch_rows):
        cached={key:image for key,(image,_) in frames.frames.items()}
        decoded=predecode_batch(runtime,batch_rows,decode_pool,cached)
        samples=[frames.prepare(row,prefetched=decoded,cpu_only=True) for row in batch_rows]
        ordered=sorted(zip(batch_rows,samples),key=lambda item:item[1][0]['input_ids'].shape[1])
        ordered_rows=[r for r,_ in ordered];samples=[s for _,s in ordered]
        chunks=[samples[i:i+args.physical_batch] for i in range(0,48,args.physical_batch)]
        return ordered_rows,[(chunk,collate(chunk,runtime)) for chunk in chunks]

    def schedule(index):
        return prefetch.submit(lambda:prepare(dataset.batch(index,rank)))

    def forward_backward(prepared,*,update=True):
        batch_rows,chunks=prepared;optimizer.zero_grad(set_to_none=True);loss_value=0.;model_seconds=0.;vision_seconds=0.;offset=0
        for samples,(cpu_inputs,visions,lengths) in chunks:
            tick=time.monotonic();encode_batch_vision(runtime,samples,8);torch.cuda.synchronize();vision_seconds+=time.monotonic()-tick
            packed={k:v.to(runtime.device) for k,v in cpu_inputs.items()}
            tick=time.monotonic()
            targets=torch.tensor([ACTIONS.index(batch_rows[offset+i]['label']) for i in range(len(samples))],device=runtime.device)
            with runtime._cached_image_features(visions):
                logits,_=batched_logits(runtime,head,samples,(packed,visions,lengths))
                loss=F.cross_entropy(logits,targets,reduction='sum')/48
                if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite supervised loss')
                loss.backward()
            loss_value+=float(loss.detach());offset+=len(samples)
            torch.cuda.synchronize();model_seconds+=time.monotonic()-tick
            del packed,logits,loss,targets
        tick=time.monotonic();sync_gradients(parameters,8);sync_seconds=time.monotonic()-tick
        if step==0 and not any(p.grad is not None and bool(p.grad.ne(0).any()) for p in backbone):
            raise RuntimeError('missing LoRA gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite gradients')
        if update:optimizer.step()
        torch.cuda.synchronize()
        return dict(loss=loss_value,gradient_norm=float(norm),model_seconds=model_seconds,
                    vision_seconds=vision_seconds,gradient_sync_seconds=sync_seconds)

    def save(hour=None):
        nonlocal last_save
        save_checkpoint(out,runtime,head,optimizer,step=step,seen=step*48,rank=rank,world=8,mode='J1',seed=42,
            model_name=MODEL_NAME,model_revision=MODEL_REVISION,extra_state=dict(training_seconds=seconds,
                global_batch=384,global_labels=dict(labels),source_presentations={'pace':step*192,'estp_it':step*192},
                physical_batch=args.physical_batch))
        checkpoint=out/f'step-{step:06d}'
        if rank==0:
            files={str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file() and p.name!='commit.json'}
            write(checkpoint/'commit.json',dict(step=step,files=files,training_seconds=seconds,evaluation_hour=hour))
            if hour is not None:
                write(run/'queue'/f'hour-{hour:02d}.json',dict(name=f'hour-{hour:02d}',hour=hour,step=step,checkpoint=str(checkpoint),training_seconds=seconds))
        dist.barrier();last_save=seconds

    def parity(prepared):
        rows_,chunks=prepared;samples=[s for chunk,_ in chunks for s in chunk]
        selected=[samples[0],samples[-1]];targets=torch.tensor([ACTIONS.index(rows_[0]['label']),ACTIONS.index(rows_[-1]['label'])],device=runtime.device)
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();dropouts=[(m,m.p) for m in runtime.model.modules() if isinstance(m,torch.nn.Dropout)]
        for m,_ in dropouts:m.p=0.
        encode_batch_vision(runtime,selected,8)
        def observe(batch):
            optimizer.zero_grad(set_to_none=True)
            if batch:
                packed,visions,lengths=collate(selected,runtime);packed={k:v.to(runtime.device) for k,v in packed.items()}
                with runtime._cached_image_features(visions):
                    logits,_=batched_logits(runtime,head,selected,(packed,visions,lengths))
                    loss=F.cross_entropy(logits,targets);loss.backward()
                probs=logits.detach().float().softmax(-1);value=float(loss.detach())
            else:
                probabilities=[];value=0.
                for i,(inputs,vision,options,decision) in enumerate(selected):
                    inputs={k:v.to(runtime.device) for k,v in inputs.items()}
                    with runtime._cached_image_features(vision):
                        logits=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=True)
                        loss=F.cross_entropy(logits[None],targets[i:i+1])/2;loss.backward()
                    probabilities.append(logits.detach().float().softmax(-1));value+=float(loss.detach())
                probs=torch.stack(probabilities)
            gradient=torch.cat([p.grad.detach().flatten().clone() for p in head.parameters()])
            return probs,value,gradient
        native_batch=observe(True);native_single=observe(False)
        # Structural parity uses a full-precision reference. BF16 kernel rounding
        # depends on sequence/batch geometry and must not be mistaken for padding
        # or gradient-normalization errors. No reference weights are optimized.
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
        dtypes=[(p,p.dtype) for p in runtime.model.parameters()]
        buffers=[(p,p.dtype) for p in runtime.model.buffers()]
        kernels=[(m,m.chunk_gated_delta_rule) for m in runtime.model.modules() if hasattr(m,'chunk_gated_delta_rule')]
        optimizer.zero_grad(set_to_none=True)
        for m,_ in kernels:m.chunk_gated_delta_rule=qwen.torch_chunk_gated_delta_rule
        runtime.model.float()
        reference_batch=observe(True);reference_single=observe(False)
        optimizer.zero_grad(set_to_none=True)
        for p,dtype in dtypes:p.data=p.data.to(dtype)
        for p,dtype in buffers:p.data=p.data.to(dtype)
        for m,kernel in kernels:m.chunk_gated_delta_rule=kernel
        def probability_error(a,b):return float((a[0]-b[0]).abs().max())
        def gradient_error(a,b):return float((a[2]-b[2]).norm()/b[2].norm().clamp_min(1e-8))
        structural_probability=probability_error(reference_batch,reference_single)
        structural_gradient=gradient_error(reference_batch,reference_single)
        result=dict(reference_probability_max_error=structural_probability,
            reference_head_gradient_relative_l2=structural_gradient,
            reference_batched_loss=reference_batch[1],reference_singleton_loss=reference_single[1],
            bf16_batch_probability_error=probability_error(native_batch,reference_single),
            bf16_single_probability_error=probability_error(native_single,reference_single),
            bf16_batch_gradient_error=gradient_error(native_batch,reference_single),
            bf16_single_gradient_error=gradient_error(native_single,reference_single),
            native_batched_loss=native_batch[1],native_singleton_loss=native_single[1],
            protocol='FP32 structural parity plus BF16 numerical regression against the same FP32 singleton reference')
        write(run/f'batch-parity.rank{rank}.json',result)
        if structural_probability>.005 or structural_gradient>.01 or abs(reference_batch[1]-reference_single[1])>.01*max(.1,abs(reference_single[1])):
            raise RuntimeError('full-precision batch/singleton gradient parity failed: '+str(result))
        if (result['bf16_batch_probability_error']>2*result['bf16_single_probability_error']+.005 or
            result['bf16_batch_gradient_error']>2*result['bf16_single_gradient_error']+.01):
            raise RuntimeError('BF16 batch numerical error exceeds singleton reference envelope: '+str(result))
        for m,p in dropouts:m.p=p
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        del native_batch,native_single,reference_batch,reference_single
        torch.cuda.empty_cache()

    if not previous:save()
    pending=schedule(step)
    if not previous:
        initial=pending.result();parity(initial);pending=None
        # Preserve the same first production batch; parity takes no updates.
        first=initial
    else:first=None
    profile_rows=[];last_progress=0.
    if args.profile:
        stress_rows=dataset.stress(rank);dump(run/f'stress.rank{rank}.jsonl',stress_rows)
        stress=prepare(stress_rows);torch.cuda.reset_peak_memory_stats();forward_backward(stress)
        peak=torch.tensor(torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(runtime.device).total_memory);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if float(peak)>.94:raise RuntimeError('insufficient worst-case memory headroom: '+str(float(peak)))
        write(run/f'capacity.rank{rank}.json',dict(passed=True,reserved_fraction=float(peak),physical_batch=args.physical_batch))
        del stress
    total=args.profile_updates if args.profile else read(run/'step-schedule.json')['total_steps']
    interval=None if args.profile else read(run/'step-schedule.json')['interval_steps']
    while step<total:
        tick=time.monotonic()
        if first is not None:prepared=first;first=None
        else:prepared=pending.result()
        prepare_wait=time.monotonic()-tick
        pending=schedule(step+1) if step+1<total else None
        rate=1. if args.profile else min(1.,(step+1)/50)
        optimizer.param_groups[0]['lr']=cfg['head_lr']*rate;optimizer.param_groups[1]['lr']=cfg['lora_lr']*rate
        metrics=forward_backward(prepared);step+=1
        duration=torch.tensor(time.monotonic()-tick);dist.all_reduce(duration,op=dist.ReduceOp.MAX);seconds+=float(duration)
        counts=torch.tensor([sum(r['source']==source and r['label']==action for r in prepared[0]) for source in ('pace','estp_it') for action in ACTIONS]);dist.all_reduce(counts)
        labels.update(dict(zip([f'{s}:{a}' for s in ('pace','estp_it') for a in ACTIONS],map(int,counts.tolist()))))
        profile_rows.append(float(duration))
        if rank==0 and (step<=3 or step%5==0 or time.monotonic()-last_progress>45):
            progress=dict(state='profiling' if args.profile else 'training',step=step,training_seconds=seconds,
                last_step_seconds=float(duration),global_batch=384,physical_batch=args.physical_batch,
                source_presentations={'pace':step*192,'estp_it':step*192},prepare_wait_seconds=prepare_wait,
                head_lr=optimizer.param_groups[0]['lr'],lora_lr=optimizer.param_groups[1]['lr'],
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,**metrics)
            write(run/'training-status.json',progress);print(json.dumps(progress),flush=True);last_progress=time.monotonic()
        hour=(step//interval)*2 if interval and step%interval==0 else None
        if not args.profile and (step==1 or hour is not None or seconds-last_save>=900 or step==total):save(hour)
        if not args.profile and step==1 and not args.resume:
            # Post-update cross-node gate uses an ordinary singleton observation.
            row=prepared[0][-1];runtime.model.eval();head.eval()
            inputs,vision,options,decision=prepared[1][-1][0][-1]
            inputs={k:v.to(runtime.device) for k,v in inputs.items()}
            with torch.no_grad(),runtime._cached_image_features(vision):value=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False)
            write(run/f'parity.rank{rank}.json',dict(row=row,logits=value.float().cpu().tolist()));dist.barrier()
            if rank==0:write(run/'first-update-gate.json',dict(step=1,checkpoint=str(out/'step-000001')))
            began=time.monotonic()
            while not (run/'cross-node-parity.json').exists():
                if (run/'cross-node-parity-failed.json').exists() or time.monotonic()-began>1800:raise RuntimeError('cross-node parity gate failed')
                time.sleep(5)
            runtime.model.train();runtime._conditional_model().model.visual.eval();head.train()
        del prepared
    if args.profile:
        measured=profile_rows[3:];write(run/f'profile.rank{rank}.json',dict(physical_batch=args.physical_batch,
            measured_step_seconds=measured,mean_step_seconds=sum(measured)/len(measured),
            examples_per_second=384/(sum(measured)/len(measured)),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
    elif rank==0:
        final=dict(state='complete',step=step,training_seconds=seconds,global_batch=384,labels=dict(labels),
            source_presentations={'pace':step*192,'estp_it':step*192},checkpoint=str(out/f'step-{step:06d}'))
        write(run/'training-complete.json',final);write(run/'training-status.json',final)
    prefetch.shutdown();decode_pool.shutdown();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--profile-updates',type=int,default=10)
    p.add_argument('--physical-batch',type=int,choices=(24,48),default=48)
    main(p.parse_args())
