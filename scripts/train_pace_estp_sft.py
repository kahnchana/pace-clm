#!/usr/bin/env python3
"""Eight-GPU, elapsed-training-time SFT with exact source-balanced batches."""
import argparse
import faulthandler
import hashlib
import json
import os
import random
import socket
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmark_qwen35_2b_training import packed_nccl_sync
from pace_clm.jevany_streaming import ACTIONS, PointerHead, controller_logits
from pace_clm.qwen_controller_runtime import MODEL_NAME, MODEL_REVISION, load_2b_controller
from pace_clm.sft_pace_estp import CyclingPermutation, supervised_row
from prepare_pace_estp_sft import read, write, sha
from train_jevany_streaming import EpisodeFrames, save_checkpoint, sync_gradients


def train(run, resume=False):
    if socket.gethostname() != 'ip-10-3-70-177':
        raise RuntimeError('this SFT is authorized on allocation 38226 only')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8: raise ValueError('eight GPUs required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    stack=(run/f'stack.rank{rank}.log').open('a')
    faulthandler.register(signal.SIGUSR1,file=stack,all_threads=True)
    # This node's NCCL returns CUDA 802; reuse the original SFT's exact packed
    # CPU/Gloo average while all eight GPUs still perform forward/backward.
    dist.init_process_group('gloo')
    communication_probe=torch.ones(1024)
    dist.all_reduce(communication_probe)
    if not bool((communication_probe==world).all()):raise RuntimeError('Gloo preflight failed')
    freeze = read(run/'freeze.json')
    for path,digest in freeze['source_hashes'].items():
        if sha(path) != digest: raise ValueError('frozen source changed: '+path)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    out=run/'train';out.mkdir(exist_ok=True)
    previous=Path(read(out/'latest.json')['checkpoint']) if resume else None
    if not resume and (out/'latest.json').exists(): raise FileExistsError('use resume')
    runtime,_=load_2b_controller(checkpoint=str(previous/'controller_adapter') if previous else None,
                               fresh_lora=previous is None)
    head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    for name,p in runtime.model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name: p.requires_grad_(True)
    runtime.model.train(); runtime._conditional_model().model.visual.eval()
    runtime.model.gradient_checkpointing_enable();runtime.model.enable_input_require_grads()
    backbone=[p for p in runtime.model.parameters() if p.requires_grad]
    parameters=list(head.parameters())+backbone
    optimizer=torch.optim.AdamW([{'params':list(head.parameters()),'lr':1e-4},{'params':backbone,'lr':2e-6}])
    step=0;seconds=0.;labels=Counter();next_hour=2;seen=0
    if previous:
        commit=read(previous/'commit.json')
        for name,digest in commit['files'].items():
            if sha(previous/name)!=digest:raise ValueError('checkpoint checksum mismatch')
        state=torch.load(previous/'training_state.pt',map_location='cpu',weights_only=False)
        if (state['world'],state['model_name'],state['model_revision'])!=(8,MODEL_NAME,MODEL_REVISION):
            raise ValueError('resume identity mismatch')
        head.load_state_dict(torch.load(previous/'pointer_head.pt',map_location='cpu',weights_only=True))
        optimizer.load_state_dict(state['optimizer']);step=state['step'];seen=state['seen_per_rank']
        seconds=state['training_seconds'];next_hour=state['next_evaluation_hour'];labels.update(state['global_labels'])
        rng=torch.load(previous/f'rng.rank{rank}.pt',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
        if rank==0:write(run/'restore-verification.json',dict(step=step,node=socket.gethostname(),extra_optimizer_steps=0))
    if rank==0:
        write(out/'trainable_counts.json',dict(lora=sum(p.numel() for p in backbone),pointer=sum(p.numel() for p in head.parameters()),
            fresh_base_initialization=not resume,model=MODEL_NAME,revision=MODEL_REVISION))
    offsets=np.load(run/'data/pace-offsets.npy',mmap_mode='r')
    pairs=np.load(run/'data/estp-training-pairs.npy',mmap_mode='r')
    estp=[json.loads(line) for line in (run/'data/estp-train.jsonl').read_text().splitlines()]
    pace_path=read(run/'data/audit.json')['pace_manifest'];pace_handle=open(pace_path,'rb')
    samplers=[CyclingPermutation(len(offsets),42),CyclingPermutation(len(pairs),420042)]
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=128)
    last_save=seconds;last_progress=time.monotonic()

    def row_at(source,position):
        index=samplers[source].at(position)
        if source==0:
            offset,length=offsets[index];pace_handle.seek(int(offset))
            row=json.loads(pace_handle.read(int(length)));row['source']='pace';return row
        episode,frame=pairs[index]
        return supervised_row(estp[int(episode)],int(frame))

    def save(milestone=None):
        nonlocal last_save
        save_checkpoint(out,runtime,head,optimizer,step=step,seen=seen,rank=rank,world=world,mode='J1',seed=42,
            model_name=MODEL_NAME,model_revision=MODEL_REVISION,extra_state=dict(training_seconds=seconds,
            next_evaluation_hour=next_hour,global_labels=dict(labels),source_presentations={'pace':step*16,'estp_it':step*16}))
        checkpoint=out/f'step-{step:06d}'
        if rank==0:
            files={str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file() and p.name!='commit.json'}
            write(checkpoint/'commit.json',dict(step=step,files=files,training_seconds=seconds))
            write(run/'training-status.json',dict(state='training',step=step,training_seconds=seconds,
                next_evaluation_hour=next_hour,checkpoint=str(checkpoint),source_presentations={'pace':step*16,'estp_it':step*16}))
            if milestone:
                name=f'hour-{milestone:02d}'
                write(run/'queue'/f'{name}.json',dict(name=name,hour=milestone,step=step,
                    checkpoint=str(checkpoint),training_seconds=seconds))
        dist.barrier();last_save=seconds

    while seconds<freeze['training_seconds']:
        started=time.monotonic();optimizer.zero_grad(set_to_none=True);local_labels=Counter();losses=[]
        # Each rank contributes two examples per source: exact global 16 + 16.
        for local in range(4):
            source=local%2;position=step*16+rank*2+local//2
            if step<2:
                write(run/f'progress.rank{rank}.json',dict(step=step,local=local,phase='preparing',source=source))
            row=row_at(source,position)
            inputs,vision,options,decision=frames.prepare(row)
            if step<2:
                write(run/f'progress.rank{rank}.json',dict(step=step,local=local,phase='forward_backward',source=source))
            with runtime._cached_image_features(vision):
                logits=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=True)
                target=torch.tensor([ACTIONS.index(row['label'])],device=runtime.device)
                loss=F.cross_entropy(logits[None],target)
                if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite supervised loss')
                (loss/4).backward()
            losses.append(float(loss.detach()));local_labels[f"{row['source']}:{row['label']}"]+=1
        sync_gradients(parameters,world)
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not bool(torch.isfinite(grad)):raise RuntimeError('nonfinite gradients')
        if step==0:
            if not any(p.grad is not None and bool(p.grad.ne(0).any()) for p in backbone):raise RuntimeError('missing LoRA gradient')
            if not any(p.grad is not None and bool(p.grad.ne(0).any()) for p in head.parameters()):raise RuntimeError('missing pointer gradient')
        optimizer.step();torch.cuda.synchronize();step+=1;seen+=4
        elapsed=torch.tensor(time.monotonic()-started);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        seconds+=float(elapsed)
        keys=[f'{s}:{a}' for s in ('pace','estp_it') for a in ACTIONS]
        counts=torch.tensor([local_labels[k] for k in keys]);dist.all_reduce(counts)
        labels.update(dict(zip(keys,map(int,counts.cpu().tolist()))))
        if rank==0 and (step==1 or step%25==0 or time.monotonic()-last_progress>45):
            progress=dict(state='training',step=step,training_seconds=seconds,last_step_seconds=float(elapsed),
                rank0_loss=sum(losses)/4,gradient_norm=float(grad),source_presentations={'pace':step*16,'estp_it':step*16},
                labels=dict(labels),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                estimated_total_steps=int(step/seconds*freeze['training_seconds']))
            write(run/'training-status.json',progress);print(json.dumps(progress),flush=True);last_progress=time.monotonic()
        milestone=None
        if seconds>=next_hour*3600:
            milestone=next_hour;next_hour+=2
        if step==1 or milestone or seconds-last_save>=900 or seconds>=freeze['training_seconds']:
            save(milestone)
        if step==1:
            # Verify checkpoint round-trip without taking another optimizer step.
            saved=torch.load(out/'step-000001/pointer_head.pt',map_location=runtime.device,weights_only=True)
            if any(not torch.equal(saved[k],v) for k,v in head.state_dict().items()):raise RuntimeError('head roundtrip mismatch')
            runtime.model.eval();head.eval()
            inputs,vision,options,decision=frames.prepare(row)
            with torch.no_grad(),runtime._cached_image_features(vision):
                probe_logits=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False)
            write(run/f'parity.rank{rank}.json',dict(row=row,logits=probe_logits.float().cpu().tolist()))
            dist.barrier()
            if rank==0:write(run/'first-update-gate.json',dict(step=1,checkpoint=str(out/'step-000001')))
            wait_started=time.monotonic()
            while not (run/'cross-node-parity.json').exists():
                if (run/'cross-node-parity-failed.json').exists():raise RuntimeError('cross-node parity failed')
                if time.monotonic()-wait_started>1800:raise RuntimeError('cross-node gate timeout')
                time.sleep(5)
            runtime.model.train();runtime._conditional_model().model.visual.eval();head.train()
    if rank==0:
        write(run/'training-complete.json',dict(step=step,training_seconds=seconds,source_presentations={'pace':step*16,'estp_it':step*16},
            labels=dict(labels),checkpoint=str(out/f'step-{step:06d}')))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();train(args.run.resolve(),args.resume)
