#!/usr/bin/env python3
"""Run raw-score closed-loop conditions while reusing frozen GPU model loads.

The legacy margin field remains for artifact compatibility; only zero is allowed.
"""
from __future__ import annotations
import argparse,faulthandler,gc,hashlib,importlib.metadata,json,os,sys,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.nn.functional as F
import eval_jevany_streaming as rollout

ROOT=Path(__file__).resolve().parents[1]

def file_barrier(directory,rank,world):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'done.rank{rank:02d}.json').write_text(json.dumps({'rank':rank,'completed':time.time()})+'\n')
    deadline=time.monotonic()+2*3600
    while not all((directory/f'done.rank{r:02d}.json').exists() for r in range(world)):
        if time.monotonic()>deadline:raise TimeoutError('diagnostic worker barrier timed out')
        time.sleep(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--gold-history',type=Path);p.add_argument('--teacher-output',type=Path)
    p.add_argument('--teacher-only',action='store_true')
    p.add_argument('--teacher-batch',type=int,default=1)
    p.add_argument('--require-native-moe',action='store_true')
    p.add_argument('--generator-kernels',choices=['torch','fla'],default='torch')
    a=p.parse_args(); conditions=json.loads(a.conditions.read_text())
    from pace_clm.evaluation_policy import require_raw_action_scores
    for condition in conditions:require_raw_action_scores(condition['margin'])
    stack_dir=a.conditions.parent/'logs'
    stack_dir.mkdir(parents=True,exist_ok=True)
    stack_file=(stack_dir/f'stack.rank{int(os.environ.get("RANK","0")):02d}.log').open('a')
    faulthandler.enable(file=stack_file)
    # Keep fatal-signal diagnostics, but do not asynchronously walk Triton's
    # native launcher frames on a timer. Both observed segfaults coincided with
    # a truncated periodic stack dump inside that extension.
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    if a.require_native_moe:
        from jevany_rl_rpc import require_native_grouped_mm
        native_counts= require_native_grouped_mm()
    print(json.dumps({'rank':rank,'runtime_versions':{n:importlib.metadata.version(n)
        for n in ('torch','transformers','peft','triton')}}),flush=True)
    from qwen35_diagnostic_kernels import pin_safe_l2norm
    settings=pin_safe_l2norm()
    if rank==0:(a.conditions.parent/'kernel_launch_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    controllers={};generators={}
    original_dense=rollout.load_dense_controller
    original_generator=rollout.QwenRuntime.load
    def dense(**kw):
        key=kw['checkpoint']
        if key not in controllers:
            if len(controllers)>=2:
                oldest=next(iter(controllers));old_runtime=controllers.pop(oldest)[0]
                if 'generate_turn' in old_runtime.__dict__:del old_runtime.__dict__['generate_turn']
                del old_runtime;gc.collect();torch.cuda.empty_cache()
            controllers[key]=original_dense(**kw)
        return controllers[key]
    def generator(**kw):
        key=tuple(sorted(kw.items()))
        if key not in generators:
            if a.generator_kernels=='torch':
                # Use Transformers' own mathematical fallbacks for the frozen
                # MoE generator only. Its FLA recurrent path segfaulted in both
                # Torch runtimes; dense controller kernels are unchanged.
                from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
                for name in ('FusedRMSNormGated','chunk_gated_delta_rule',
                             'fused_recurrent_gated_delta_rule','causal_conv1d_fn',
                             'causal_conv1d_update'):
                    setattr(moe,name,None)
                moe.is_fast_path_available=False
            # Use the production CPU-to-GPU loader. Serialize legacy runtime
            # transfers; native Torch 2.10 workers can load in parallel. This
            # changes loading concurrency, not weights or generation settings.
            run_id=os.environ.get('TORCHELASTIC_RUN_ID',str(os.getppid()))
            model_key=hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            directory=a.conditions.parent/'generator_loading'/run_id/model_key
            directory.mkdir(parents=True,exist_ok=True)
            deadline=time.monotonic()+2*3600
            if rank and not a.require_native_moe:
                while not (directory/f'rank{rank-1:02d}.json').exists():
                    if time.monotonic()>deadline:raise TimeoutError('generator loading rank did not finish')
                    time.sleep(1)
            print(json.dumps({'rank':rank,'generator_loading':'started'}),flush=True)
            generators[key]=original_generator(**kw)
            print(json.dumps({'rank':rank,'generator_kernels':a.generator_kernels}),flush=True)
            torch.cuda.synchronize()
            (directory/f'rank{rank:02d}.json').write_text(json.dumps({'loaded':True,'rank':rank})+'\n')
            print(json.dumps({'rank':rank,'generator_loading':'complete'}),flush=True)
        return generators[key]
    rollout.load_dense_controller=dense
    rollout.QwenRuntime.load=staticmethod(generator)
    for condition in ([] if a.teacher_only else conditions):
        directory=Path(condition['output']);directory.mkdir(parents=True,exist_ok=True)
        print(json.dumps({'rank':rank,'condition':condition['name'],'started':time.time()}),flush=True)
        sys.argv=['eval_jevany_streaming.py','--controller',condition['controller'],
            '--output-dir',condition['output'],
            '--response-margin',str(condition['margin']),'--answer-model','promoted',
            '--max-new-tokens','96','--record-action-logits']
        if condition.get('benchmark'):
            if condition.get('manifest'):raise ValueError('condition specifies both benchmark and validation manifest')
            sys.argv+=['--benchmark',condition['benchmark']]
        else:
            sys.argv+=['--validation-manifest',condition['manifest']]
        rollout.main()
        if a.require_native_moe:
            print(json.dumps({'rank':rank,**native_counts}),flush=True)
        file_barrier(directory/'worker_completion',rank,world)
        if rank==0:
            (directory/'rollout_complete.json').write_text(json.dumps({'condition':condition,'ranks':world,'completed':time.time()})+'\n')
    if a.gold_history:
        from train_jevany_streaming import EpisodeFrames
        from benchmark_qwen35_2b_training import batched_logits,encode_batch_vision
        from pace_clm.jevany_streaming import ACTIONS,PointerHead
        rows=[json.loads(l) for l in a.gold_history.read_text().splitlines()]
        assigned=[(i,r) for i,r in enumerate(rows) if i%world==rank]
        a.teacher_output.mkdir(parents=True,exist_ok=True)
        for checkpoint in sorted({c['controller'] for c in conditions}):
            key=str(Path(checkpoint)/'controller_adapter')
            if key not in controllers:
                from pace_clm.qwen_controller_runtime import MODEL_NAME,MODEL_REVISION
                dense(model_name=MODEL_NAME,model_revision=MODEL_REVISION,checkpoint=key)
            runtime,_=controllers[str(Path(checkpoint)/'controller_adapter')]
            runtime.model.eval()
            head=PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
            head.load_state_dict(torch.load(Path(checkpoint)/'pointer_head.pt',map_location='cpu',weights_only=True));head.eval()
            frames=EpisodeFrames(runtime,response_blind=False,cache_frames=512,defer_vision=True)
            target=a.teacher_output/f'{Path(checkpoint).name}.rank{rank:02d}.jsonl'
            if target.exists():raise FileExistsError(target)
            with target.open('w') as f,torch.no_grad():
                for start in range(0,len(assigned),a.teacher_batch):
                    group=assigned[start:start+a.teacher_batch]
                    samples=[frames.prepare(row) for _,row in group]
                    encode_batch_vision(runtime,samples,8)
                    if len(samples)==1:
                        from pace_clm.jevany_streaming import controller_logits
                        inputs,vision,options,decision=samples[0]
                        logits=controller_logits(runtime,head,inputs,vision,options,decision,train_decoder=False).unsqueeze(0)
                    else:
                        logits,_=batched_logits(runtime,head,samples)
                    labels=torch.tensor([ACTIONS.index(row['label']) for _,row in group],device=runtime.device)
                    losses=F.cross_entropy(logits,labels,reduction='none')
                    for (i,row),values,loss in zip(group,logits.float().cpu().tolist(),losses.cpu().tolist()):
                        f.write(json.dumps(dict(index=i,video_id=row['video_id'],frame_index=row['frame_index'],
                            label=row['label'],label_index=ACTIONS.index(row['label']),logits=values,nll=loss,
                            memory=bool(row.get('memory')),context_frames=len(row['turns'])))+'\n')
                    f.flush()
            del frames,head;gc.collect()
        file_barrier(a.teacher_output/'worker_completion',rank,world)
        if rank==0:(a.teacher_output/'complete.json').write_text(json.dumps({'rows':len(rows),'checkpoints':sorted({c['step'] for c in conditions})})+'\n')

if __name__=='__main__':
    main()
    # All scientific files are closed and all ranks passed the completion
    # barrier. Avoid the observed stall in model/library interpreter teardown.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
