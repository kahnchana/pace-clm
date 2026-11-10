#!/usr/bin/env python3
"""Local-only full-state reconstruction and matched context-order continuations."""
from __future__ import annotations
import fcntl,json,os,socket,subprocess,sys,threading,time,traceback
from pathlib import Path
import shutil
from qwen35_context_order import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-context-order-pilot'
PY=str(ROOT.parent/'stream-gemma/.venv/bin/python')

def write(path,v):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def status(state,**kwargs):
    value=dict(state=state,host=socket.gethostname(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    write(OUT/'training_status.json',value);write(OUT/'status.json',value);print(json.dumps(value),flush=True)
def gpu_free():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def await_gpus():
    deadline=time.monotonic()+3*3600
    while not gpu_free():
        status('waiting_for_this_nodes_gpus')
        if time.monotonic()>deadline:raise TimeoutError('this node remained busy')
        time.sleep(15)
def summaries(path):return [json.loads((path/f'summary.rank{r}.json').read_text()) for r in range(8)]
def run_training(plan,name,output,order,target,*,parent=None,parent_recipe=None,verify=False):
    await_gpus()
    args=[PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        'scripts/train_jevany_2b_batched.py','--manifest',plan['source'],
        '--expected-manifest-sha256',plan['source_sha256'],'--row-order',str(order),'--output',str(output),
        '--microbatch','16','--microbatch-ratio','1.0','--require-global-batch','128','--backend','gloo',
        '--frame-cache','1024','--conv-batch-chunk','128','--predecode-workers','4','--prefetch-mode','prepare',
        '--fixed-l2norm-launch','--save-every',str(plan.get('save_every',16)),'--log-every','1','--seed','42','--stop-after-steps',str(target)]
    if verify:args+=['--initialize-from',str(parent),'--verify-restore-only']
    elif (output/'latest.json').exists():args+=['--resume']
    else:args+=['--fork-from',str(parent),'--fork-parent-recipe',str(parent_recipe)]
    if not verify and plan.get('continuation_seed') is not None:
        args+=['--continuation-seed',str(plan['continuation_seed'])]
    log=OUT/f'logs/{name}.log'
    with log.open('a') as f:
        f.write(json.dumps({'command':args,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})+'\n');f.flush()
        process=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        stop=threading.Event()
        def sample():
            with (OUT/f'logs/{name}_gpu.csv').open('a') as gpu:
                while not stop.is_set():
                    subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],stdout=gpu,stderr=subprocess.DEVNULL)
                    gpu.flush();stop.wait(5)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        code=process.wait();stop.set();thread.join(timeout=10)
        if code:raise subprocess.CalledProcessError(code,args)
    if not verify:
        latest=json.loads((output/'latest.json').read_text())
        if latest['step']!=target:raise ValueError('training did not reach frozen target')
        result=summaries(output)
        if any(r['peak_reserved_gib']>.9*r['device_memory_gib'] for r in result):raise ValueError('less than 10 percent reserved-memory headroom')
        return result

def main():
    if socket.gethostname().split('.')[0]!='ip-10-3-70-177':raise RuntimeError('wrong node')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'launch_plan.json').read_text())
    for path,key in [(plan['reference_order'],'reference_order_sha256'),(plan['shuffled_order'],'shuffled_order_sha256')]:
        if sha256(path)!=plan[key]:raise ValueError(f'launch plan checksum changed: {key}')
    os.environ.update(PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',
        PACE_DISTRIBUTED_BACKEND='gloo',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',
        TRITON_CACHE_DIR='/dev/shm/pace-context-order-triton',TMPDIR='/dev/shm/pace-context-order-tmp')
    for k in ['TRITON_CACHE_DIR','TMPDIR']:Path(os.environ[k]).mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+600
    while not (OUT/'preflight/normalization_gradient_parity.json').exists():
        status('waiting_for_normalization_gradient_parity')
        if time.monotonic()>deadline:raise TimeoutError('gradient preflight missing')
        time.sleep(5)
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['train_jevany_2b_batched.py','train_jevany_streaming.py','qwen35_context_order.py','qwen35_diagnostic_kernels.py','run_qwen35_context_order_pilot.py','prepare_qwen35_context_order_pilot.py']:
        dest=code/name
        if not dest.exists():shutil.copy2(ROOT/'scripts'/name,dest)
    verify=OUT/'preflight/exact_restore'
    if not all((verify/f'verified.rank{r}.json').exists() for r in range(8)):
        status('verifying_parent_checkpoint_restore',checkpoint=plan['parent_checkpoint'])
        run_training(plan,'exact_restore',verify,plan['reference_order'],1344,parent=plan['parent_checkpoint'],verify=True)
    shared=OUT/'shared/train'
    if not (shared/'summary.rank0.json').exists():
        status('reconstructing_shared_checkpoint',from_step=1332,target_step=1344)
        run_training(plan,'shared_reconstruction',shared,plan['reference_order'],1344,
            parent=plan['parent_checkpoint'],parent_recipe=plan['parent_recipe'])
    common=shared/'step-001344'
    import torch
    initial=torch.load(common/'training_state.pt',map_location='cpu',weights_only=False)
    if initial['step']!=1344 or initial['seen_per_rank']*8!=172032:raise ValueError('shared cursor mismatch')
    results={}
    for arm in plan['training_arm_order']:
        output=OUT/arm/'train';order=plan['shuffled_order'] if arm=='shuffled' else plan['reference_order']
        if (output/'summary.rank0.json').exists() and json.loads((output/'summary.rank0.json').read_text())['step']==1792:
            result=summaries(output)
        else:
            status('training',arm=arm,from_step=1344,target_step=1792,updates=448,global_batch=128)
            result=run_training(plan,f'train_{arm}',output,order,1792,parent=common,parent_recipe=shared/'recipe.json')
        state=torch.load(output/'step-001792/training_state.pt',map_location='cpu',weights_only=False)
        if state['seen_per_rank']*8!=229376:raise ValueError('final exposure mismatch')
        results[arm]=dict(final_step=1792,presentations=229376,continuation_presentations=57344,
            continuation_training_hours=(state['training_seconds']-initial['training_seconds'])/3600,
            global_labels=state['global_labels'],rank_summaries=result)
        write(OUT/'training_results.json',results)
    if results['shuffled']['global_labels']!=results['reference']['global_labels']:raise ValueError('arm class exposures differ')
    write(OUT/'training_complete.json',dict(results=results,shared_checkpoint=str(common)))
    status('training_complete_awaiting_evaluation',results=str(OUT/'training_results.json'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=str(error),traceback=traceback.format_exc());raise
