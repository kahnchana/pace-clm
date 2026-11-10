#!/usr/bin/env python3
"""Continue local-only rollouts after the independent classification diagnostic exits."""
import csv,io,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/jevany-qwen35-2b-mixture-12h/scaling-diagnostic'
PY=str(ROOT.parent/'stream-gemma/.venv/bin/python')
if socket.gethostname().split('.')[0]!='ip-10-3-70-177':raise RuntimeError('wrong node')
deadline=time.monotonic()+3*3600
while not (OUT/'gold_history/complete.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('classification diagnostic did not finish')
    time.sleep(5)
while True:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    safe=all(int(v.strip())>=115000 for v in free.splitlines() if v.strip())
    jobs=[]
    for pid,memory in csv.reader(io.StringIO(active)):
        try:args=Path('/proc',pid.strip(),'cmdline').read_bytes().split(b'\0')
        except OSError:safe=False;break
        jobs.append(dict(pid=int(pid),memory_mib=int(memory),program=next((x.decode(errors='replace') for x in args[1:4] if x.endswith(b'.py')),args[0].decode(errors='replace'))))
        # Completed classification workers can linger in library teardown with
        # only a CUDA context. Never overlap a live model or an unrelated job.
        if int(memory.strip())>2048 or b'--teacher-only' not in args or not any(x.endswith(b'eval_qwen35_development_sweep.py') for x in args):safe=False
    state=dict(state='ready' if safe else 'queued_for_this_node',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        minimum_free_mib=min(int(v.strip()) for v in free.splitlines() if v.strip()),active_jobs=jobs)
    temp=OUT/'gpu_queue.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(OUT/'gpu_queue.json')
    if safe:break
    if time.monotonic()>deadline:
        (OUT/'gpu_failure.json').write_text(json.dumps({'error':'This node remained occupied; GPU queue timed out.'})+'\n')
        raise TimeoutError('GPUs not safely available')
    time.sleep(5)
os.environ.update(PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',ROOT.parent/'stream-gemma/src',ROOT.parent/'stream-gemma'])),
    HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',
    PACE_DISTRIBUTED_BACKEND='gloo',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',
    TRITON_CACHE_DIR='/dev/shm/pace-representative-triton',TMPDIR='/dev/shm/pace-representative-tmp')
with (OUT/'logs/development_gpu.log').open('a') as f:
    result=subprocess.run([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        'scripts/eval_qwen35_development_sweep.py','--conditions',str(OUT/'development_conditions.json')],
        cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (OUT/'gpu_failure.json').write_text(json.dumps({'error':f'Development rollout job exited {result.returncode}; see development_gpu.log.'})+'\n')
        result.check_returncode()
