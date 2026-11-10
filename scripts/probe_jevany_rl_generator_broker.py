#!/usr/bin/env python3
"""Real frozen-generator smoke test of the shared broker; no actor or optimizer."""
import json,os,queue,time
from pathlib import Path
from jevany_rl_rpc import serve,Sidecar


def main():
 root=Path(__file__).resolve().parents[1]
 out=root/'outputs/jevany-2b-rl/generator-broker-smoke';out.mkdir(exist_ok=True)
 cfg=json.loads((root/'outputs/jevany-2b-rl/profile-7p1-v8/freeze.json').read_text())
 cfg.update(output=str(out),learners=2,generator=str(root.parent/'stream-gemma/outputs/lora-qwen-w1-long-neg'))
 path=out/'worker_runtime.json';path.write_text(json.dumps(cfg))
 events,requests=queue.Queue(),queue.Queue();answers=[queue.Queue(),queue.Queue()]
 server,key=serve(events,requests,answers)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PACE_RL_RPC_KEY']=key.hex()
 worker=Sidecar([cfg['generator_python'],'-u',str(root/'scripts/jevany_rl_generator_sidecar.py'),
                 '--port',str(server.address[1]),'--gpu','7','--config',str(path)],env)
 def receive():
  deadline=time.monotonic()+1200
  while time.monotonic()<deadline:
   try:return events.get(timeout=2)
   except queue.Empty:
    if worker.exitcode is not None:raise RuntimeError('generator exited before replying')
  raise TimeoutError('generator smoke timed out')
 try:
  event,rank,value=receive();assert event=='ready',(event,rank,value)
  traces=json.loads((root/'outputs/jevany-2b-rl/profile-7p1-v8/batch-000000/rollouts.json').read_text())
  rows=[];videos=set()
  for trace in traces:
   for step in trace['steps']:
    row=step['row']
    if step['action']==2 and row['video_path'] not in videos:
     rows.append(row);videos.add(row['video_path']);break
   if len(rows)==2:break
  assert len(rows)==2
  for target,name,row in [(0,'a',rows[0]),(1,'b',rows[0]),(0,'c',rows[1]),(1,'d',rows[1])]:
   requests.put((target,name,row,time.monotonic()))
  received={}
  for target in range(2):
   for _ in range(2):
    key,text,stats=answers[target].get(timeout=1200);received[key]=(text,stats)
  assert received['a'][0]==received['b'][0] and received['c'][0]==received['d'][0]
  requests.put((1,'e',rows[0],time.monotonic()))
  hot=answers[1].get(timeout=10);assert hot[0]=='e' and hot[2]['served_from_cache']
  batch=out/'batch';batch.mkdir(exist_ok=True)
  requests.put(('judge',0,[dict(kind='semantic',question='What happens?',candidate='A person walks.',reference='A person walks.')],str(batch)))
  event,rank,value=receive();assert event=='judge' and value['results'][0]['score']>=.75,(event,rank,value)
  stats=server.generator_broker.snapshot()
  assert stats['logical_requests']==5 and stats['unique_computations']==2
  assert stats['cache_hits']+stats['inflight_duplicates']==3 and not stats['pending_contexts']
  result={'passed':True,'generator_broker':stats,'judge_passthrough':True,'optimizer_steps':0}
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 finally:
  requests.put(None);worker.join(timeout=20)
  if worker.is_alive():worker.terminate();worker.join(timeout=5)
  server.generator_broker.close();server.stop_event.set()


if __name__=='__main__':main()
