"""Local queue bridge for a frozen generator running a different Torch runtime."""
import os
import subprocess
import threading
from multiprocessing.managers import BaseManager


from pace_clm.rl_broker import FrozenGeneratorBroker, AnswerSink


class QueueManager(BaseManager):
    pass


def serve(events,requests,answers):
    key=os.urandom(32)
    broker=FrozenGeneratorBroker(requests,answers,on_error=lambda error:events.put(('error','generator-broker',error)))
    QueueManager.register('events',callable=lambda:events,exposed=('put',))
    QueueManager.register('requests',callable=lambda:broker,exposed=('get',))
    QueueManager.register('answer',callable=lambda rank:AnswerSink(broker,rank),exposed=('put',))
    manager=QueueManager(address=('127.0.0.1',0),authkey=key)
    server=manager.get_server()
    server.generator_broker=broker
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server,key


def connect(port,key):
    QueueManager.register('events')
    QueueManager.register('requests')
    QueueManager.register('answer')
    manager=QueueManager(address=('127.0.0.1',port),authkey=key)
    manager.connect()
    return manager


class Sidecar:
    def __init__(self,command,env):
        self.process=subprocess.Popen(command,env=env)
    @property
    def pid(self):return self.process.pid
    @property
    def exitcode(self):return self.process.poll()
    def is_alive(self):return self.process.poll() is None
    def terminate(self):self.process.terminate()
    def join(self,timeout=None):
        try:self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass


def require_native_grouped_mm():
    import torch
    from transformers.integrations import moe
    if not hasattr(torch.nn.functional,'grouped_mm'):
        raise RuntimeError('frozen generator requires native grouped_mm')
    original=moe._grouped_mm
    counts={'native_grouped_mm_calls':0}
    def checked(inputs,weight,offs):
        if not moe._can_use_grouped_mm(inputs,weight,offs):
            raise RuntimeError('refusing the slow Python MoE expert fallback')
        counts['native_grouped_mm_calls']+=1
        return original(inputs,weight,offs)
    moe._grouped_mm=checked
    return counts
