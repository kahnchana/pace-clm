"""Memoize the immutable generator environment without sharing Torch tensors."""
import hashlib,json,queue,threading,time,traceback
from collections import OrderedDict


class FrozenGeneratorBroker:
    """Share immutable greedy answers and in-flight work across generator GPUs."""
    STOP=('__broker_stop__',)

    def __init__(self,incoming,answers,max_cache=8192,start=True,on_error=None):
        self.incoming=incoming
        self.on_error=on_error
        self.answers=answers
        self.ready=queue.Queue()
        self.cache=OrderedDict()
        self.pending={}
        self.request_context={}
        self.maximum=max_cache
        self.lock=threading.Lock()
        self.counts=dict(logical_requests=0,unique_computations=0,cache_hits=0,inflight_duplicates=0)
        self.thread=None
        if start:
            self.thread=threading.Thread(target=self.dispatch,daemon=True)
            self.thread.start()

    @staticmethod
    def context_key(row):
        # Every field used for image selection and generator messages. Physical
        # actor padding, policy version and trajectory identity are not inputs.
        value={k:row[k] for k in ('video_path','question','frame_index','memory','turns')}
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def dispatch(self):
        try:
            while True:
                task=self.incoming.get()
                if task==self.STOP:return
                self.submit(task)
        except BaseException:
            if self.on_error:self.on_error(traceback.format_exc())
            else:raise

    def submit(self,task):
        if task is None or task[0]=='judge':
            self.ready.put(task)
            return
        rank,request_id,row,submitted=task
        key=self.context_key(row)
        with self.lock:
            self.counts['logical_requests']+=1
            if key in self.cache:
                text=self.cache[key]
                self.cache.move_to_end(key)
                self.counts['cache_hits']+=1
                self.answers[rank].put((request_id,text,{
                    'gpu':None,'batch_id':'cache:'+request_id,'batch':0,'seconds':0.,
                    'served_from_cache':True,'deduplicated':True,
                    'request_seconds':time.monotonic()-submitted,
                    'allocated_gib':0.,'reserved_gib':0.}))
            elif key in self.pending:
                self.counts['inflight_duplicates']+=1
                self.pending[key].append((rank,request_id,submitted))
            else:
                self.counts['unique_computations']+=1
                self.pending[key]=[(rank,request_id,submitted)]
                self.request_context[request_id]=(key,rank)
                self.ready.put(task)

    def get(self,*args,**kwargs):
        return self.ready.get(*args,**kwargs)

    def complete(self,rank,message):
        request_id,text,stats=message
        if not text.strip():raise RuntimeError('cannot cache an empty frozen-generator answer')
        with self.lock:
            key,expected_rank=self.request_context.pop(request_id)
            if rank!=expected_rank:raise RuntimeError('generator answer arrived on the wrong channel')
            waiting=self.pending.pop(key)
            self.cache[key]=text
            self.cache.move_to_end(key)
            while len(self.cache)>self.maximum:self.cache.popitem(last=False)
            for index,(target,original_id,submitted) in enumerate(waiting):
                self.answers[target].put((original_id,text,{
                    **stats,'served_from_cache':False,'deduplicated':index>0,
                    'request_seconds':time.monotonic()-submitted}))

    def snapshot(self):
        with self.lock:return {**self.counts,'cache_entries':len(self.cache),'pending_contexts':len(self.pending)}

    def close(self):
        if self.thread:
            self.incoming.put(self.STOP)
            self.thread.join(timeout=2)


class AnswerSink:
    def __init__(self,broker,rank):self.broker,self.rank=broker,rank
    def put(self,message):self.broker.complete(self.rank,message)
