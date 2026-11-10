"""Offline generator batching diagnosis, using a resident frozen model."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from train_jevany_streaming import EpisodeFrames
from pace_clm.jevany_streaming import decision_messages,option_text
from pace_clm.experiment import direct_inputs


def run(runtime,out,request):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
    if not hasattr(moe,'_rl_original_padding_mask'):
        moe._rl_original_padding_mask=moe.apply_mask_to_padding_states
    moe.apply_mask_to_padding_states=moe._rl_original_padding_mask
    if request.get('singleton_mask'):
        def mask(x,m):
            return x*m[:,:,None] if m is not None and m.shape[1]>1 else x
        moe.apply_mask_to_padding_states=mask
    if request.get('native_judge'):
        import importlib,jevany_rl_rpc
        importlib.reload(jevany_rl_rpc)
        counts=jevany_rl_rpc.require_native_grouped_mm()
        from jevany_rl_judge import TransformersLocalJudge
        from run_jevany_rl import judge_requests,apply_scores
        judge=TransformersLocalJudge(runtime.device,runtime=runtime)
        calibration=judge.calibrate(out/f'calibration-{request["sequence"]}.json')
        traces=json.loads(Path(request['traces']).read_text())
        payload,indexes=judge_requests(traces)
        scored=judge.score_cached(payload,out/'cache.sqlite',out/f'scores-{request["sequence"]}.jsonl')
        return {'passed':True,'calibration':[r['score'] for r in calibration],
                'judgments':apply_scores(traces,indexes,scored),'rewards':[t['reward'] for t in traces],**counts}
    if request.get('audit_only'):
        from jevany_rl_judge import TransformersLocalJudge
        from stream_gemma.grpo.judge import JudgeRequest
        judge=TransformersLocalJudge(runtime.device,runtime=runtime)
        calibration=judge.calibrate(out/f'calibration-{request["sequence"]}.json')
        compared=json.loads(Path(request['generator_comparison']).read_text())
        pairs=[JudgeRequest('semantic',r['row']['question'],r['batched_text'],r['single_text']) for r in compared if r['batched_text'].strip()!=r['single_text'].strip()]
        return {'passed':True,'calibration':[r['score'] for r in calibration],'generator_semantic_scores':[r['score'] for r in judge.score_many(pairs)]}
    traces=json.loads(Path(request['traces']).read_text())
    rows=[s['row'] for t in traces for s in t['steps'] if s['action']==2]
    if request.get('diverse'):
        from collections import defaultdict
        groups=defaultdict(list)
        for row in rows:groups[(row['video_path'],row['question'])].append(row)
        rows=[group[i] for i in range(max(map(len,groups.values()))) for group in groups.values() if i<len(group)]
    rows=rows[request.get('offset',0):][:request.get('count',4)]
    frames=EpisodeFrames(runtime,response_blind=False,cache_frames=2048,defer_vision=True)
    inputs_list,visions=[],[]
    prefix=runtime.tokenizer('</Response>',add_special_tokens=False,return_tensors='pt')['input_ids'].to(runtime.device)
    per_row_vision=[]
    for row in rows:
        sample=frames.prepare(row,cpu_only=True)
        vision=sample[1];runtime.encode_frames_features(vision)
        images=[frames.frames[(row['video_path'],t['frame_index'])][0] for t in row['turns']]
        messages=decision_messages(row,images)
        messages[-1]['content']=[p for p in messages[-1]['content'] if p.get('text')!=option_text()]
        inputs=direct_inputs(runtime,messages,vision)
        inputs['input_ids']=torch.cat([inputs['input_ids'],prefix],-1)
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(prefix)],-1)
        inputs_list.append(inputs);visions.extend(vision);per_row_vision.append(vision)
    longest=(max(v['input_ids'].shape[1] for v in inputs_list)//512+1)*512
    def pack(items):
        packed={}
        for name,fill in [('input_ids',runtime.tokenizer.pad_token_id),('attention_mask',0),('mm_token_type_ids',0)]:
            packed[name]=torch.cat([F.pad(v[name],(longest-v[name].shape[1],0),value=fill) for v in items])
        for name in ('pixel_values','image_grid_thw'):packed[name]=torch.cat([v[name] for v in items])
        return packed
    kwargs=dict(max_new_tokens=96,do_sample=False,return_dict_in_generate=True,output_scores=True)
    torch.cuda.synchronize();batch_started=time.monotonic()
    with torch.no_grad(),runtime._cached_image_features(visions):
        batch=runtime.model.generate(**pack(inputs_list),**kwargs)
    torch.cuda.synchronize();batch_seconds=time.monotonic()-batch_started
    result=[]
    for i,inputs in enumerate(inputs_list):
        copies=len(rows) if request.get('fixed_shape') else 1
        with torch.no_grad(),runtime._cached_image_features(per_row_vision[i]*copies):
            single=runtime.model.generate(**pack([inputs]*copies),**kwargs)
        a=batch.sequences[i,longest:];b=single.sequences[0,longest:]
        n=min(len(a),len(b));diff=(a[:n]!=b[:n]).nonzero().flatten()
        item={'row':rows[i],'batched_text':runtime.tokenizer.decode(a,skip_special_tokens=True),
              'single_text':runtime.tokenizer.decode(b,skip_special_tokens=True),'token_equal':bool(torch.equal(a,b))}
        if len(diff):
            j=int(diff[0]);x=batch.scores[j][i].float();y=single.scores[j][0].float()
            xp=x.log_softmax(-1);yp=y.log_softmax(-1)
            item.update(first_difference=j,batched_token=int(a[j]),single_token=int(b[j]),
                        batch_margin=float(x[a[j]]-x[b[j]]),single_margin=float(y[b[j]]-y[a[j]]),
                        probability_max=float((xp.exp()-yp.exp()).abs().max()),
                        kl=float((yp.exp()*(yp-xp)).sum()))
        result.append(item)
        (out/f'generator-detail-{request["sequence"]}.json').write_text(json.dumps(result,indent=2)+'\n')
        del single
    return {'passed':True,'batch_seconds':batch_seconds,'fixed_shape':request.get('fixed_shape',False),'cases':[{k:v for k,v in r.items() if k!='row'} for r in result]}
