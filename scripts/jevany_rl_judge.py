"""Offline semantic judge: pinned model, explicit rubric, strict JSON, versioned cache."""
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext

import torch

from stream_gemma.grpo.judge import JudgeRequest
from stream_gemma.grpo.local_judge import (
    MODEL_ID, MODEL_REVISION, SEMANTIC_RUBRIC, REPETITION_RUBRIC,
    _parse_score, resolve_pinned_snapshot,
)

VERSION = 'jevany-offline-judge-v8-native-grouped-mm'
CLARIFICATION = (
    'Compare only the candidate and reference supplied in the user message. '
    'Different actors, objects, locations, or actions are new information. '
    'Shared generic words do not mean repetition. A candidate that introduces an '
    'unrelated event has repetition score 0, even if it is unhelpful for the question. '
    'A reference of "- none" means there is no prior information: repetition score 0. '
    'For semantic scoring, equivalent paraphrases deserve full credit; contradictory '
    'actions deserve 0. For repetition, score 1 only when all candidate information '
    'is already in the reference. Judge content, not sentence structure. '
    'An action transition is NEW even with the same actor and object. '
    'Examples for repetition: prior "A man walks", candidate "The man sits down": '
    'score 0. Prior "A car drives", candidate "The car crashes": score 0. '
    'Prior "A woman cuts vegetables", candidate "She continues cutting vegetables": '
    'score 1. Do not call a change to a different action a continuation. '
    'Return exactly {"score": a number from 0 to 1, "reason": a short explanation}.'
)


class TransformersLocalJudge:
    def __init__(self, device, runtime=None):
        self.runtime = runtime
        if runtime is not None:
            from stream_gemma.config import QWEN_MODEL_NAME, QWEN_MODEL_REVISION
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
            from pace_clm.rl import mask_padding_states
            modeling_qwen3_5_moe.apply_mask_to_padding_states = mask_padding_states
            self.tokenizer = runtime.tokenizer
            self.tokenizer.padding_side = 'left'
            self.hf_model = runtime.model
            self.device = device
            self.model_id, self.revision = QWEN_MODEL_NAME, QWEN_MODEL_REVISION
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snapshot = resolve_pinned_snapshot()
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                        padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True).to(device).requires_grad_(False).eval()
        self.device = device
        self.model_id, self.revision = MODEL_ID, MODEL_REVISION

    def render(self, request):
        rubric = SEMANTIC_RUBRIC if request.kind == 'semantic' else REPETITION_RUBRIC
        system = rubric+'\n'+CLARIFICATION
        user = json.dumps({'question': request.question, 'candidate': request.candidate,
                           'gold_response' if request.kind == 'semantic' else 'prior_response_context':
                           request.reference})
        return self.tokenizer.apply_chat_template(
            [{'role':'system','content':system},{'role':'user','content':user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def score_many(self, requests):
        if not requests:
            return []
        score_prefix='{"score": '
        prompts = [self.render(r)+score_prefix for r in requests]
        inputs = self.tokenizer(prompts, padding=True, pad_to_multiple_of=512,
                                return_tensors='pt').to(self.device)
        paths=[self.tokenizer.encode(str(v)+'}',add_special_tokens=False)
               for v in (0.,.25,.5,.75,1.)]
        prompt_length=inputs['input_ids'].shape[1]
        eos=self.tokenizer.eos_token_id

        def allowed_tokens(batch_id,ids):
            suffix=ids[prompt_length:].tolist()
            allowed={p[len(suffix)] for p in paths if len(suffix)<len(p) and p[:len(suffix)]==suffix}
            if allowed:return sorted(allowed)
            if any(suffix[:len(p)]==p for p in paths):return [eos]
            raise RuntimeError('invalid constrained judge score prefix')

        with (self.hf_model.disable_adapter() if self.runtime else nullcontext()), torch.no_grad():
            output=self.hf_model.generate(**inputs,do_sample=False,
                max_new_tokens=max(map(len,paths))+1,prefix_allowed_tokens_fn=allowed_tokens,
                eos_token_id=eos,pad_token_id=self.tokenizer.pad_token_id)
        suffixes=self.tokenizer.batch_decode(output[:,prompt_length:],skip_special_tokens=True)
        results = []
        for request,prompt,suffix in zip(requests,prompts,suffixes,strict=True):
            value=json.loads(score_prefix+suffix)
            if set(value)!={'score'} or value['score'] not in (0.,.25,.5,.75,1.):
                raise RuntimeError('invalid constrained semantic score')
            score=float(value['score'])
            reason='Constrained JSON score from the frozen semantic judge'
            raw=json.dumps({'score':score,'reason':reason})
            key = hashlib.sha256(f'{VERSION}:{self.revision}:{prompt}'.encode()).hexdigest()
            results.append({'request_id':request.request_id, 'kind':request.kind,
                            'request':asdict(request), 'score':score, 'reason':reason,
                            'model':self.model_id, 'revision':self.revision, 'version':VERSION,
                            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                            'rendered_prompt':prompt,'raw_response':raw,'cache_key':key,'cached':False,
                            'generated_score_text':score_prefix+suffix,
                            'padded_length':inputs['input_ids'].shape[1]})
        return results

    def calibrate(self, audit=None):
        examples = [
            ('semantic','Notify me when the door opens.','A person opens the door.','A person opens the door.',True),
            ('semantic','Notify me when the door opens.','A dog sleeps on a sofa.','A person opens the door.',False),
            ('semantic','Notify me when a person sits.','The person remains standing.','A person sits down.',False),
            ('semantic','Notify me when a person sits.','The person takes a seat.','A person sits down.',True),
            ('repetition','Describe new events.','A person opens the door.','A person opens the door.',True),
            ('repetition','Describe new events.','A dog jumps into a pool.','A person opens the door.',False),
            ('repetition','Report object movements.','A cyclist rides away.','A train stops at a station.',False),
            ('repetition','Report object movements.','The red ball falls.','The red ball rolls along the table.',False),
            ('repetition','Describe activity.','A woman is drinking tea.','A woman drinks tea.',True),
            ('repetition','Describe activity.','A woman is drinking tea.','- none',False),
        ]
        requests = [JudgeRequest(*e[:4]) for e in examples]
        results = self.score_many(requests)
        if audit:
            Path(audit).write_text(json.dumps(results,indent=2)+'\n')
        if any((r['score'] < .75 if e[4] else r['score'] > .25)
               for r,e in zip(results,examples,strict=True)):
            raise RuntimeError(f'local judge calibration failed: {[(r["score"],r["reason"]) for r in results]}')
        return results

    def score_cached(self, requests, cache, audit):
        requests = [JudgeRequest(**r) for r in requests]
        results = [None]*len(requests)
        missing = []
        with sqlite3.connect(cache) as db:
            db.execute('CREATE TABLE IF NOT EXISTS jevany_judge_v2 (key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            for i,r in enumerate(requests):
                prompt=self.render(r)+'{"score": '
                key = hashlib.sha256(f'{VERSION}:{self.revision}:{prompt}'.encode()).hexdigest()
                row = db.execute('SELECT payload FROM jevany_judge_v2 WHERE key=?',(key,)).fetchone()
                if row:
                    value = json.loads(row[0])
                    parsed, _ = _parse_score(value['raw_response'])
                    if value['cache_key'] != key or parsed != value['score']:
                        raise RuntimeError('invalid local judge cache entry')
                    value['cached'] = True
                    results[i] = value
                else:
                    missing.append((i,r))
            with Path(audit).open('a') as log:
                for start in range(0,len(missing),16):
                    batch = missing[start:start+16]
                    scored = self.score_many([r for _,r in batch])
                    for (i,r),value in zip(batch,scored,strict=True):
                        db.execute('INSERT INTO jevany_judge_v2 VALUES (?,?)',
                                   (value['cache_key'],json.dumps(value)))
                        results[i] = value
                        log.write(json.dumps(value)+'\n')
                    log.flush()
                    db.commit()
        return results
