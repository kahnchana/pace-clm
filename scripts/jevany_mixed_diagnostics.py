"""Read-only diagnostics for saved causal rollouts; never change model inputs."""
import math
import re
import statistics
from zs_infer.pauc_f import assign_value_adding_fires


def training_diagnostics(traces):
    result={}
    for domain in ('ours','web','ego'):
        subset=[t for t in traces if t['spec']['domain']==domain];groups={}
        bins=[{'decisions':0,'fires':0,'response_probability_sum':0.,'max_padded_tokens':0} for _ in range(3)]
        for t in subset:
            groups.setdefault(t['spec']['span_id'],[]).append(100*t['reward'])
            for i,s in enumerate(t['steps']):
                b=bins[min(2,3*i//len(t['steps']))];b['decisions']+=1;b['fires']+=s['action']==2
                b['response_probability_sum']+=math.exp(s['old_logp'][2])
                b['max_padded_tokens']=max(b['max_padded_tokens'],s['row']['padded_length'])
        result[domain]={'episodes':[{'span_id':q,'four_pauc_f':v,'mean_pauc_f':statistics.mean(v),
            'best_pauc_f':max(v),'flat':max(v)==min(v)} for q,v in groups.items()],
            'temporal_thirds':bins}
    return result


def temporal_diagnostics(predictions,specs,scored):
    by_id={r['question_id']:r for r in scored}
    buckets=[{'seconds':label,'decisions':0,'fires':0,'useful':0,'wasted':0,
        'response_probability_sum':0.,'decisions_with_aged_memory':0,'memory_age_seconds_sum':0.,
        'input_tokens_max':0,'input_frames_max':0,'event_windows':0,'windows_with_fire':0,
        'repeated_event_descriptions':0} for label in ('0–120','120–240','240–360+')]
    for pred in predictions:
        q=pred['question_id'];spec=specs[q]
        if spec['domain']!='ego':continue
        fires=[float(r['time']) for r in pred['model_response_list']]
        useful=assign_value_adding_fires(fires,by_id[q]['answer'])
        for i,t in enumerate(fires):
            b=buckets[min(2,int(t//120))];b['fires']+=1;b['useful']+=i in useful;b['wasted']+=i not in useful
        seen=set()
        for gold in spec['gold_answers']:
            left,right=gold['reply_timespan'];b=buckets[min(2,int(left//120))]
            b['event_windows']+=1;b['windows_with_fire']+=any(left<=t<=right for t in fires)
            content=' '.join(gold['content'].lower().split())
            b['repeated_event_descriptions']+=content in seen;seen.add(content)
        for item in pred.get('jevany_action_trace',[]):
            t=spec['timestamps'][item['frame_index']];b=buckets[min(2,int(t//120))]
            b['decisions']+=1;logits=item['logits'];mx=max(logits);p=[math.exp(v-mx) for v in logits]
            b['response_probability_sum']+=p[2]/sum(p)
            b['input_tokens_max']=max(b['input_tokens_max'],item.get('input_tokens',0))
            b['input_frames_max']=max(b['input_frames_max'],item.get('input_frames',0))
            memory=item.get('memory_question','').partition('Prior response memory:')[2]
            match=re.search(r'At (\d+):(\d+(?:\.\d+)?):',memory)
            if match:
                b['decisions_with_aged_memory']+=1
                b['memory_age_seconds_sum']+=t-(60*int(match[1])+float(match[2]))
    for b in buckets:
        b['fire_rate']=b['fires']/b['decisions'] if b['decisions'] else None
        b['mean_response_probability']=b['response_probability_sum']/b['decisions'] if b['decisions'] else None
        b['mean_memory_age_seconds']=b['memory_age_seconds_sum']/b['decisions_with_aged_memory'] if b['decisions_with_aged_memory'] else None
    return {'buckets':buckets,'scope':'saved trajectories; window counts assigned by start time; repeated means identical normalized gold text; cached traces may lack token/frame diagnostics'}
