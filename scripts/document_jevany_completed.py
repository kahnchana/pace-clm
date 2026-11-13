#!/usr/bin/env python3
"""Generate the final experiment record only after training and judging finish."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32'
DOCUMENT=ROOT/'docs/experiments/jevany_2b_episode_balanced_mixed_results.md'
DOMAINS=('web_all','ego_all','web_heldout','ego_heldout','ours_heldout')
LABELS={'web_all':'Full WEB','ego_all':'Full EGO','web_heldout':'WEB held out','ego_heldout':'EGO held out','ours_heldout':'Original validation'}
REQUIRED_CHECKPOINT={'pointer_head.pt','training_state.pt','rl_state.json',
    'controller_adapter/adapter_model.safetensors',*[f'rng.rank{i}.pt' for i in range(7)]}


def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(value);tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
        *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])
def link(path):
    return f'[{path.name}]({path.resolve()})'


def completion_ready(run):
    paths=[run/'workflow-status.json',run/'latest.json',run/'step-000032/commit.json',run/'evaluation/full-step32/result.json']
    if not all(p.exists() for p in paths):return False,'waiting for checkpoint 32 and final evaluation'
    if read(paths[0]).get('state')!='complete' or read(paths[1]).get('step')!=32:
        return False,'waiting for successful workflow completion'
    final=read(paths[3]);cohorts=final.get('cohorts',{})
    commit=read(paths[2])
    if commit.get('step')!=32 or not REQUIRED_CHECKPOINT<=commit.get('files',{}).keys():
        return False,'checkpoint commit lacks required model/optimizer/RNG files'
    if any(cohorts.get(k,{}).get('episodes')!=n for k,n in [('web_all',500),('ego_all',326)]):
        return False,'final full-domain result is incomplete'
    if any(not (run/f'batch-{i:06d}/report.json').exists() for i in range(32)):
        return False,'waiting for all 32 committed update reports'
    if any(not (run/'evaluation'/p.stem/'result.json').exists() for p in (run/'queue').glob('*.json')):
        return False,'waiting for remaining queued evaluations'
    return True,'training and every queued evaluation complete'


def full_results(run):
    results={}
    for path in sorted((run/'evaluation').glob('*/result.json')):
        d=read(path)
        if d.get('job',{}).get('kind')=='full' and all(k in d.get('cohorts',{}) for k in ('web_all','ego_all')):
            results[d['job']['step']]={'path':path,'result':d}
    return dict(sorted(results.items()))


def pooled_domain(cohorts,domain):
    if domain+'_all' in cohorts:return cohorts[domain+'_all']['pauc_f']
    a,b=[cohorts[domain+'_'+split] for split in ('train','heldout')]
    precision=(a['useful']+b['useful'])/(a['fires']+b['fires']) if a['fires']+b['fires'] else 0.
    recall=(a['temporal_recall']*a['gold_events']+b['temporal_recall']*b['gold_events'])/(a['gold_events']+b['gold_events'])/100
    return 100*2*precision*recall/(precision+recall) if precision+recall else 0.


def training_rows(run):
    episodes=[];updates=[]
    for path in sorted(run.glob('batch-*/report.json')):
        report=read(path);step=report['step'];updates.append(report)
        traces=read(path.parent/'scored.json');groups={}
        for t in traces:groups.setdefault(t['spec']['span_id'],[]).append(t)
        for q,group in groups.items():
            scores=[t['reward_metrics']['pauc_f'] for t in group]
            rewards=[t['reward'] for t in group]
            if len(group)!=4:raise ValueError('expected exactly four trajectories per episode')
            episodes.append({'step':step,'domain':group[0]['spec']['domain'],'episode_id':q,
                'decisions_per_trajectory':len(group[0]['steps']),
                'gold_events':group[0]['reward_metrics']['num_gold_turns'],
                **{f'rollout_{i+1}_pauc_f':v for i,v in enumerate(scores)},
                **{f'rollout_{i+1}_reward':v for i,v in enumerate(rewards)},
                'mean_pauc_f':statistics.mean(scores),'best_pauc_f':max(scores),'min_pauc_f':min(scores),
                'mean_reward':statistics.mean(rewards),'pauc_f_flat_group':max(scores)==min(scores),
                'reward_flat_group':max(rewards)==min(rewards),
                'nonzero_advantage_trajectories':sum(t['steps'][0]['advantage']!=0 for t in group)})
    return episodes,updates


def write_csv(path,rows):
    if not rows:return
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def render(run,document,preview=False):
    ready,reason=completion_ready(run)
    if not preview and not ready:raise ValueError(reason)
    base=read(run/'evaluation/baseline-sft/result.json');results=full_results(run)
    if not results:raise ValueError('no completed full evaluations')
    last=max(results);latest=results[last]['result'];cohorts=latest['cohorts'];baseline=base['cohorts']
    episodes,updates=training_rows(run);cfg=read(run/'freeze.json');continuation=read(run/'continuation.json')
    source_ids={d:{r['episode_id'] for r in episodes if r['domain']==d} for d in ('ours','web','ego')}
    checkpoint=run/f'step-{last:06d}';commit=read(checkpoint/'commit.json')
    if ready:
        if last!=32:raise ValueError('final record must describe checkpoint 32')
        if sum(u['decisions'] for u in updates)!=cfg['episode_batches'][-1]['exposure_after']:
            raise ValueError('committed decision count differs from frozen schedule')
        for name,digest in commit['files'].items():
            target=checkpoint/name
            if not target.resolve().is_relative_to(checkpoint.resolve()):raise ValueError('invalid committed file path')
            if sha(target)!=digest:raise ValueError('final checkpoint integrity failed: '+name)
    now=datetime.now(timezone.utc).isoformat();artifacts=run/'final-documentation';artifacts.mkdir(exist_ok=True)
    summary=[]
    summary.append(('Completed' if ready else 'PRELIMINARY: latest completed full evaluation is')+f' step {last}.')
    for domain in ('web_all','ego_all','web_heldout','ego_heldout'):
        x=cohorts[domain]['pauc_f'];b=baseline[domain]['pauc_f']
        summary.append(f"{LABELS[domain]} PAUC-F: {x:.2f} versus SFT {b:.2f} ({x-b:+.2f} points).")
    if not ready:summary.append('Training/final judging are incomplete; this is not the final result.')
    else:
        summary.append('Combined WEB/EGO scores include training identities; heldout splits are the transfer check.')
        for domain in ('web_heldout','ego_heldout'):
            ci=latest.get('paired_vs_sft',{}).get(domain,{}).get('paired_95_ci')
            if ci:
                assessment='includes zero' if ci[0]<=0<=ci[1] else 'is positive' if ci[0]>0 else 'is negative'
                summary.append(f"{LABELS[domain]} paired 95% interval [{ci[0]:+.2f}, {ci[1]:+.2f}] {assessment} (conditional on saved judgments).")
    write(artifacts/'summary.txt','\n'.join(summary)+'\n')
    rows=[]
    for step,item in results.items():
        d=item['result']
        for domain in DOMAINS:
            c=d['cohorts'][domain];paired=d.get('paired_vs_sft',{}).get(domain,{})
            interval=paired.get('paired_95_ci',[None,None])
            rows.append({'step':step,'cohort':domain,'episodes':c['episodes'],'pauc_f':c['pauc_f'],
                'sft_pauc_f':baseline[domain]['pauc_f'],'difference_points':c['pauc_f']-baseline[domain]['pauc_f'],
                'paired_95_low':interval[0],'paired_95_high':interval[1],
                'useful':c['useful'],'wasted':c['wasted'],'fires':c['fires'],
                'fire_precision':c['fire_precision'],'temporal_recall':c['temporal_recall']})
    write_csv(artifacts/'full-results.csv',rows);write_csv(artifacts/'training-episode-rollouts.csv',episodes)
    heading='Completed experiment' if ready else 'PRELIMINARY — awaiting step 32 and final judging'
    text=['# Episode-balanced WEB/EGO RLOO: results and experiment record','',f'**{heading}.** Updated {now}.','',
        '\n\n'.join(summary),'','## What was tested','',
        'The experiment tested whether changing 50/25/25 original/WEB/EGO sampling from decision counts to episode counts improves EGO learning while retaining WEB and original-data quality. It began fresh from Qwen3.5-2B J1-S SFT step 8243. The continuation restores the same learned run, rather than restarting from SFT.',
        '', 'Each update collects four original episodes, two WEB episodes and two EGO episodes, with four complete causal trajectories per episode. The full 32-update schedule contains 138,460 decisions and 128/64/64 distinct episodes. The original 23-update schedule prefix is preserved exactly.',
        '', 'This is a practical sampling comparison, not an isolated causal estimate of episode diversity: source decision shares, identities, optimizer-update counts and compute exposure differ from the earlier decision-balanced run.',
        '', '## Completed full evaluations','',
        table(['Checkpoint','WEB (500)','EGO (326)','WEB held out (100)','EGO held out (181)','Original validation (128)'],
            [['SFT',*[f"{baseline[k]['pauc_f']:.2f}" for k in DOMAINS]]]+[
                [str(step),*[f"{item['result']['cohorts'][k]['pauc_f']:.2f}" for k in DOMAINS]] for step,item in results.items()]),
        '', 'Scores use official GPT-4.1 judged PAUC-F, omega 0.5, and raw learned argmax with response margin zero. Combined domain scores pool precision counts and gold-weighted recall before calculating PAUC-F; they are not averages of split PAUC-F values.',
        '', '## Latest checkpoint against SFT','',
        table(['Cohort','PAUC-F','Change','Paired 95% interval','Useful: SFT → checkpoint','Wasted: SFT → checkpoint'],[
            [LABELS[k],f"{cohorts[k]['pauc_f']:.2f}",f"{cohorts[k]['pauc_f']-baseline[k]['pauc_f']:+.2f}",
             ('['+', '.join(f'{v:+.2f}' for v in latest['paired_vs_sft'][k]['paired_95_ci'])+']') if k in latest.get('paired_vs_sft',{}) else 'not saved',
             f"{baseline[k]['useful']} → {cohorts[k]['useful']}",f"{baseline[k]['wasted']} → {cohorts[k]['wasted']}"] for k in DOMAINS]),
        '', 'Intervals are paired episode bootstraps conditional on saved judgments and reused cohort selection. They do not capture judge uncertainty. Selecting the best observed checkpoint after evaluation is exploratory; every evaluated checkpoint is shown above.',
        '', 'Best observed combined WEB checkpoint: step '+str(max(results,key=lambda s:results[s]['result']['cohorts']['web_all']['pauc_f']))+
        '; best observed combined EGO checkpoint: step '+str(max(results,key=lambda s:results[s]['result']['cohorts']['ego_all']['pauc_f']))+
        '. These are posthoc observations on benchmarks that include training identities, not independent model-promotion evidence.',
        '', '## Exposure and overlap','',
        table(['Source','Distinct episodes committed','RL decisions committed'],[[d,len(source_ids[d]),sum(u['episode_schedule']['domain_decisions'][d] for u in updates)] for d in ('ours','web','ego')]),
        '', 'Allowed source pools are 8,321 original episodes, 400 WEB training videos and 145 EGO training videos. WEB heldout 100, EGO heldout 181 and original validation 128 are excluded from this RL run. Separate sampled/unsampled training-pool summaries are in the saved evaluation JSON. Those labels refer to the planned schedule; the committed exposure counts above are actual.',
        '', 'Combined full-WEB/full-EGO results include training examples and do not establish a new heldout quality floor. The historical 54.67 full-WEB floor remains a historical reference, not an untouched-test comparison for mixed training. Historical benchmark reuse also precludes an untouched-test claim for the heldout cohorts. Perfect20 is a selected diagnostic.',
        '', '## Training and inference recipe','',
        '- RLOO with four complete trajectories per episode. Event-containing episodes use official PAUC-F/100 rewards; genuinely event-free episodes receive -0.25 per physical fire. Trajectory-relative advantages are shared across that trajectory’s decisions.',
        '- Source objective: 0.5 mean original episode loss + 0.25 mean WEB + 0.25 mean EGO, with the existing constant-32 normalization.',
        '- Head/LoRA learning rates 1e-5/1e-6; KL 0.02; entropy 0.001; weight decay 0.01; gradient clipping 1; dropout zero; one optimizer step per collection.',
        '- Decoder LoRA and pointer head learn. Base/vision/reference and promoted answer-generator weights remain frozen.',
        '- Supervised replay coefficient 0.0035140820988695176; 126 credits per 4,096 RL decisions. The finite replay bank was extended by repeating its original prefix for the final 227 credits.',
        '- Native videos sampled at 1 FPS using canonical physical-frame mappings; native WEB/EGO reply windows retained. Original examples retain trusted four-second windows. No future frames or gold answers enter policy inputs.',
        '- Every decision uses at most 32 recent frames, timestamps, self-generated action/answer history and the latest aged-out response memory. Decoder context is rebuilt per decision; an entire six-minute EGO video is not accumulated in one prompt.',
        '- RL actions are sampled directly from learned probabilities. Evaluation uses raw learned argmax, margin zero. No action offsets, calibration, thresholds, cooldowns or inference-time suppression were added. Frozen answer generation is greedy, at most 96 new tokens.',
        '', 'The inherited v2 last-response memory prompt discourages repetition and explicitly favors silence later in videos. It was held fixed in this experiment; its contribution to EGO behavior remains unproven and was not isolated by this sampling experiment.',
        '', '## GPU layout and evaluation workflow','',
        'Training: allocation 38177, ip-10-3-120-167, seven controller/learner GPUs and one frozen answer-generator GPU. Evaluation: allocation 38226, ip-10-3-70-177, eight independent video workers. Learner correctness checks stay on the training node. GPT judging uses the authorized gateway and separate training/evaluation caches. No additional nodes were allocated.',
        '', 'Immutable checkpoints include model, optimizer, per-rank RNG and sampler state. Evaluations consume committed checkpoints asynchronously. The original pilot evaluated full milestones 6/12 and its stopped final 15; the extension evaluates full 16/24/32 and fixed subsets 20/28. A full step-30 evaluation was also completed after the technical interruption. Full evaluations are retained; stale fast probes may be superseded. Training waits when pending full evaluation is more than 10k decisions behind, allowing one update’s overshoot.',
        '', '## Operational history and repairs','',
        '1. Initial Slurm steps exposed a single-CPU task affinity despite eight assigned GPUs. They were restarted before any update with CPU binding disabled, exposing 96 CPUs.',
        '2. At the first cross-node gate, identical weights, RGB pixels and token inputs produced different vision embeddings. The evaluation process inherited host CUDA libraries and loaded cuDNN 9.7.1 versus training’s 9.1. Clearing LD_LIBRARY_PATH restored exact replay on all seven batches. Startup now asserts PyTorch 2.5.1+cu121, CUDA 12.1 and cuDNN 90100. Affected v2 gate/baseline outputs were archived and regenerated; original v1 cache reuse retained its exact cross-node gate and unchanged model/input/scorer provenance.',
        '3. The preset quality rule stopped the original pilot at update 15 after full steps 6 and 12 had EGO-heldout useful counts 239/235 versus SFT 280, below 90% twice. The user explicitly overrode quality stopping and extended the same learned run to update 32. Coverage/silence diagnostics became warnings; correctness checks were retained.',
        '4. Update 31 hit a supervised-replay bank bound: the bank contained 4,032 rows, while the longer budget requires 4,259 credits. The initial extension missed this capacity check. The repair preserves the original bank and appends its first 227 examples in order, keeping all prior replay selections and coefficients unchanged. The failed uncommitted batch and previous freezes/source were archived, then checkpoint 30’s model/optimizer/RNG/sampler state was restored.',
        '', 'The host NVIDIA drivers differ (570.133.20 training / 595.91.07 evaluation). The confirmed mismatch was the loaded cuDNN library; the original unchanged probe became exact after matching that library. No driver change or relaxed numerical tolerance was used.',
        '', '## Training rollout outcomes','',
        'Each row below contains two different EGO episodes, four trajectories each. Best-of-four is a sampled diagnostic, not raw-argmax evaluation. Episode difficulty changes between updates, so these rows cannot establish a learning curve on a fixed cohort. The accompanying CSV preserves all four scores for every source and episode.',
        '', table(['Source','Episode groups','Mean episode PAUC-F','Mean best-of-four','Flat reward groups','Nonzero-advantage trajectories'],[
            [d,len([e for e in episodes if e['domain']==d]),
             f"{statistics.mean(e['mean_pauc_f'] for e in episodes if e['domain']==d):.2f}",
             f"{statistics.mean(e['best_pauc_f'] for e in episodes if e['domain']==d):.2f}",
             f"{100*statistics.mean(e['reward_flat_group'] for e in episodes if e['domain']==d):.1f}%",
             f"{100*sum(e['nonzero_advantage_trajectories'] for e in episodes if e['domain']==d)/(4*len([e for e in episodes if e['domain']==d])):.1f}%"] for d in ('ours','web','ego')]),
        '', 'Reward flatness uses actual training rewards. For event-free episodes, PAUC-F is zero while the negative-fire reward can still vary; the CSV preserves both quantities.',
        '', table(['Step','EGO episode 1 best','EGO episode 2 best','Mean best-of-four','Mean all eight'],[
            [step,*[f"{r['best_pauc_f']:.2f}" for r in [e for e in episodes if e['step']==step and e['domain']=='ego']],
             f"{statistics.mean(e['best_pauc_f'] for e in episodes if e['step']==step and e['domain']=='ego'):.2f}",
             f"{statistics.mean(e['mean_pauc_f'] for e in episodes if e['step']==step and e['domain']=='ego'):.2f}"] for step in sorted({e['step'] for e in episodes})]),
        '', '## Runtime and correctness accounting','',
        f"Committed updates: {len(updates)}. Successful-update time: {sum(u['seconds'] for u in updates)/3600:.2f} hours; successful-update eight-GPU cost: {sum(u['gpu_hours'] for u in updates):.2f} GPU-hours. This excludes model loading, restore gates, idle evaluation backpressure, failed attempts and standalone evaluation; it is not total allocation cost.",
        '', f"Maximum recorded rollout/replay probability error across committed updates: {max(v['parity_probability_max'] for u in updates for v in u['learner'].values()):.6g}. Final checkpoint file hashes {'verified' if ready else 'will be verified on completion'}. Source manifests, schedule, runtime contract, freezes, replay repair and checkpoint commits are retained."]
    temporal=latest.get('ego_temporal',{}).get('buckets',[])
    if temporal:
        text+=['','## EGO response timing at the latest full checkpoint','',table(['Video seconds','Decisions','Fires','Useful','Wasted','Mean Response probability'],[
            [b['seconds'],b['decisions'],b['fires'],b['useful'],b['wasted'],f"{100*b['mean_response_probability']:.2f}%" if b.get('mean_response_probability') is not None else 'not recorded'] for b in temporal]),
            '', 'These are descriptive temporal buckets; event density and episode content differ across video time. They do not by themselves prove a causal time-dependent suppression effect.']
    prior=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-rloo-v1';priorrows=[]
    for name in ('full-step5','full-step10','full-step14'):
        f=prior/'evaluation'/name/'result.json'
        if f.exists():
            d=read(f);priorrows.append([d['job']['step'],f"{pooled_domain(d['cohorts'],'web'):.2f}",f"{pooled_domain(d['cohorts'],'ego'):.2f}"])
    ref=run/'evaluation/reference-mixed-v1-step16/result.json'
    if ref.exists():
        d=read(ref);priorrows.append([16,f"{d['cohorts']['web_all']['pauc_f']:.2f}",f"{d['cohorts']['ego_all']['pauc_f']:.2f}"])
    text+=['','## Earlier decision-balanced mixed run','',table(['Checkpoint','Full WEB','Full EGO'],priorrows),
        '', 'This is a different training recipe and episode exposure, shown for context rather than pooled with the episode-balanced run.',
        '', '## Artifacts and provenance','',
        '\n'.join('- '+link(p) for p in [run/'continuation.json',run/'freeze.json',run/'workflow-freeze.json',run/'runtime-contract.json',run/'replay-bank-repair.json',checkpoint/'commit.json',results[last]['path'],artifacts/'full-results.csv',artifacts/'training-episode-rollouts.csv',artifacts/'summary.txt']),
        '', 'Related records: [experiment plan](../plans/jevany_2b_episode_balanced_mixed_rloo.md), [original mixed plan](../plans/jevany_2b_mixed_web_ego_rloo.md), and [experiment log](experiment_log.md).','']
    write(document,'\n'.join(text))
    payload={'complete':ready,'updated_utc':now,'step':last,'document':str(document),'summary':summary,
        'final_checkpoint':str(checkpoint),'commit_sha256':sha(checkpoint/'commit.json'),
        'generator_sha256':sha(Path(__file__)),'source_results':{str(k):{'path':str(v['path']),'sha256':sha(v['path'])} for k,v in results.items()}}
    write(artifacts/'summary.json',json.dumps(payload,indent=2)+'\n')
    if ready:
        log=ROOT/'docs/experiments/experiment_log.md';marker='<!-- jevany-episode-balanced-v2-to32-complete -->'
        if marker not in log.read_text():
            entry=['',marker,'### '+now[:10]+' — Episode-balanced mixed RLOO step 32 complete','',
                '- Completed 32 updates / '+str(sum(u['decisions'] for u in updates))+' RL decisions after the user-authorized coverage-stop override and replay-bank repair.',
                '- Final full WEB/EGO PAUC-F: '+f"{cohorts['web_all']['pauc_f']:.4f} / {cohorts['ego_all']['pauc_f']:.4f}"+
                '; heldout WEB/EGO: '+f"{cohorts['web_heldout']['pauc_f']:.4f} / {cohorts['ego_heldout']['pauc_f']:.4f}"+'. Combined scores include training identities.',
                '- All queued evaluations and final judging completed; checkpoint 32 hashes verified. Training remained on allocation 38177 and evaluation on 38226, with raw learned argmax / margin zero.',
                '- See [the final experiment record]('+document.name+') for the full recipe, every full result, paired comparisons, trajectory diagnostics, operational failures/fixes and artifact provenance.','']
            with log.open('a') as handle:handle.write('\n'.join(entry))
        write(run/'completion-documentation-status.json',json.dumps({'state':'complete','utc':now,'document':str(document),'summary':str(artifacts/'summary.txt'),'step':32},indent=2)+'\n')
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--document',type=Path,default=DOCUMENT)
    p.add_argument('--watch',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args();run=a.run.resolve()
    if a.watch:
        last=None
        while True:
            ready,reason=completion_ready(run)
            if ready:break
            workflow=read(run/'workflow-status.json') if (run/'workflow-status.json').exists() else {}
            state={'state':'waiting_for_completion','workflow_state':workflow.get('state'),'reason':reason}
            if state!=last:
                out={**state,'utc':datetime.now(timezone.utc).isoformat()};print(json.dumps(out),flush=True)
                write(run/'completion-documentation-status.json',json.dumps(out,indent=2)+'\n');last=state
            time.sleep(30)
    if a.watch:write(run/'completion-documentation-status.json',json.dumps({'state':'generating_final_documentation','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    try:result=render(run,a.document.resolve(),preview=a.preview)
    except Exception as exc:
        if a.watch:write(run/'completion-documentation-status.json',json.dumps({'state':'failed','utc':datetime.now(timezone.utc).isoformat(),'error':str(exc)},indent=2)+'\n')
        raise
    print(json.dumps({'complete':result['complete'],'step':result['step'],'document':result['document']}),flush=True)


if __name__=='__main__':main()
