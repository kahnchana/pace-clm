#!/usr/bin/env python3
"""Bounded whole-episode RLOO workflow on the original eight-H200 allocation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from jevany_rl_reward import ROOT, SG, sha, write
from jevany_episode_reward import contract
from pace_clm.rl_episode import draw_schedule, ANCHOR_COEFFICIENT
from run_jevany_e2_rloo import coverage_decision
from run_jevany_rl_full_current_node import checkpoint_view
from run_qwen35_control import verify_web
from run_jevany_e2_full_web import validate_judgments, rows, write_rows, score as score_full_web, PROTOCOL_KEYS, BASELINE
from zs_infer.pauc_f import score_pauc_f

PY=SG/'.venv/bin/python'
RLOO=ROOT/'outputs/jevany-2b-rl/reward-aligned-rloo-v1'
ANCHOR=ROOT/'outputs/jevany-2b-rl/rloo-sft-anchor-v1'
SFT=ROOT/'outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243'
DEFAULT=ROOT/'outputs/jevany-2b-rl/episode-rloo-v2'
NODE='ip-10-3-120-167'


def read(path):return json.loads(Path(path).read_text())


def training_args(run,step):
    argv=[PY,'-u',ROOT/'scripts/run_jevany_episode_rloo.py','--output',run,
          '--checkpoint',SFT,'--spans',run/'data/episodes.jsonl',
          '--anchor-manifest',ANCHOR/'anchor/replay.jsonl',
          '--learners','7','--rollouts','4','--starts','2',
          '--physical-batch','24','--microbatch','24','--collection-batch','24',
          '--frame-cache','2048','--updates',str(step),'--full-run',
          '--decision-budget',str(131072-read(run/'prior-gate-exposure.json')['collected_decisions']),
          '--reward-version','pauc-f-aligned-gpt-v1']
    if (run/'freeze.json').exists():argv.append('--resume')
    return argv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    development=run/'development';development.mkdir(exist_ok=True)
    for name in ('selection.json','freeze.json','sft-baseline.json'):
        source=RLOO/'development'/name;target=development/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('development selection changed')
        if not target.exists():shutil.copy2(source,target)
    prior=read(run/'prior-gate-exposure.json')
    if prior['committed_optimizer_updates']!=0:raise ValueError('retry requires fresh SFT after zero committed updates')
    episodes=rows(run/'data/episodes.jsonl');batches=draw_schedule(episodes,budget=131072-prior['collected_decisions'])
    milestones=[]
    for target in (32768,65536,98304):
        step=next((i+1 for i,b in enumerate(batches) if b['exposure_after']>=target),len(batches))
        if step not in milestones:milestones.append(step)
    if len(batches) not in milestones:milestones.append(len(batches))
    audit=read(run/'data/audit.json')
    if audit['true_event_free_episodes']:
        raise ValueError('event-free duration scale requires an explicit revised preparation review')
    anchor=read(ANCHOR/'anchor/freeze.json')
    if not anchor['training_only'] or anchor['benchmark_overlap'] or sha(ANCHOR/'anchor/replay.jsonl')!=anchor['replay_sha256']:
        raise ValueError('supervised replay provenance invalid')
    proof=read(run/'replay-layout-proof.json')['results'][0]
    if (proof['variant']!='collected-layout' or proof['probability_max']>.005 or
            proof['kl_max']>1e-4 or proof['kl_mean']>1e-5 or proof['selected_logp_max']>.05):
        raise ValueError('collected-layout replay diagnostic must pass unchanged tolerances')
    sources=[Path(__file__),ROOT/'scripts/run_jevany_episode_rloo.py',
             ROOT/'scripts/jevany_rl_episode_workers.py',ROOT/'scripts/jevany_episode_reward.py',
             ROOT/'scripts/verify_jevany_episode_resume.py',ROOT/'scripts/report_jevany_episode_rl.py',
             ROOT/'scripts/prepare_jevany_episode_rl.py',ROOT/'scripts/audit_jevany_episode_aggregation.py',
             ROOT/'src/pace_clm/rl_episode.py',ROOT/'src/pace_clm/rl.py',
             ROOT/'scripts/probe_jevany_episode_layout.py',ROOT/'tests/test_rl_episode.py',run/'replay-layout-proof.json',
             ROOT/'src/pace_clm/rl_anchor.py',ROOT/'scripts/jevany_rl_reward.py',
             ROOT/'scripts/eval_jevany_streaming.py',ROOT/'scripts/run_jevany_e2_full_web.py',
             ROOT/'scripts/jevany_rl_workers.py',ROOT/'scripts/jevany_rl_generator_sidecar.py',
             ROOT/'scripts/jevany_rl_rpc.py',ROOT/'scripts/report_jevany_e2.py',
             ROOT/'scripts/run_qwen35_control.py',ROOT/'scripts/run_jevany_rl_full_current_node.py',
             ROOT/'src/pace_clm/jevany_streaming.py',ROOT/'src/pace_clm/qwen_controller_runtime.py',
             ROOT/'scripts/benchmark_qwen35_2b_training.py',ROOT/'scripts/train_jevany_streaming.py',
             ROOT/'scripts/merge_jevany_benchmark.py',ROOT/'scripts/judge_exp10_utility.py',
             SG/'zs_infer/pauc_f.py',SG/'zs_infer/scoring.py',
             run/'data/episodes.jsonl',run/'data/audit.json',run/'aggregation-audit.json',
             run/'prior-gate-exposure.json',
             SFT/'controller_adapter/adapter_model.safetensors',SFT/'pointer_head.pt',
             ANCHOR/'anchor/replay.jsonl',ANCHOR/'anchor/freeze.json']
    hashes={str(p):sha(p) for p in sources}
    plan={'node':NODE,'allocation':38177,'gpus':list(range(8)),
          'gpu_roles':{'learners':list(range(7)),'frozen_generator':[7]},
          'initial_checkpoint':str(SFT),'reward':contract(),'episode_batches':batches,
          'budget_decisions_max':131072,'actual_scheduled_decisions':batches[-1]['exposure_after'],
          'prior_failed_gate_collected_decisions':prior['collected_decisions'],
          'total_collected_decisions_cap':prior['collected_decisions']+batches[-1]['exposure_after'],
          'evaluation_steps':milestones,'evaluation_exposures':[batches[s-1]['exposure_after'] for s in milestones],
          'loss':'global sum / (global trajectories * constant 32); no individual length normalization',
          'replay_layout':'complete original collected batches, row order and padding; no regrouping',
          'anchor_coefficient':ANCHOR_COEFFICIENT,'replay_credit':'floor(cumulative decisions * 126 / 4096); original frozen row order',
          'resume':'restart uncommitted batch at last published optimizer commit; no mid-episode restore',
          'source_hashes':hashes,'inference':'raw learned argmax; response_margin=0',
          'automatic_extension':False,'quality_floor':54.67,'minimum_full_web_useful':631,
          'minimum_subset_useful':166,'subset_size':128,'full_web_size':500,
          'seed':42,'actor_passes':1,'temperature_train':1.,'head_lr':1e-5,'lora_lr':1e-6,
          'weight_decay':.01,'kl_beta':.02,'entropy_beta':.001,'dropout':0.,'grad_clip':1.,
          'physical_batch':24,'microbatch':24,'collection_batch':24,'frame_context':32,'fps':1,'visual_tokens_per_frame':144}
    path=run/'workflow-freeze.json'
    if path.exists() and read(path)!=plan:raise ValueError('frozen workflow changed')
    if not path.exists():write(path,plan)
    snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in hashes.items():
        target=snapshot/(digest+Path(name).suffix)
        if not target.exists():shutil.copyfile(name,target)
    if args.prepare_only:print(json.dumps({'evaluation_steps':milestones,'exposures':plan['evaluation_exposures'],'batches':len(batches)}));return
    if socket.gethostname()!=NODE:raise RuntimeError('authorized node only')
    allocation=subprocess.check_output(['squeue','-h','-j','38177','-o','%T %N'],text=True).strip()
    if allocation!='RUNNING '+NODE:raise RuntimeError('original allocation unavailable')
    if len(os.sched_getaffinity(0))<8:raise RuntimeError('requires at least eight CPU cores')
    logs=run/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        TORCHINDUCTOR_COMPILE_THREADS='4',PACE_DISTRIBUTED_BACKEND='gloo',
        TRITON_CACHE_DIR='/tmp/pace-rl-triton',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=':'.join(map(str,[ROOT/'outputs/jevany/deps032',ROOT/'src',ROOT/'scripts',SG/'src',SG])))

    def status(state,**fields):
        record={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'node':NODE,'pid':os.getpid(),**fields}
        write(run/'workflow-status.json',record)
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)

    def command(argv,log):
        for name,digest in hashes.items():
            if sha(name)!=digest:raise ValueError('frozen source changed: '+name)
        print(json.dumps({'command':list(map(str,argv)),'log':str(logs/log)}),flush=True)
        with (logs/log).open('a') as f:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

    def evaluate(step):
        full=run/f'evaluation/web-step{step}';full.mkdir(parents=True,exist_ok=True)
        output=run/f'evaluation/dev-step{step}';output.mkdir(parents=True,exist_ok=True)
        if not (full/'pauc_f_summary-gpt-4.1-v1.json').exists():
            view=checkpoint_view(run,run/f'step-{step:06d}')
            command([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                ROOT/'scripts/eval_jevany_streaming.py','--controller',view,'--benchmark','web',
                '--output-dir',full,'--response-margin','0','--answer-model','promoted',
                '--max-new-tokens','96','--record-action-logits'],f'web-step{step}-rollout.log')
            command([PY,ROOT/'scripts/merge_jevany_benchmark.py','--input-dir',full],f'web-step{step}-merge.log')
            status('gpt_scoring_full_web',round=step,videos=500)
            score_full_web(full,SG/'datasets/proactive-videoqa-web/anno.json',RLOO/'full_web_v1/gpt-cache.sqlite')
        verify_web(full)
        protocol=read(full/'manifest.json')
        control=read(BASELINE/'manifest.json')
        if any(protocol[key]!=control[key] for key in PROTOCOL_KEYS):
            raise ValueError('full-WEB RLOO anchor generation protocol differs from SFT control')
        selected=read(development/'selection.json');wanted=set(selected)
        predictions={r['question_id']:r for r in rows(full/'predictions.jsonl') if r['question_id'] in wanted}
        scored={r['question_id']:r for r in read(full/'pauc_llm_scores-gpt-4.1.json') if r['question_id'] in wanted}
        if len(predictions)!=128 or len(scored)!=128:raise ValueError('incomplete frozen development subset')
        write_rows(output/'predictions.jsonl',[predictions[q] for q in selected])
        write(output/'pauc_llm_scores-gpt-4.1.json',[scored[q] for q in selected])
        write(output/'manifest.json',{**protocol,'rows':128,'question_ids':selected})
        requests=[r for r in rows(full/'pauc_batch_input-gpt-4.1.jsonl') if '='.join(r['custom_id'].split('=')[:-2]) in wanted]
        responses=[r for r in rows(full/'pauc_batch_output-gpt-4.1.jsonl') if '='.join(r['custom_id'].split('=')[:-2]) in wanted]
        validate_judgments(requests,responses)
        write_rows(output/'pauc_batch_input-gpt-4.1.jsonl',requests)
        write_rows(output/'pauc_batch_output-gpt-4.1.jsonl',responses)
        score=score_pauc_f(output/'predictions.jsonl',output/'pauc_llm_scores-gpt-4.1.json',omega=.5,judge_model='gpt-4.1')
        write(output/'pauc_f_summary-gpt-4.1-v1.json',score)
        write(output/'provenance.json',{'source':str(full),'same_saved_outputs_and_judgments':True,
              'selected_ids_sha256':sha(development/'selection.json')})
        return score

    try:
        status('correctness_and_performance_gate')
        if not (run/'latest.json').exists():command(training_args(run,1),'gate-first-batch.log')
        if not (run/'gate-passed.json').exists():
            command([PY,ROOT/'scripts/verify_jevany_episode_resume.py','--run',run],'gate-restore.log')
            restored=read(run/'resume_verification.json');report=read(run/'batch-000000/report.json')
            learners=list(report['learner'].values())
            if not restored['verified'] or restored['additional_optimizer_steps']!=0 or restored['step']!=1:
                raise ValueError('restore gate failed')
            if report['decisions']!=batches[0]['decisions'] or sum(v['anchor_examples'] for v in learners)!=batches[0]['anchor_count']:
                raise ValueError('episode/replay exposure mismatch')
            if any(v['head_grad_norm']<=0 or v['lora_grad_norm']<=0 or abs(v['kl_sum'])>1e-4 or v['parity_probability_max']>.005 or v['reserved_gib']>=115 for v in learners):
                raise ValueError('initial KL, gradient, parity or memory gate failed')
            if report['generator_broker']['logical_requests']!=report['fires']:raise ValueError('duplicate/missing generator fires')
            gate={'verified':True,'restore':restored,'decisions':report['decisions'],
                  'measured_decisions_per_second':report['decisions']/report['seconds'],
                  'max_reserved_gib':max(v['reserved_gib'] for v in learners),
                  'frozen_concurrency':{'episode_groups_by_batch':[len(b['episode_indices']) for b in batches],
                                        'physical_batch':24,'collection_batch':24},
                  'estimated_core_gpu_hours':report['gpu_hours']*batches[-1]['exposure_after']/report['decisions']}
            write(run/'gate-passed.json',gate);status('gate_passed',**{k:v for k,v in gate.items() if k!='restore'})
        baseline=read(development/'sft-baseline.json');full_baseline=read(BASELINE/'pauc_f_summary-gpt-4.1-v1.json')
        checks=[];full_checks=[]
        for step in milestones:
            latest=read(run/'latest.json')
            if latest['step']<step:
                status('training',completed_updates=latest['step'],target_update=step,
                       target_decisions=batches[step-1]['exposure_after'])
                command(training_args(run,step),f'train-to-{step}.log')
            status('full_web_evaluation',step=step,decisions=batches[step-1]['exposure_after'],videos=500)
            subset=evaluate(step)
            checks.append((step,subset));write(development/'checks.json',checks)
            full_checks.append((step,read(run/f'evaluation/web-step{step}/pauc_f_summary-gpt-4.1-v1.json')))
            write(development/'full-web-checks.json',full_checks)
            pause=coverage_decision(baseline,checks) or coverage_decision(full_baseline,full_checks)
            if pause:
                status('stopped_by_coverage_gate',reason=pause[0],step=pause[1])
                command([PY,ROOT/'scripts/report_jevany_episode_rl.py','--run',run],'report.log');return
            command([PY,ROOT/'scripts/report_jevany_episode_rl.py','--run',run],f'report-step{step}.log')
        status('complete',optimizer_steps=len(batches),sampled_decisions=batches[-1]['exposure_after'],automatic_extension=False)
        command([PY,ROOT/'scripts/report_jevany_episode_rl.py','--run',run],'report.log')
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
