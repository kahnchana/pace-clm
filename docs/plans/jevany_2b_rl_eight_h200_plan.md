# JevAny Qwen3.5-2B RL: eight-H200 implementation and pilot plan

Date: 2026-11-06; completion/results updated 2026-11-08.
Root: `/fsx/home/kranasinghe/repo/pace-clm`.
Status: **E1 completed on 2026-11-07 at 23:12:05 UTC**. The continuation and
both full WEB evaluations used only the eight H200s on `ip-10-3-120-167`,
inside existing allocation 38177. Corrected step 1 was included in the
32-update total; full GPT-scored WEB evaluations completed at steps 16 and 32.

The canonical [completed experiment report](../experiments/jevany_2b_rl_e1_results.md)
records **53.31 / 53.09 PAUC-F at steps 16/32 versus SFT 54.67**, sample counts,
actual objectives, efficiency, diagnostics and artifact links. Neither RL
checkpoint improved PAUC-F. The optimizer was **RLOO-style policy gradient,
not standard clipped GRPO**, and its reward was an event-ledger proxy, not
PAUC-F. Metric-reward and GRPO follow-ups remain proposals, not executed runs.

Completion state: `outputs/jevany-2b-rl/profile-7p1-v10/full_run/status.json`.
Launcher log and recorded command are in that directory. The execution notes
and design choices below are retained as history.
Slurm step: **38177.1**. The GPT gateway probe succeeded; the one-video WEB
canary completed all 15 decisions, merging and official scoring successfully
(zero fires, so its score is a plumbing check, not a quality measurement).
Training toward step 16 began at 2026-11-07 16:28 UTC after those checks.
All seven learners subsequently confirmed restored step 1 and exact equality
of all 300 SFT reference tensors; the frozen generator/local judge was ready.
`full_run/startup_verification.json` records the successful eight-worker startup
and current-node-only runtime identity.

Corrected reference loading: PEFT's second-adapter loading path rounds
the SFT reference through BF16 before promoting it to FP32. The trainable SFT
copy loads exactly. The implementation now restores saved FP32 reference
tensors after promotion and verifies all 300 tensors exactly. Sixteen CPU
tests pass. A fresh SFT-start update and automatic restore passed under the
corrected source. Retain the preceding four-update run as the hardware
benchmark; use the corrected checkpoint for a full experiment.

Readiness evidence: `outputs/jevany-2b-rl/profile-7p1-v10/readiness.json`.

## Objective and fixed contract

Continue the **same trainable parameter set as the 2B J1-S SFT run**:
decoder LoRA across all 24 layers plus the JevAny q/k pointer head.
Keep original base weights, vision, and the promoted MoE answer generator frozen.
Do not substitute head-only RL or full-parameter fine-tuning.

- Common starting checkpoint:
  `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
  Its completed WEB result was **54.67 judged PAUC-F**, 701 useful / 290 wasted
  fires, using frozen promoted MoE answers. This is a historical reference;
  rerun SFT under the final evaluation runtime before claiming an RL gain.
- Base: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Preserve existing LoRA: r64, alpha128, dropout configuration 0.05; 62,521,344
  saved LoRA parameters and 1,049,088 pointer parameters. For RL likelihood
  parity, disable stochastic dropout consistently during rollout and actor
  replay while retaining gradients on current LoRA/head.
- Actions: `Silence`, `Hold`, `Response`; Hold maps to the existing Standby tag.
  Preserve J1 history, actual prior answers, last-response v2 memory, direct
  prefix, 1 FPS, 32 causal frames and 384px / 144 visual tokens per frame.
- Generator: `../stream-gemma/outputs/lora-qwen-w1-long-neg`, with its pinned
  MoE base/adapter, existing forced-response prompt, greedy decoding and
  96-token answer cap. Only 2B chooses when to respond.
- Train on spans from the existing **10% training-video cohort**, derived
  from original annotations and the J1-S source manifest. Do not take another
  10% sample or train on validation/WEB. Audit split overlap by video identity
  and path. Hash checkpoints, data, prompts and reward settings before launch.

## Selected allocation: seven controller/learners, one generator

| H200 GPUs | Collection/scoring phase | Gradient-update phase |
|---|---|---|
| 0–6 | Batched 2B decision rollouts | Replicated 2B LoRA + pointer training |
| 7 | Resident frozen MoE; queued, dynamically batched answer requests and local judging | Generator remains resident and idle |

Each learner holds a complete 2B model. Use data parallelism and NCCL gradient
synchronization, not tensor-parallel sharding of 2B. Six learners and two MoE
replicas were the initial layout; native generator kernels and shared answers
allow one generator, while actor learning dominates complete-update cost.
The implementation shares each
generator's resident frozen Qwen base with the local semantic judge: disable
the response adapter during judging, then restore it for answer generation.
Judging finishes before actor backward. Training rewards remain entirely local;
remote reward judging was blocked during setup and is not used. On 2026-11-07,
the user explicitly authorized GPT scoring for full WEB evaluations at steps
16 and 32; these use the existing GPT-4.1 gateway scoring path.
Freeze the chosen reward backend across comparative experiments.

Use activation checkpointing, token/physical-shape bucketing, a physical cap
of 24 and gradient accumulation; current collection/replay is mostly batch
16 or 20. Initial probes started at batch 4. SFT probes measured
about 13 GiB allocated for full-context microbatch 4 and 39 GiB for microbatch
16, but these are not RL memory guarantees or speed estimates.

### Implemented pipeline and measured gates

- Entry points: `scripts/run_jevany_rl_cached.sh` (persistent kernel staging),
  `scripts/run_jevany_rl_profile.sh` (profiling plus automatic
  restore verification), `scripts/run_jevany_rl_gate.sh`, and
  `scripts/run_jevany_rl.py`. Profiling defaults to four updates and rejects
  more than 16. A separate explicit `--full-run` option permits a later full
  launch without changing frozen training code. The full workflow now uses
  that option for its step-16 and step-32 training phases.
- Data: 4,096 frozen spans, comprising 2,048 single-event, 1,024 multiple-event
  and 1,024 no-event spans, from the existing 11,899-video SFT cohort. Exclude
  445 source records matching known evaluation identities. Audit split aliases,
  source hashes and complete four-second event windows before sampling.
- Collection/replay must preserve **both physical dimensions per observation**.
  Save 512-token padding buckets and physical batch size. Collection uses
  physical buckets 4/8/16/20/24 up to the configured cap; replay groups by the
  saved shape. Filler rows receive no action readout or loss. A GPU check of
  54 uncertain observations across all five shapes, including reordered
  companions, produced **zero log-probability error**. This avoids the batch
  rounding drift found when unpadded singleton decisions were replayed at a
  different batch shape. Report real-row and real-token fractions so filler
  compute cannot masquerade as useful utilization.
- Distribution guards remain fixed: maximum probability error `0.005`, maximum
  `KL(rollout || replay)` `1e-4`, mean KL per rank `1e-5`, and selected-action
  log-probability error `0.05` when its old probability is at least `0.01`.
  Record all-action log-probability error separately because near-zero actions
  make its absolute error ill-conditioned. A complete 4,096-decision update
  passed with **zero probability/KL error**, nonzero LoRA/head gradients, and
  no gradients on frozen parameters.
- Cache frozen vision features, preserving decoder gradients. When features
  are available, omit redundant raw-patch concatenation/upload; require the
  complete causal feature cache before forward. The optimized path preserved
  token IDs, masks, frame grids, logits and **all 304 gradient tensors exactly**
  in the GPU test. For eight full-context observations, warm CPU preparation
  fell from 0.83–0.97s to 0.24s and raw-patch payload from 864 MiB to 0.047 MiB.
- Keep the trainable controller on Torch 2.5.1 / Transformers 5.5.0. Frozen
  generator/judge sidecars use the existing read-only Torch 2.10 / Transformers
  5.5.3 environment for native grouped GEMM. Refuse Python expert fallback.
  The local queue bridge carries plain Python prompts/answers, not Torch
  tensors. Native GPU projections matched the fallback exactly in both test
  shapes; decoded video pixels also matched across environments. A warm
  full-context batch of eight answers took 4.15s in the resident diagnostic.
  These component timings do not substitute for complete-update throughput.
- Share generator answers and in-flight work in a root-side broker across
  replicas. Keys contain the video, frame, question, memory and actual causal
  turns; weights and preprocessing remain fixed. Duplicate callers receive the
  same greedy answer with their own request IDs. A real-generator smoke test
  served five logical requests with two computations, checked cross-rank
  delivery, a later cache hit, and judge passthrough. Sixteen CPU tests cover
  scheduling, cache semantics, exact reference loading, reward and loss contracts.
- Limit ready observations to within two steps of the slowest active trajectory
  in each learner, and wait for 75% of the active batch when answers are pending.
  This prevents quiet streams running far ahead and fragmenting both collection
  and replay into tiny batches. Check for terminal completion after draining
  replies, including instant cached replies, before waiting again.
- Preserve the SFT adapter's FP32 arithmetic. A BF16-compute screen retained
  FP32 master weights but moved action probabilities by up to 0.0251 and reached
  KL 0.00148, outside the fixed bounds; it was rejected without optimizer steps.
- Judge: frozen pinned Qwen MoE base with response adapter disabled, original
  semantic/repetition rubrics and constrained JSON scores
  `0, 0.25, 0.5, 0.75, 1`. The current backend is
  `jevany-offline-judge-v8-native-grouped-mm`. It passed ten calibration cases
  and all 263 unique judgments from the real rollout batch. Threshold agreement
  with the previous runtime was 257/263; 125/128 trajectory rewards matched.
  This measures format and cross-runtime consistency, not independent reward
  accuracy. Version and freeze the backend across RL comparisons. Failed calls
  never become zero rewards.
- Mask recurrent inputs for left-padded singleton MoE batches too; installed
  Transformers otherwise skips that mask. Audit batched/singleton generator
  wording separately from actor likelihood parity: BF16 MoE greedy ties can
  produce paraphrases. Verify input tokens/masks and always retain the actual
  generated answer for future observations and actor replay. Resume restores
  model/optimizer/RNG/sampler state, without promising bitwise async text.
- Memory capacity test: 32 causal frames, 31 prior 96-token answers, resident
  reference LoRA and allocated AdamW state. A 262,144-token budget capped
  physical batches at 28 for 9,216 tokens, peaking at **90.02 GiB allocated /
  95.41 GiB reserved**. This was a no-step capacity test; current cap 24 is
  smaller. The successful real update peaked at 53.78/57.96 GiB per learner.
- Run gradient-sized NCCL checks on all eight devices and within the exact
  learner group/device visibility before loading models. Record reduction,
  norm and optimizer timings separately. Limit compiler pools to four workers
  per process, prefetch CPU inputs, and use independent GPU visibility.

### Current bounded throughput profiles

The final candidate uses 7 learners + 1 generator, 32 starts × 4 rollouts ×
32 decisions = **4,096 decisions/update**, physical cap 24, bounded progress,
shared generator memoization and 2,048 cached frames per learner. Run four
updates (one warmup plus three measured), then verify checkpoint restoration
with zero additional optimizer steps. This hardware gate completed successfully
in `outputs/jevany-2b-rl/profile-7p1-v9/` (Slurm 38764, exit 0).

| Three warm complete updates | Measured result |
|---|---:|
| Update time | 677.00s mean; 661.59–692.47s range |
| Useful sampled decisions | 4,096/update; 6.05/s |
| Allocated eight-GPU cost | 0.3673 GPU-hours / 1,000 decisions |
| Collection / judge / learner / save | 213.01 / 19.14 / 443.01 / 1.18s mean |
| Collection / replay real-row fraction | 89.69% / 96.39% |
| Collection / replay real-token fraction | 75.11% / 80.72% |
| Generator requests / actual computations | 609 / 378; 37.93% reused |
| Mean / maximum generator request latency | 6.63 / 23.10s |
| Peak learner allocated / reserved memory | 44.68 / 48.04 GiB |
| Peak learner / generator device memory | 54.75 / 81.74 GiB |
| Learner utilization during learning / whole cycle | 92.17% / 74.09% |
| Whole-node utilization across complete cycles | 66.39% |
| Maximum rollout/replay probability or log-probability error | 0 |
| Minimum local LoRA / pointer gradient norm | 0.4995 / 0.01645 |

`profile_summary.json`, phase-filtered `gpu_samples.csv`, per-batch reports,
hash-verified `source_snapshot/`, and `resume_verification.json` retain the
evidence. All seven learners restored step 4 / sampler cursor 128 without an
additional optimizer step. The generator recorded 19,200 native grouped-MM
calls and refused expert fallback.

This hardware benchmark predates the exact-reference loading correction;
`benchmark_scope.json` explicitly excludes its checkpoint from the full
experiment. The correction changes startup reference values only, with no
change to per-update operations, batch shapes or kernels. The final corrected
gate started fresh from SFT in `profile-7p1-v10/` and **passed** (Slurm 38765,
exit 0): one complete 4,096-decision update in 718.43s, exact verification of
all 300 reference tensors per rank, then automatic restore on all seven ranks.
Initial KL to SFT and rollout/replay errors were exactly zero. Minimum local
LoRA/head gradient norms were 0.7215 / 0.02214; maximum allocated/reserved
learner memory was 43.55 / 46.96 GiB. All 29 frozen source/data/model hashes
match. `resume_verification.json` confirms step 1, sampler cursor 32, and zero
additional optimizer steps. This single corrected update is a correctness
gate, not another three-warm-update benchmark.

The corrected continuation checkpoint is
`outputs/jevany-2b-rl/profile-7p1-v10/step-000001`. Source snapshots, the cold
gate report, exact-reference checks, and `readiness.json` retain its evidence.
Only test another GPU split if measured queues and phase times justify it;
select using complete-update cost.

The larger logical batch changed the optimization schedule: the matched
131,072-decision budget used 32 updates instead of the original 128.
The 32-update E1 continuation completed from the corrected profiling
checkpoint under the same frozen code with explicit `--full-run --resume`.
The prelaunch warm-rate projection was about 6.02 node-hours / 48.14 allocated
GPU-hours for 32 updates, plus initialization and quality evaluations. Actual
continuation timing is recorded in the completed experiment report.

The completed workflow targeted
32 total updates, including the corrected step-1 gate, for 131,072 decisions;
it did not add another 32 updates. Keep the same output and every recipe
argument so the configuration/source hash check can enforce continuity.

```bash
srun --jobid=38177 --overlap --nodelist=ip-10-3-120-167 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=none \
  ../stream-gemma/.venv/bin/python -u scripts/run_jevany_rl_full_current_node.py
```

This is the historical executed command; the workflow has finished. Do not
rerun it merely to inspect results.
`--cpu-bind=none` preserves all 96 available CPU cores despite the existing
allocation's one-CPU request; this affinity was checked before launch.
The orchestrator uses a lock, freezes/snapshots its workflow sources, checks
the 29 frozen training artifacts, and records each subprocess log. Its three
CPU tests cover evaluation ordering, restart behavior, and checkpoint metadata
views. The views symlink immutable RL weights and supply J1 evaluator metadata
without changing optimizer checkpoints or the validated training source.

Both scheduled WEB checks used the original SFT evaluation environment, eight
local ranks, all 500 videos, the frozen promoted answer model, greedy zero-margin
controller, 32-frame causal history, 96-token response cap, and GPT-4.1
`pauc-f-v1` / omega 0.5 scoring. All prediction IDs, judge responses and scorer
settings passed verification before proceeding. Training paused for evaluation and
scoring at step 16, then resumed automatically. This user-requested schedule
supersedes the earlier proposed development checks at 0/4/8/16/32. No new
baseline evaluation or evaluations at steps 4/8 were scheduled.

Diagnostics before the last scheduling/cache changes:

- The initial cold correctness update took 1,284.85s with six learners and
  Python expert fallback. CPU preparation took 196–211s per learner.
- The native-generator, lightweight-input seven-learner implementation passed
  two complete updates with zero likelihood error. Its first warm update took
  **676.97s / 6.05 decisions/s**, with 44.57 GiB allocated / 48.71 GiB reserved
  per learner. Collection was 191.75s, judging 27.31s, and learning 455.99s.
- That warm batch had 211 response requests but only 121 distinct generator
  contexts; mean request latency was 34.4s and only 68.2% of physical collection
  rows were real observations. Those measurements motivated the shared broker
  and bounded progress above. They are a baseline, not the final efficiency result.
- Slurm clears the local Triton cache between allocations. Save content-addressed
  kernels to `outputs/jevany-2b-rl/kernel-cache/`, stage them locally before a
  new launch, and save newly compiled kernels on exit. Keep cold initialization
  and warm complete-update throughput separate.

## One optimizer update

1. Freeze the current policy version. Sample **32 starting spans × 4 rollouts**,
   for 32 seconds at 1 FPS: 128 trajectories / 4,096 nominal decisions.
   Use a shared causal reference prefix within each group, independent sampling
   seeds and temperature 1.0. Use self-generated actions/history after branching.
2. GPUs 0–6 sample categorical actions from the pointer logits. On Response,
   request an answer from GPU 7 and put that actual answer into future
   history. Save exact observations, frame IDs, memory, action IDs, probabilities
   and policy version. Do not force a quota of Response actions.
3. Score actual answers for semantic correctness/duplication; compute timing
   and missed-event penalties from annotations. Batch/cache judging. A failed
   judge call is not a zero score. Finish the complete batch before updating.
4. Compute detached group-relative advantages. Reconstruct every saved
   observation with the current actor, recompute action log-probabilities and
   backpropagate through **LoRA and pointer**. Detached hidden-feature replay
   is insufficient when LoRA is trainable. Cache frozen vision features only.
5. Accumulate all microbatch gradients, synchronize across the seven learners,
   clip and take **one AdamW step**. The 128 trajectories do not divide evenly
   across seven ranks: use global-loss weighting and synchronize only after
   each rank finishes its local microbatches; do not average unequal rank means.
6. Begin fresh rollouts with the updated policy. Save optimizer/RNG/sampler
   state and completed-batch IDs for resume. Model and optimizer state restore
   exactly; asynchronous BF16 generator batching can change greedy wording,
   so future answer text is not guaranteed bitwise identical after restart.
   Do not train against stale
   rollout weights or perform multiple optimizer steps on the same batch.

Keep a frozen SFT LoRA/head reference for three-way KL. Its probabilities need
a separate no-grad decoder forward on the exact sampled observations. The
reference adapter may share immutable base weights with the current adapter;
never switch adapters between a differentiable forward and its backward.

## Reward and loss used in E1

Use an additive event ledger, with chronological one-to-one matching to the
existing MarkII training windows `[onset, onset + 4s)`. Include only complete
windows and settle event outcomes at fixed window-close times across rollouts.

```text
D = max(1, number of annotated event windows)
R = [sum(matched correctness × max(0, 1 - delay_seconds/4))
     - 0.50 × missed events
     - 0.25 × other unmatched fires
     - 0.35 × duplicate fires] / D
```

Correctness is a frozen semantic score in [0,1], with 0.5 required for a match.
Duplicate and other unmatched penalties are mutually exclusive. Distinct
recurring events remain eligible even if wording repeats. No-event/no-fire
spans earn zero; silence has no positive bonus. Use a calibrated local judge
or GPT scoring of actual generated answers, not timing-only or lexical scores.

Minimize negative advantage-weighted categorical log-probability, plus
`0.02 × KL(current || SFT)` minus `0.001 × entropy`, averaging over groups
and normalizing each trajectory's action sum by a fixed 32. No gradient flows
through rewards, generator or sampled history. Use unnormalized leave-one-out
advantages, without group-standard-deviation normalization. This is an
on-policy group-relative policy-gradient adaptation, not unchanged stock GRPO.

E1 optimizer: AdamW, pointer LR `1e-5`, LoRA LR `1e-6`, weight decay 0.01,
gradient clipping 1.0. The parameter set matches SFT; RL learning rates need
not equal the SFT rates. Freeze settings after the stability gate.

## First 3–5 runs

This is the original experiment proposal. E1 completed; E2–E5 below have not
run. The completed report's reward-alignment and GRPO comparison proposals
record the current follow-up discussion separately.

All runs train LoRA + pointer and restart from the same SFT checkpoint.
Hold data schedule, generator, reward and sampling budget fixed for E1–E3.

| Run | Comparison |
|---|---|
| E1 | Whole-trajectory leave-one-out credit: own reward minus other rollouts' mean |
| E2 | E1 with per-step remaining-return credit, excluding already-settled outcomes |
| E3 | E2 plus `0.10 ×` supervised action CE on fixed training-only SFT replay |
| E4, optional | Winning recipe with 8 rollouts/group at matched total sampled decisions; explicitly adjust update count |
| E5, optional | Repeat the selected recipe with a second RL sampling seed |

Core pilot budget: **32 updates/run**, including the corrected step-1 gate;
131,072 nominal decisions/run. Save every update. The completed E1 evaluated
full WEB with GPT scoring at 16/32, as requested on 2026-11-07; these quality
evaluations are separate from the efficiency gate. E4 can use
32 starts × 8 rollouts × 32 steps for 16 updates to match sampled decisions;
it changes the update schedule too. The original eight-start recipe used
128 updates for the same decision budget.

## Profile before choosing the final GPU split

Measure at least **three warm complete updates**, including reference scoring,
actor backward, answer generation and judging. Keep the global batch unchanged
while comparing allocations; report phase times, queue waits, throughput,
allocated/reserved memory, and `8 × wall-hours` allocated GPU-hours.

- If answer queues dominate, test **5 controller/learners + 3 generators**.
- If generators are mostly idle, test **7 controller/learners + 1 generator**.
- If local judging dominates, test **5 controller/learners + 2 generators +
  1 dedicated judge**, using the same judge and reward contract.
- Only after correctness/memory profiling, consider resident 2B learner copies
  on generator GPUs so all eight join the update phase. Do not make this
  co-location complexity a prerequisite for the first pilot.

Select by complete-update wall time, not peak utilization. Do not claim an
optimal split or project training duration from SFT throughput alone.

## Gates, evaluation and implementation references

Before scaling, verify exact observation/log-prob parity, LoRA/head gradients,
frozen base/vision/generator/reference, reward edge cases, resume and memory.
Monitor useful coverage, wasted fires and Hold/Silence behavior. Proposed pause
gate: coverage below 90% of the matched baseline at two consecutive checks, or
immediate near-silence collapse. Freeze gates on development data before E1.

Select on development; evaluate the frozen candidate and matched 2B SFT
baseline on **all 500 WEB videos**, GPT-4.1 judged `pauc-f-v1`, omega 0.5,
greedy controller and response margin 0. Report PAUC-F, precision/recall,
useful/wasted fires, latency, memory, GPU-hours and paired bootstrap intervals.
WEB has been inspected before and must not be described as untouched.

Reuse x3's dense controller loader, `scripts/train_jevany_2b_batched.py`,
`src/pace_clm/jevany_streaming.py` and `scripts/eval_jevany_streaming.py`.
Adapt existing Stream-Gemma GRPO span/environment/judge abstractions; its token
learner is not a drop-in categorical pointer learner. Save new artifacts under
`outputs/jevany-2b-rl/` without overwriting SFT runs.

Motivation: Self-Focus for shared-prefix/future-return credit; MMDuet2 for short
multi-turn rollouts and correctness/timing/redundancy feedback. No new sensing
actions, retrieval, response-generator training or resolution changes here.
