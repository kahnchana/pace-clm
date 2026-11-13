# Experiment log

## 2026-11-06 — Qwen3.5-2B mixture runs and all WEB evaluations complete

- Completed existing-order control A and representative-mixture B from
  byte-identical fresh seed-42 LoRA/head states on the same audited eligible
  pool. Fixed global batch was 128 in both arms; ordering preserved original
  row contents and multiplicities. All eight full 500-video WEB evaluations
  passed prediction, judging and protocol integrity checks.
- The user replaced enforced training-time stopping with preflight-estimated,
  frozen step budgets. A completed 3,608 updates / 461,824 presentations in
  13.533 training hours; B completed 1,776 / 227,328 in 8.914 hours. Different
  backends, physical chunks, actual durations and exposures prevent a clean
  equal-time or equal-dose A/B conclusion.
- A's scheduled PAUC-F scores were 45.44 / 49.47 / 43.20 / 48.70; B's were
  40.49 / 47.74 / 51.73 / 47.61. Intermediate maxima are diagnostic, not
  substituted final endpoints. Video-bootstrap intervals confirm substantial
  within-run drops but do not measure seed or judge variability.
- Documented control GPU underutilization, conservative physical chunks,
  checkpoint restoration versus numerical reproducibility, startup recoveries,
  exposure skew and the distinction between per-frame CE and closed-loop
  judged quality. Recommend bounded diagnosis before scaling unchanged;
  no continuation, new objective or new training was launched.
- The requested completion notification was delivered into the original chat.
  See the [completed report and artifact map](qwen35_2b_mixture_12h_results.md),
  [A record](qwen35_2b_control_12h.md) and
  [B record](qwen35_2b_representative_12h.md).

## 2026-11-04 — smaller Qwen3.5 J1-S controllers

- Completed the official 2B controller's matched one-epoch, seed-42 run:
  263,760 training presentations, 8,243 updates, then all 500 WEB videos with
  the frozen promoted 35B MoE answer generator. GPT-4.1 judged PAUC-F was
  54.67 versus 56.51 for historical J1-S, with 991 versus 1,357 fires.
  See [the full result](qwen35_2b_j1s_web.md).
- Passed the 0.8B and 4B preflight gates and started the size sweep. At
  21:19 UTC, 0.8B training was still running and 4B full training had not
  started; neither had WEB results. See the [size-sweep record](qwen35_dense_size_sweep_j1s_web.md).

## 2026-09-27 — repository foundation

- Created standalone Apache-2.0 repository.
- Pinned and validated promoted PACE-Qwen contract.
- Added versioned MVP/action configuration.
- Added causal input, label, pooling, freezing, LM-head bypass, and projection
  gradient tests.
- No GPU experiment was launched; real-model smoke extraction and tiny-set
  overfitting remain later execution-order milestones.

## 2026-09-27 — PACE-200 CLM pilot

- Verified the pinned promoted Qwen/PACE model on GPU: hidden width 2048,
  final valid language-token pooling, no LM-head call, frozen backbone and
  adapter, and exact cached/uncached action-feature parity.
- Froze a balanced 768-example PACE-train manifest (256 examples per state,
  353 videos) and the existing 200-video validation manifest (10,658 decision
  points); train and validation video IDs are disjoint.
- Extracted reusable train/validation features and passed the balanced
  24-example overfit gate at 100%.
- Trained the required raw-cosine, linear, MLP, and separate-head CLM methods,
  plus projection-width, shared-head, and short-label variants. Trained methods
  use seeds 42, 43, and 44.
- Evaluated the existing PACE token-generation decision on the same inputs.
  It leads with 83.70 macro-F1 and 80.10 timing PAUC-F. The best CLM timing
  result is 65.14 PAUC-F for 256-d; no CLM variant beats the token baseline.
- Saved checkpoints, exact manifests, reusable features, per-decision logits
  and predictions, confusion matrices, unchanged-scorer outputs, and run
  provenance under `outputs/`.
- Verified the response-generation handoff on eight predicted CLM Response
  events from eight deterministic short videos. All eight produced non-empty,
  query-relevant text through the original PACE generation path; this is a
  qualitative smoke sample, not a judged generation metric.
- Semantic judged PAUC-F was intentionally not run; this pilot is timing-only.
  WEB/EGO remains untouched.

## 2026-10-02 — fresh gate and scale-up follow-through

- Froze two new disjoint 200-video Markii cohorts, with 11,563 development and
  12,658 final decisions, excluding the prior PACE-200 and mixture cohorts.
- Replayed the 3,260-update run to exact internal-dev-NLL minima at steps
  200–900, and trained a doubled-Hold-loss variant on the run 5 mixture.
- Selected a +0.5 Response margin on the balanced MLP using only the new
  development cohort and firing/Hold constraints. It reduced final fires but
  reduced timing PAUC-F from 81.92 to 81.42.
- On the new final cohort, the frozen run 5 MLP scored 83.38 mean timing PAUC-F
  across three seeds versus 82.08 for PACE token generation, with fewer fires
  and repeats. Its paired video bootstrap PAUC-F difference was +1.31 points,
  95% interval [+0.17, +2.50].
- Early replay, doubled Hold loss, and full PACE-exposure training all scored
  below the frozen 24.3k run 5 MLP on final timing PAUC-F. No CLM head showed
  a repeatable advantage over its MLP comparator.
- Saved exact manifests, features, checkpoints, per-video and token predictions,
  scorer outputs, bootstrap, and completion marker under
  `outputs/pace-clm-gate/`. Judged response quality and WEB/EGO remain unrun.

## 2026-10-10 — full WEB/EGO judged transfer

- Evaluated the frozen run 5 MLP seed-42 decision head with the promoted Qwen
  backbone and original response generator on all 500 WEB and 326 EGO videos.
- Completed 839/839 WEB and 716/716 EGO GPT-4.1 judge requests after explicit
  authorization; no reported request errors or missing scored videos.
- With `pauc-f-v1` and `omega=0.5`, judged PAUC-F was 53.50 WEB and 43.95 EGO,
  below promoted PACE Qwen token generation at 57.47 and 46.05. The MLP fired
  925 versus 1,132 times on WEB and 1,310 versus 1,634 times on EGO, but lost
  judged recall on both splits.
- Saved full predictions, judge outputs, scored records, and summaries under
  `outputs/pace-clm-web-ego-full/`. Promoted PACE WEB/EGO raw outputs are absent
  locally, preventing paired video bootstrap and baseline rescoring.

## 2026-10-22 — JevAny streaming pilot and 10% launch

- Pinned JevAny at `b273aea682e5b748a2605808e52ccafb7a378924`, kept its
  Apache-2.0 license, and verified the promoted PACE adapter SHA-256
  `eb11e600de59edf48ef2d71bb632ae5944c0474441f264bb3ce6ceae8bb71da9`.
- Rebuilt the promoted episode exposure: 26,080 episodes and 263,760 supervised
  positions (215,502 Silence, 7,038 Hold, 41,220 Response). The manifest SHA-256
  is `5451fab4bc38f49719044d5c348d9541c7539a4a49d258760889225c63db8176`.
- Passed the three-state real-model smoke. All 500 intended LoRA tensors received
  gradients. A 32-frame case admitted all 5,859 expanded tokens and passed
  forward/backward with the isolated FLA 0.3.2 kernel at 122.0 GiB peak
  allocated GPU memory. The fallback Qwen kernel OOMed on the same case.
- Ran J0 and J1 tiny pilots on the same nine examples, three passes, seed 42.
  J0 mean loss fell from 1.87 on pass one to 0.27 on pass three; J1 fell from
  1.90 to 0.33. These are overfit plumbing checks, not quality comparisons.
  J1 checkpoint reload passed and the original generator adapter hash stayed
  unchanged.
- Launched the single-seed 10% J1 run on the eight H200s already allocated by
  Slurm job 36781. The training log is
  `outputs/jevany/logs/full-current-node.log`; resumable checkpoints go under
  `outputs/jevany/full-j1-seed42/`. Training and held-out validation results
  are pending.

## 2026-10-23 — JevAny run recovery and evaluation wiring

- The first full attempt reached optimizer step 50 with recent loss 0.123, then
  was restarted because its forward-only vision cache caused the frozen visual
  tower to run again during activation-checkpoint backward replay. No checkpoint
  existed, so these updates are excluded from the replacement run.
- The replacement run began at 00:35:57 UTC in allocation 36781, seed 42, with
  the cache fix. It saves first at step 25, then every 250 steps. The compiled
  Triton cache is on the node's large `/dev/shm` filesystem.
- `scripts/run_jevany_posttrain.sh` monitors the run, resumes up to three
  interrupted attempts from checkpoints, and then launches judged selection
  and final validation plus complete WEB/EGO rollouts and existing scorers.
  Live logs and current status are under `outputs/jevany/`; numerical results
  remain pending.
- Step 25 completed at 01:28 UTC with recent training loss 0.2524. The first
  complete 888 MB recovery checkpoint includes all eight RNG states, optimizer,
  pointer head, and controller adapter. Held-out results remain pending.

## 2026-10-24 — JevAny full training complete

- J1 seed 42 completed one pass over all 263,760 supervised presentations on
  eight H200s: 8,243 optimizer updates and final checkpoint step 8,243. Total
  training time after load was 91,742 seconds. Generated-response validation
  and WEB/EGO judging are next; training loss alone is not a quality result.

## 2026-10-24 — Full WEB/EGO evaluation launched

- A final-checkpoint one-video WEB probe passed on GPU 7: 15 controller
  decisions, two valid response events, no malformed emissions, and 253 seconds
  of rollout time. This exercised both the trained controller and the original
  generator adapter.
- The complete 500-video WEB rollout began on all eight GPUs in tmux session
  `jevany-posttrain` at 02:23 UTC. The 326-video EGO rollout follows on the
  same eight GPUs. Full WEB predictions are judged with GPT-4.1 as soon as
  merging finishes, concurrently with EGO inference; EGO judging follows its
  complete rollout. Selection/final validation follow the benchmark scores.
- The evaluation log is `outputs/jevany/logs/posttrain.log`. A separate
  `jevany-watch` session can restart resumable evaluation if its worker exits
  before `outputs/jevany/reports/full-results.json` is written. Numerical
  benchmark results remain pending.

## 2026-11-07 — First 2B RL pilot and both judged WEB checks complete

- Continued strong 2B J1-S SFT step 8,243 with decoder LoRA plus pointer-head
  RL; base, vision, SFT reference and promoted answer generator stayed frozen.
  The actual optimizer was whole-trajectory **RLOO-style policy gradient**,
  not standard clipped GRPO. Training used a local semantic/event-ledger proxy,
  not PAUC-F or GPT rewards.
- Completed 32 total optimizer updates: 1,024 distinct 32-decision spans from
  911 videos, four rollouts/start, 4,096 trajectories and 131,072 decisions.
  This consumed 25% of the prepared span pool. The corrected step-1 gate was
  included in the 32-update budget.
- The continuation and both evaluations used only eight H200s on
  `ip-10-3-120-167`, allocation 38177 / step 38177.1. Steps 2–32 took 5.44h of
  complete-update time; the November 7 workflow, including preflight and two
  WEB evaluations, finished at 23:12:05 UTC after about 6h52m.
- Full 500-video WEB, GPT-4.1 `pauc-f-v1`, omega 0.5: step 16 scored **53.31**
  PAUC-F and step 32 **53.09**, versus SFT **54.67**. Final useful/wasted fires
  were **544/163**, versus **701/290** for SFT. Precision improved but useful
  coverage fell; neither RL checkpoint improved the headline metric.
- All updates recorded zero rollout/replay log-probability error; final
  seven-learner restore passed at step 32 / sampler cursor 1,024. Analysis
  recorded November 8 separates reward alignment, exploration, optimizer
  choice and additional training exposure as follow-up questions, not proven
  explanations or completed experiments.
- See the [full RL E1 report](jevany_2b_rl_e1_results.md) for objective formulas,
  diagnostics, runtime scope, limitations, proposed comparisons and artifact
  links. No follow-up run is launched by documenting these results.

### 2026-11-09: E2 comparison and E3 supervised-anchor RLOO launch

- Reward-aligned RLOO and fixed GRPO v2 completed 16 collection rounds each,
  stopping for useful-response coverage below 90% of SFT at consecutive checks.
  Full 500-video GPT-4.1 WEB at round 16: RLOO 54.5561 PAUC-F, 605 useful /
  187 wasted; GRPO 53.7149, 577 / 191; retained SFT 54.6651, 701 / 290.
  Neither improved full-WEB PAUC-F. GRPO v1's failed input-replay gate remains
  preserved separately; v2 passed exact initial ratios and restore checks.
- The training-only coverage audit found reward inversions in about 4% of
  pairs differing in useful-response count, useful actions inheriting negative
  trajectory advantages, and about 20–22% of events missed by all four rollouts.
  Score-function diagnostics do not establish parameter-gradient causality.
- User authorized the [E3 supervised-anchor RLOO pilot](jevany_2b_rl_e3_anchor.md),
  launched at 19:56:55 UTC on the original eight H200s, node
  `ip-10-3-120-167`, allocation 38177. Fresh SFT, unchanged RLOO/reward settings,
  plus 126 class-balanced original training examples per round and a coefficient
  calibrated once to a 15% initial gradient-norm ratio. All 41 focused CPU tests
  passed. The first update and seven-learner restore gate passed at 20:17:57 UTC;
  training resumed toward round 8. These checks do not imply quality improved.
- Full 500-video WEB/GPT evaluation is scheduled at rounds 8/16/24/32 with
  coverage stops. The user explicitly prohibits runtime behavioral heuristics:
  evaluation remains raw learned argmax with zero response margin. Active
  artifacts: `outputs/jevany-2b-rl/rloo-sft-anchor-v1/`.

### 2026-11-10: E3 completed; episode-level follow-up planned

- E3 completed all 32 rounds / 32 optimizer steps at 03:43:29 UTC. All four
  500-video WEB evaluations and GPT judgments finished. PAUC-F at rounds
  8/16/24/32: 52.9117 / 54.2602 / 54.8260 / 54.3782, versus SFT 54.6651.
- The best observed checkpoint, round 24, retained all 701 useful responses
  with 280 wasted versus SFT's 290. Its +0.1608 PAUC-F point gain has paired
  95% interval [-0.6665, +0.9796]; no conclusive gain is established. Round 32
  regressed to 628 useful / 212 wasted. Training finished its preset budget;
  no automatic extension launched. See the [completed E3 record](jevany_2b_rl_e3_anchor.md).
- The [episode-level PAUC-F plan](../plans/jevany_2b_episode_level_rloo.md)
  proposes complete causal video-request episodes with continuous generated
  history, unchanged raw-argmax inference, and a bounded decision-matched pilot
  on the original eight GPUs. Implementation and training have not launched.

### 2026-11-12: episode-balanced mixed WEB/EGO pilot launched

- User authorized the [episode-balanced mixed pilot](../plans/jevany_2b_episode_balanced_mixed_rloo.md).
  Fresh J1-S SFT step 8243, unchanged reward/optimizer/replay/prompt contracts,
  with exactly four original, two WEB and two EGO episodes per update and four
  complete trajectories per episode. Frozen seed-42 schedule: 23 updates,
  98,836 decisions, 92 original / 46 WEB / 46 EGO distinct episodes.
- All 58 focused CPU checks passed. Both assigned eight-H200 allocations were
  running and idle before launch. Training workflow started at 15:05 UTC on
  `ip-10-3-120-167`, seven learners and one generator; asynchronous eight-worker
  evaluation started on `ip-10-3-70-177`. Initial steps were stopped before any
  training update to correct one-CPU task affinity. At 15:08 UTC both restarted
  with `--cpu-bind=none` and 96 available CPUs, in steps 38177.37 / 38226.25.
  No new allocation was requested. Run artifacts are under
  `outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2/`.
- First-update restore, replay and independent evaluation gates must pass before
  automatic continuation. Full evaluations are queued for steps 6/12/18/23;
  all inference uses raw learned argmax and zero response margin. Reports include
  combined WEB/EGO, heldout and sampled/unsampled splits, and EGO temporal
  diagnostics. Previous mixed-v1 step 16 is also queued for its missing evaluation.
- Previous mixed-v1 failed waiting for learners on a 10,284-decision final batch;
  no saved stacks establish its root cause. New batches are bounded at 5,248
  decisions, with rank progress and timeout stack capture. Failure handling now
  queues the latest committed checkpoint for full evaluation. Launch does not
  establish correctness-gate success or improved quality; results remain pending.

### 2026-11-12: first-gate runtime mismatch identified and corrected

- Episode-balanced v2 saved step 1; training-node replay and seven-learner
  restore passed. Independent evaluation-node replay stopped the workflow at
  15:41 UTC. Model weights, decoded pixels and token inputs matched, while
  frozen vision embeddings differed. The evaluation process inherited host
  CUDA libraries and loaded cuDNN 9.7.1 versus training's 9.1.
- Removing the inherited `LD_LIBRARY_PATH` override made the original probe
  reproduce all seven saved batches exactly, without changed tolerances or
  modified action scores. A diagnostic using training vision embeddings also
  recovered exact decoder scores, locating the difference in vision encoding.
- At 16:50 UTC, both existing node workflows restarted with an explicit
  PyTorch/CUDA/cuDNN startup assertion and clean library path. Source freezes,
  schedule, learned checkpoint, optimizer and RNG state remain unchanged.
  Affected v2 first-gate/baseline outputs are preserved in the run's runtime
  audit directory and regenerated; continuation from step 1 remains gated on
  those records. No model-quality improvement is claimed from this repair.

### 2026-11-13: user overrides coverage stop and extends through step 32

- Episode-balanced v2 stopped at step 15 / 63,216 decisions after EGO-heldout
  useful responses fell below 90% of SFT at full steps 6 and 12. The user then
  explicitly authorized continuation through step 32 regardless of coverage.
- Prepared `outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/`, retaining
  the original stop record and frozen artifacts. The extension preserves the
  entire original schedule prefix and adds updates 24–32: 32 total updates,
  138,460 decisions, 128 original / 64 WEB / 64 EGO episodes. It resumes the
  exact step-15 model, optimizer, RNG and sampler state; 17 updates remain.
- Coverage/silence quality results now produce warnings rather than stop
  requests. Training objective, optimizer recipe and raw-argmax inference are
  unchanged. Run-local code snapshots preserve the original run's source hashes.
  Two targeted tests, code compilation and exact prepared/generated resume
  contract equivalence passed. At about 03:34 UTC the training workflow began
  seven-learner restore verification on allocation 38177; the evaluation
  supervisor waits for the existing reference job on allocation 38226 before
  taking the same eight evaluation GPUs. No new allocation was requested.
- Full continuation evaluations are scheduled at steps 16/24/32 and subsets at
  20/28. Retain evaluation backpressure and complete final judging; no new quality
  result is implied by launching the continuation.

### 2026-11-13: continuation reaches step 30; replay-bank capacity repaired

- The continuation committed step 30 / 129,368 decisions, with zero recorded
  rollout/replay probability error on completed updates. Update 31 failed at
  12:31 UTC because its supervised replay slice exceeded the original 4,032-row
  bank; the 32-update extension requires 4,259 credits. No step-31 optimizer
  checkpoint was committed. This was a replay-capacity omission, not the
  overridden coverage rule.
- The repaired training-only replay manifest retains the original prefix and
  appends the first 227 frozen examples in order. Replay coefficient, episode
  schedule, learned step-30 checkpoint and all selections through step 30 remain
  unchanged. Prior freezes/source and the failed uncommitted batch are archived
  in the continuation's replay-bank repair audit directory. Three targeted tests
  and exact trainer-contract reconstruction passed.
- At about 14:24 UTC the same node workflows restarted for exact step-30 restore
  and the final two updates. Evaluation remains on allocation 38226 and awaits
  final step 32. Completed full step-30 PAUC-F is 56.560 WEB / 42.826 EGO,
  including training identities; heldout WEB/EGO are 55.561 / 42.063. No final
  step-32 result is available at this recovery launch.
