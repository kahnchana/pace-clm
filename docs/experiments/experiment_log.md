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

<!-- jevany-episode-balanced-v2-to32-complete -->
### 2026-11-13 — Episode-balanced mixed RLOO step 32 complete

- Completed 32 updates / 138460 RL decisions after the user-authorized coverage-stop override and replay-bank repair.
- Final full WEB/EGO PAUC-F: 56.6040 / 42.3052; heldout WEB/EGO: 55.8751 / 41.4761. Combined scores include training identities.
- All queued evaluations and final judging completed; checkpoint 32 hashes verified. Training remained on allocation 38177 and evaluation on 38226, with raw learned argmax / margin zero.
- See [the final experiment record](jevany_2b_episode_balanced_mixed_results.md) for the full recipe, every full result, paired comparisons, trajectory diagnostics, operational failures/fixes and artifact provenance.

### 2026-11-13: step-32 to step-128 ESTP-IT continuation launched

- User authorized completion through step 128 irrespective of evaluation scores.
  The [continuation plan](../plans/jevany_2b_estp_episode_balanced_to128.md)
  preserves the completed step-32 checkpoint and schedule prefix. The new run is
  `outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128/`.
- Frozen updates 33–128 add 384 original, 192 WEB, 81 remaining EGO and 111
  ESTP-IT episodes (768 distinct episodes / 3,072 trajectories). Each batch has
  four original, two WEB and two EGO+IT family episodes, with objective weights
  0.5/0.25/0.25 and unchanged constant-32 normalization. Exposure is 381,336
  additional decisions / 519,796 cumulative; the complete 15,989-credit replay
  bank was prepared before launch. All 1,024 scheduled episodes are unique.
- IT source-video splits are 932/117/117 train/dev/test. Training uses one
  eligible SQ episode from each of 111 distinct source videos, preserving native
  answer windows and fixed 1-FPS sampling. Malformed, dynamic-query and
  unreachable-window training records are excluded with audited counts; no
  windows are widened. Fixed dev/test probes contain 128 episodes each. Exact
  media hashes found no duplicate new training videos or collisions with the
  protected evaluation media. This is primarily explicit SQ training, not a
  contextual ESTP curriculum.
- Preserve all 1,212 ESTP-Bench cases and use the unchanged official ESTP-F1
  reference for endpoint scoring. Native zero-width/out-of-clip targets are not
  coerced into PAUC windows. Fifty-seven cases across 25 source videos have
  annotated clip ends up to 1.5 seconds beyond available media; inference ends at
  available physical frames while the official gold annotations remain intact.
- All 25 focused CPU tests, compilation, shell syntax, schedule-prefix equivalence
  and replay-capacity checks passed. Both existing allocations remain RUNNING;
  no new node was requested. At 20:24 UTC, tmux socket `pace-estp128` launched
  training step 38177.43 and evaluation step 38226.34 with CPU binding disabled.
  Both nodes asserted PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 90100. The
  seven-learner step-32 restore passed at 20:27 UTC with zero additional optimizer
  steps; step-33 learn/restore and cross-node parity remain required before full continuation. Launch does not establish
  correctness-gate success or improved quality.
- Quality/coverage/silence values cannot stop training. The durable supervisor
  archives uncommitted failed attempts and restores committed checkpoints after
  technical errors. Asynchronous evaluations are planned at 48/64/80/96/112.
  After all 128 updates, the training node releases its learners and runs the
  endpoint step-32 reference and step-128 suite, including IT test and full ESTP;
  final documentation is gated on all checkpoint/evaluation artifacts.

### 2026-11-15: four-source balanced continuation to step 160 launched

- User authorized 32 further updates from committed step 128, using two original,
  two WEB, two native EGO and two ESTP-IT episodes per update, four trajectories
  per episode, and objective weights 0.25 per source. RLOO, learning rates,
  original SFT reference, reward, constant-32 normalization and supervised replay
  coefficient are unchanged. See [plan](../plans/jevany_2b_four_source_balanced_to160.md).
- New run: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160`.
  Frozen phase: 64 episodes/source, 192,656 additional decisions, cumulative
  712,452 decisions. Native EGO reuses 64 of its already consumed 145 training
  episodes, explicitly accepted by the user. Original/WEB selections and 64 IT
  source videos are fresh; existing heldouts remain excluded.
- Thirteen CPU checks passed across the new schedule/source-weight tests and
  existing ESTP contracts. Parent checkpoint/source hashes and complete schedule
  prefix were verified; run-local snapshots and new-media hashes are frozen.
- At approximately 19:18 UTC, durable tmux sessions `balanced160-train` and
  `balanced160-eval` launched on existing allocations 38177 and 38226. Node-local
  temporary/compiler caches use `scripts/with_jevany_nvme_cache.sh`; durable
  artifacts and judge caches remain on FSx. Launch does not establish restore,
  first-update or cross-node parity gate success.
- New development evaluations are at 128/144/160 on the assigned evaluation node.
  After the continuation releases training GPUs, its supervisor recovers the
  parent's disk-space-failed final suite on the original training node, preserving
  original jobs/shards/judgments. Workflow completion requires training, new
  development evaluations and the parent endpoint recovery to finish.
- Step 129 committed successfully after 1,852.01 seconds for 6,572 decisions.
  All seven learners had nonzero head/LoRA gradients; maximum reserved learner
  memory was 53.43 GiB. Rollout/replay probability error was zero on every rank.
  Step-128 and step-129 exact restores passed with zero additional optimizer
  updates; the independent evaluation-node observation replay passed. The full
  continuation was released at 20:01 UTC, with baseline step-128 evaluation
  queued on allocation 38226. Training/evaluation were not yet complete at launch.

### 2026-11-16: balanced continuation and development evaluations completed

- All 32 requested updates completed through step 160 at 09:01 UTC. The final
  phase committed 192,656 decisions, reaching 712,452 cumulative decisions;
  64 episode selections/source, with 64 native EGO reuses and fresh selections
  from the other sources. No continuation recovery log was produced.
- Successful update time was 13.45 node-hours / 107.57 allocated GPU-hours,
  excluding loading, gates and evaluation. Maximum rollout/replay probability
  error across the entire phase was zero; peak reserved learner memory was
  74.67 GiB, below the unchanged 115-GiB gate.
- Development evaluations at 128/144/160 completed by 11:14:58 UTC. Full WEB
  PAUC-F: 57.974 → 58.182 → 58.426. Full EGO: 43.850 → 44.114 → 44.223.
  EGO heldout was essentially unchanged from step 128 (43.160 → 43.176);
  useful fires and temporal recall fell while precision improved.
- Versus the saved cohort-matched SFT baseline, step 160 gained +3.738 full-WEB,
  +1.497 full-EGO, +2.305 WEB-heldout, +0.692 EGO-heldout and +6.296 original
  validation points. The EGO-heldout interval includes zero. Full WEB/EGO contain
  RL training examples; these are reused evaluations and one training seed.
- The earlier 2,570-case endpoint suite resumed on the original training node
  after GPU release. Step-32 reference is complete; step-128 endpoint scoring is
  still pending. Do not claim official step-160 ESTP-Bench results.
- [Consolidated setup, lineage, exact recipe, comparisons and artifact index](jevany_2b_rl_current_setup.md).
  [Generated complete cohort table](jevany_2b_balanced_to160_results.md).

### 2026-11-16: ESTP explicit PAUC-F post-hoc scoring authorized

- After requesting no further model evaluations following the current endpoint,
  the user explicitly authorized additional GPT scoring to calculate the same
  explicit PAUC-F metric used in stream-gemma. This exception permits only
  post-hoc scoring of saved step-32/128 streams; no new model inference or
  step-160 ESTP evaluation was launched.

- Reused unchanged `zs_infer.estp_pauc_f`: 851 explicit cases, strict native
  windows, clip horizon, omega 0.5, eight-task macro aggregation. The protocol
  excludes 71 zero-duration gold turns and retains all physical fires. Absolute
  timestamp and identity adaptation was verified for every step-32 prediction.
- Step-32 scoring completed: task-macro PAUC-F 25.1553, pooled diagnostic
  24.8604; 817 cumulative-answer judgments / 623 distinct GPT-4.1 requests, all
  complete. This uses the PAUC rubric, not converted ESTP-F1 judgments.
- Durable CPU/GPT-only scoring waits for the existing full-step128 endpoint
  result, then computes the matching explicit PAUC-F comparison. Evidence:
  [post-hoc report](jevany_2b_estp_explicit_pauc_results.md), run-local
  `estp-pauc-posthoc-status.json`, and each endpoint's `estp_pauc_explicit/`.

### 2026-11-16: step-160 TV/VAD evaluation explicitly authorized and launched

- User requested TV and VAD evaluation of committed RL step 160, restricted to
  all eight GPUs on the current node `ip-10-3-70-177`, allocation 38226.
  Slurm step `38226.47` launches eight inference workers. No other evaluations
  are added by this authorization.
- Full TV 500 questions and VAD 101 questions resolve to available media;
  native end-aligned approximately 1-FPS sampling gives 50,227 decisions.
  Full clips, original annotation windows, visual-only input (subtitles off),
  the same frozen promoted generator, 96 answer tokens and raw learned argmax
  with response margin zero match the existing controller evaluation recipe.
- Committed checkpoint hashes and pinned PyTorch 2.5.1+cu121 / CUDA 12.1 /
  cuDNN 90100 verified. Temporary/compiler caches use local NVMe; durable
  predictions, action traces, GPT-4.1 PAUC-F judgments and summaries use FSx.
  Judging/scoring follows inference automatically. Results remain pending.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/tv-vad-step160/`.
  Entry points: `scripts/eval_jevany_tv_vad160.py prepare|run` and
  `scripts/run_jevany_tv_vad160.sh`. Original frozen evaluation sources are
  unchanged; the new job has its own manifest, source hashes and judge cache.

### 2026-11-16: step-160 controller with GPT-5.6 Sol WEB evaluation launched

- User explicitly requested full WEB with our RL controller and `gpt-5.6-sol`
  as generator, using the same gateway environment/credentials as GPT scoring.
  Both a text preflight and an actual WEB-frame image preflight returned HTTP
  200 from the exact requested model. No credentials are included in artifacts.
- All 500 existing WEB cases use the frozen step-160 controller and raw learned
  argmax (margin zero), unchanged native 1-FPS grid, 32-frame causal window,
  aged-out response memory and self-generated history. Sol answers may therefore
  change later firing times. The API generator uses a concise causal prose
  prompt, 96 completion tokens and `reasoning_effort=none`; it receives no
  gold answers or future frames. Failed/empty answers fail the run and preserve
  completed shards rather than introducing substitute responses.
- Eight controller workers launch on `ip-10-3-70-177` within allocation 38226.
  Sol image requests and subsequent GPT-4.1 PAUC-F scoring use the authorized
  gateway. NVMe holds compiler/temporary caches; FSx holds durable artifacts.
- The run automatically reports full WEB, RL training/heldout subdivisions,
  and paired bootstrap differences against saved step-160 MoE-generator results.
  Results are pending. Original evaluation files and completed runs are preserved.
- Ten focused API-payload/failure-handling and evaluation-policy tests passed;
  checkpoint integrity and pinned controller runtime are mandatory launch gates.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/web-sol-step160/`.
  Entry points: `scripts/run_jevany_web_sol.py prepare|run` and
  `scripts/run_jevany_web_sol.sh`; dedicated rollout:
  `scripts/eval_jevany_openai_streaming.py`.

### 2026-11-16: Sol EGO/TV/VAD parallel evaluation authorized

- User requested EGO, TV and VAD with the step-160 RL controller and the same
  `gpt-5.6-sol` generator as the completed WEB run, explicitly asking for three
  concurrent controller processes per GPU on this node. One eight-worker
  inference group per suite shares the eight GPUs of allocation 38226 on
  `ip-10-3-70-177` (24 controller workers total).
- WEB with Sol completed at 19:46:20 UTC: full WEB judged PAUC-F 58.515,
  heldout 58.055; timing-only full WEB 71.800, heldout 72.244. Sol versus saved
  step-160 MoE gains are +0.089 full WEB and +0.024 heldout; paired intervals
  include zero. No additional WEB inference is included in the new launch.
- New parallel suites retain exact existing media grids, original answer
  windows, visual-only input, 32-frame causal context, self-generated answer
  history, raw learned argmax with margin zero, and 96-token Sol answers with
  reasoning effort none. GPT-4.1 judged PAUC-F and CPU timing-only PAUC-F follow
  inference separately for each suite, including EGO training/heldout reporting
  and paired comparisons against saved step-160 MoE predictions. No other
  suites or checkpoints are added.
- Existing frozen WEB and MoE evaluation code/artifacts remain unchanged.
  The orchestrator holds the node's existing GPU lock once and starts three
  independent torchrun groups; each group's LOCAL_RANK maps its eight workers
  to the same eight GPUs. Source/checkpoint hashes and pinned runtime are checked.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/ego-tv-vad-sol-step160/`,
  with separate `ego/`, `tv/`, `vad/` shards, API audit trails, judge caches,
  timing-only results and status files. Launch entry point:
  `scripts/run_jevany_sol_parallel.sh`. Results pending.

### 2026-11-16: TV subtitles corrected to match stream-gemma MoE reference

- User explicitly required TV subtitles and an immediate rerun on the current
  evaluation node. The new run uses step-160 controller + GPT-5.6 Sol, with the
  native current-frame-only subtitle channel from the stream-gemma Qwen MoE
  reference `outputs/evals/tv_subtitles_current/qwen-w1-long-neg/`.
- All 500 questions and all 10,949 annotated subtitle turns match the reference.
  The reference uses measured source FPS (`source_fps=0`) and starts at physical
  frame zero; native 1-FPS sampling and causal subtitle binning reproduce that
  convention. Each subtitle appears only on the first frame that can see its
  timestamp, never in retained prior-frame text. Both controller and Sol receive
  the same current subtitle evidence; controller/system prompts remain fixed.
- This is a new subtitle-enabled inference run, preserving the earlier visual-only
  TV predictions and scores. It coexists with the authorized EGO run using eight
  additional controller workers on allocation 38226 / ip-10-3-70-177. The EGO
  global GPU lock stays intact; the new job has a dedicated lock and admits only
  the known EGO workers as co-resident GPU processes, with memory headroom checked.
- GPT-4.1 judged and CPU timing-only PAUC-F follow inference automatically. Native
  answer windows, response margin zero, raw learned argmax, Sol model/96-token
  settings, NVMe caches and durable FSx storage remain in effect.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/tv-subtitles-sol-step160/`.
  Entry point: `scripts/run_jevany_tv_subtitles_sol.sh`. The run freezes the
  reference summary and runtime subtitle source hashes. Results pending.

### 2026-11-16: Step-160 MoE TV with subtitles authorized

- User requested the missing promoted-MoE generator TV subtitle evaluation on
  all eight GPUs of allocation 38226 / ip-10-3-70-177; allocation and all eight
  empty GPUs verified before launch.
- Reuses the Sol subtitle manifest construction: 500 TV questions, measured FPS,
  samples from physical frame zero, causal current-frame-only native subtitles
  visible to both controller and MoE. Fixed 32-frame context, 96-token answers,
  self-generated history, pinned runtime and raw learned argmax with margin zero.
- Dedicated evaluator copies the existing MoE generation path without modifying
  frozen earlier evaluators. Checkpoint/generator hashes, media metadata, subtitle
  coverage and raw-argmax traces are checked. GPT-4.1 judging and unchanged PAUC-F
  plus timing-only scoring follow automatically. Earlier results are preserved.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/tv-subtitles-moe-step160/`.
  Entry point: `scripts/run_jevany_tv_subtitles_moe.sh`.
- Launched as Slurm step `38226.52`; all eight worker processes are active and
  initializing controller/MoE models on the eight GPUs. Runtime matches
  torch 2.5.1+cu121 / CUDA 12.1 / cuDNN 90100. All eight focused causality/policy
  checks passed. The manifest is byte-identical to completed Sol subtitle TV,
  including all 10,949 native subtitle turns. Results pending.
- Completed 2026-11-16 at 22:19:59 UTC: all 500 questions, GPT-4.1 judging
  and scoring complete. Judged PAUC-F (omega 0.5) is 27.5122; timing-only
  diagnostic 34.3794. There were 450 fires, 112 useful and 338 wasted.
  Difference versus prior visual-only MoE is +0.4066 points (paired 95% CI
  [-2.1284, 2.9554]); sampling origin also differs, so this is not a controlled
  subtitle-only comparison. Completed subtitle Sol is 27.4791.

### 2026-11-16: step-160 2B + MoE StreamingBench PO/PO-I campaign

- User approved the high-level PO/PO-I plan and requested full completion and
  final numbers. Frozen checkpoint: balanced-v4 step 160; answer generator:
  promoted Qwen3.5-35B-A3B with unchanged stream-gemma long-neg adapter.
- Eight one-GPU workers on allocation 38226 / ip-10-3-70-177. The fixed workload
  contains 250 ordinary PO questions and 50 PO-I videos, each with independent,
  append and replace-scoped request policies. Duration-balanced planned costs
  are 10,203–10,204 frame steps per rank (81,628 total before smoke overhead).
- Reuses the pinned PO-I manifest, original query/response timestamps, trained
  streaming/request hooks and deterministic scorers. Native 1 FPS, 32-frame
  history, v2 aged-out response memory, no subtitles, 128-token greedy MoE
  answers, raw learned controller argmax and response margin zero.
- Nineteen CPU tests passed, including request-hook prompt equivalence,
  memory scoping, scoring and raw-action policy checks. A complete five-request
  sample_15 GPU smoke gates full inference and compares independent-policy
  emissions against ordinary PO with the same available horizon.
- Ordinary PO reports official timing and corrected first-nonempty-response
  timing separately, preserving the reference blank-placeholder audit. PO-I
  reports timing/answer/joint/stale metrics on 185 differing-output transitions,
  all-request diagnostics and 10,000 physical-video paired bootstraps. No GPT
  judge calls are needed.
- Artifacts: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/sb-po-poi-moe-step160/`.
  Entry point: `scripts/run_jevany_sb_moe.sh`; runtime adapter:
  `scripts/jevany_po_moe_runtime.py`. Source/model/data hashes are frozen;
  per-task atomic records support recovery. Results pending at launch.

### 2026-11-16: step-160 + Gemini Flash four-suite evaluation completed

- EGO, TV with current-frame-only native subtitles, and VAD completed inference,
  GPT judging, and unchanged PAUC-F scoring. Their full-suite omega-0.5 scores
  are 42.6107, 29.2891, and 41.2679 respectively; EGO heldout is 43.6626.
- The original WEB launcher stopped at 277 episodes because Flash returned
  HTTP 200 with no candidates and `promptFeedback.blockReason=PROHIBITED_CONTENT`.
  The captured frames show a kitchen scene; Google gives no per-frame/category
  explanation. Repeated original retries did not resolve the provider block.
- The user explicitly authorized skipping blocked entries, continuing the rest,
  and assigning a perfect score to blocked entries if they still failed.
  Isolated recovery on allocation 38177 preserved all 277 completed episodes,
  attempted the remaining 223 with unchanged inputs and learned decisions,
  and finished with 498 measured and two provider-blocked episodes.
- Blocked IDs are `web:RjC6aV1SNIQ.3.mp4` and `web:io6ppR83ZMU.9.mp4`, both
  RL training examples. They have explicit post-hoc perfect-score imputations;
  no synthetic answer or judge result is passed off as a model measurement.
- WEB measured-only omega 0/0.5/1 PAUC-F: 52.6869 / 57.0401 / 61.0009.
  Including user-requested perfect imputations: 52.9423 / 57.2641 / 61.1994.
  Heldout WEB is unaffected: 52.3824 / 56.5694 / 60.4179.
- Combined report, cohort distinctions, raw provider audits, and results are in
  `evaluation/web-ego-tv-subs-vad-flash-step160/{completion-report.md,result.json,three-omega-summary.json}`
  under `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/`.

### 2026-11-17: step-160 2B + MoE PO and PO-I completed
- Finished at 2026-11-17T00:09:31.045008+00:00; all 250 PO questions, 50 PO-I videos and three policies completed. All eight GPU workers exited successfully and the GPUs are idle.
- The complete five-request GPU smoke passed exact independent/ordinary-PO response comparisons. Nineteen CPU tests passed; full traces validated raw learned argmax, causal windows and scoped request memory. Frozen source hashes were verified after scoring.
- PO official time accuracy **32.8%** (82/250), corrected first-nonempty time **21.2%** (53/250), answer **13.6%** (34/250). The official total includes 29 blank-placeholder timing credits.
- PO-I primary population: 185 differing-output request transitions across 50 physical videos. Percentages below use the existing deterministic timing/answer/stale scorers. Full 10,000 paired physical-video bootstrap intervals and all-request diagnostics are in the saved result.

| PO-I policy | Timing | Answer | Joint | Stale videos |
|---|---:|---:|---:|---:|
| Independent | 28.11% | 20.00% | 20.00% | 2.00% |
| Append | 23.24% | 14.59% | 14.59% | 12.00% |
| Replace scoped | 22.16% | 14.59% | 14.59% | 10.00% |

- Model remains the committed step-160 controller with the frozen promoted MoE generator; no training, inference score offsets or GPT judging were added.
- Durable results and audit: `evaluation/sb-po-poi-moe-step160/{result.json,validation.json,smoke-passed.json,po/,poi/,records/}` within the balanced-v4 run. No further evaluation jobs are queued by this campaign.

## 2026-11-17 — fresh PACE/ESTP-IT supervised controller, 16-hour run launched

New lineage: pinned official Qwen3.5-2B, fresh rank-64 decoder LoRA and pointer
head; cross-entropy on exactly 16 PACE plus 16 ESTP-IT decisions per update.
The user explicitly placed this SFT on the current node, allocation 38226 /
ip-10-3-70-177, with asynchronous evaluation on 38177 / ip-10-3-120-167.
This is a documented exception to the older RL placement, not RL training.

Run: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-seed42-16h`.
Budget: 57,600 accumulated training seconds; full WEB 500, EGO 326 and IT
Explicit development 4,894 at hours 2/4/6/8/10/12/14/16, with frozen promoted
MoE answers and GPT-4.1 judging. Native windows, self-generated histories and
raw argmax/margin zero remain fixed. IT test and ESTP-Bench are excluded.
Separate four-trajectory readiness diagnostics run at hours 8 and 16.

Startup validation passed: four CPU contract tests, first-update gradient and
checkpoint gate, eight-observation cross-node replay, exact committed step-28
optimizer/RNG round trip, GPT gateway access and three complete inference/scoring
smoke cases. The node's NCCL CUDA-802 failure occurred before any optimizer
update; training uses the original SFT's exact CPU/Gloo gradient average while
all eight GPUs do forward/backward.

Validated controller batching accelerates full evaluations; singleton MoE
generation is unchanged. Across 401 smoke decisions no actions or answers
changed, and maximum probability difference was 0.00125731, below the existing
0.005 tolerance. Each checkpoint retains batch/singleton checks and automatic
original-evaluator fallback on technical failure. Preserve all scheduled
evaluations if a backlog develops.

Status is **running**, not completed. `training-complete.json` covers training
only; `complete.json` requires all eight judged evaluations and both readiness
probes. The orchestrator then writes `report.md`. No improvement claim follows
from startup checks. See the run's README and
`docs/plans/pace_estp_joint_sft_16h.md` for the full frozen procedure.

**Stopped by the user on 2026-11-18 before the first full evaluation.** The
automatic monitor and this run's training/evaluation Slurm steps were stopped;
both existing allocations remain available and all sixteen GPUs were verified
at 0 MiB usage. Last observed progress was step 593; the latest committed
checkpoint is step 489. Artifacts are retained under the run above, including
`user-stopped.json`. A replacement prefetched large-batch run has not been
launched; the user requested its high-level plan first.

### 2026-11-18: efficient mixed SFT replacement validation

User approved the fresh-base global-384 replacement with 192 PACE and 192 ESTP-IT decisions per update, profiling physical 48 versus accumulated 24, a profiled fixed-step cadence for eight approximately two-hour intervals, and full WEB/EGO/IT Explicit development evaluations. Checkpoint zero is saved for correctness smoke only because the pointer head is random. Training/evaluation retain allocations 38226/38177 respectively. Artifacts: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h`.

An initial scratch BF16 batch-versus-singleton check failed (rank-zero maximum probability difference 0.049423; head-gradient relative L2 0.043079); no production weights were trained or promoted. No-update diagnostics showed full-precision batch/singleton probability agreement within 0.000012, identifying geometry-dependent BF16 numerics rather than a padding/normalization discrepancy. Revised validation preserves the original structural probability/gradient/loss tolerances using a full-precision reference, and bounds native batch numerical error against native singleton error to that same reference. Across all eight first-batch rank checks, FP32 probability errors were at most 0.000009 and head-gradient relative errors at most 0.000021; native BF16 passed the documented regression envelope. Scratch failures, old source snapshot and freeze, and diagnostic logs remain preserved. The physical-48 long-context capacity test follows before measured profiling and fresh production initialization. No quality findings are available yet.

Both candidate configurations passed full-precision structural parity, native BF16 numerical regression, and long-context capacity validation. Seven steady updates after three warm updates measured 50.3286 seconds/update for physical 48 (105.609 GiB peak reserved) and 64.3033 for physical 24 accumulated twice (57.428 GiB). The frozen choice is physical 48/global 384, with K=143 and total 1,144 updates, estimating 57,575.9 seconds of training updates. Full evaluation steps: 143, 286, 429, 572, 715, 858, 1001, 1144. Expected total decision presentations: 439,296, equally split by source. Fresh production initialization follows; scratch weights are discarded. GPT credential loading was validated on the assigned evaluation node without exposing credentials or launching quality scoring. Final training and evaluation findings remain pending.

Fresh production started successfully after the selected profile. Committed step-zero head and adapter hashes exactly match the untouched scratch initialization; no trained scratch weights carry over. The first production update committed with the expected source counts and learning-rate warmup. The evaluation node reproduced all eight training-rank singleton observations exactly (maximum probability error zero in every case), releasing the learner from its correctness gate. Training is ongoing; checkpoint-zero correctness smoke and all scheduled quality evaluations are separate stages. Supervisors, bounded launcher recovery, GPU telemetry and CPU-only saved-result readiness reporting are active. No quality conclusion is available at launch.

### 2026-11-18: continue mixed SFT to the prepared ESTP-IT epoch boundary

After the initial 1144-step run completed in 38383.675 training seconds, the user authorized continuation on the same training node, followed by investigation and repair of slow evaluation. The continuation at `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1` resumes the committed step-1144 adapter/head, AdamW state, eight RNG states and deterministic sampler offset. It targets cumulative step 2275 (1131 additional updates), completing the 436629 prepared ESTP-IT rows while maintaining equal source presentations. Each source will have 436800 presentations; PACE thus reaches approximately 1.66 passes of its curated 263760-row manifest. This is not a switch to the separate 8.34M-decision PACE source manifest and is not one additional ESTP-IT epoch.

Training remains on eight GPUs in allocation 38226 / ip-10-3-70-177, physical batch 48/global 384, head LR 0.00035 and LoRA LR 0.000007. Warmup is not restarted. Existing original evaluation jobs, checkpoints and partial predictions on allocation 38177 remain preserved; no additional periodic full suites are enqueued by this continuation. After training, compare earlier evaluation cohort sizes/protocols and measured phase timings before validating a faster evaluator with unchanged learned argmax, generator, histories and scoring. Evaluation repair remains pending, not completed.

The first resumed update reached step 1145 with finite loss 0.22827 and gradient norm 3.10301, source counters 219840 each, and unchanged post-warmup learning rates. Durable recovery supervision and GPU telemetry are active. Checkpoints save every approximately 900 accumulated training seconds and at the endpoint.

### 2026-11-18: correct SFT evaluation population; score completed EGO

The user clarified that the requested third evaluation set is ESTP-Bench Explicit, not ESTP-IT Explicit development, and instructed stopping the latter, completing WEB first, then running ESTP-Bench Explicit. The obsolete combined evaluation step 38177.76 and its restart monitor were stopped. Its predictions and original queue remain as audit artifacts, guarded by the parent stop marker. SFT epoch continuation on allocation 38226 was left running. The corrected checkpoint-143 workflow on allocation 38177 is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h/evaluation/corrected-step143`.

The corrected populations are WEB: 500 episodes / 500 media files / 8803 controller decisions; EGO: 326 episodes / 326 media files / 117684 decisions; ESTP-Bench Explicit: 851 episodes / 70 source video IDs and media files / 241138 decisions. The cancelled IT development workload had 1447186 decisions, approximately six times the intended ESTP-Bench workload. This population mismatch explains a substantial part of the slowdown in addition to the failed batching correctness gate. Do not compare the IT development proxy with the separate ESTP-Bench explicit benchmark as if they were the same population.

At the user's explicit request, all 326 completed EGO predictions were snapshotted and validated for population, full trace lengths and raw learned argmax, then GPT-scored without new inference using unchanged native-window PAUC-F and the promoted frozen MoE answers. GPT-4.1 completed all 377 unique requests (393 before deduplication); scoring wall time was 39.990 seconds. Overall EGO pooled PAUC-F is 30.163680, fire precision 29.105839%, temporal recall 31.301314%; 319 useful fires of 1096 physical fires, 1575 gold events. Mean per-episode PAUC-F is 23.194217. Historical RL identity subsets: 145 prior-train episodes score 28.124133 pooled; 181 prior-heldout episodes score 31.585812 pooled. This fresh SFT itself used no EGO training data. EGO inference elapsed reconstructed from the maximum saved worker EGO completion time is 4757.4 seconds (79.29 minutes), excluding model loading and the subsequent IT waiting time.

WEB standalone inference was launched first; ESTP-Bench Explicit inference and unchanged estp-pauc-f-v1 judging follow only after WEB scoring completes. A bounded monitor supervises only this corrected workflow. Final video/episode counts and separate inference/GPT-scoring timings will be generated in its report.md. WEB and ESTP scores and timing findings are still pending at this entry.

The user additionally authorized exactly one latest-checkpoint evaluation after the current corrected checkpoint-143 suites finish. `evaluation/latest-after-step143` records a durable dependency on the current complete.json and launcher GPU release. At dispatch, its watcher will choose the newest fully committed epoch-continuation checkpoint, verify and pin its commit, and run WEB, EGO and ESTP-Bench Explicit with fresh inference for all three. Checkpoint selection occurs once; it does not follow training updates during evaluation. Source hashes, request/contract-keyed judgment reuse and bounded recovery are recorded. The cancelled IT queue remains stopped.

Checkpoint-143 WEB completed while that follow-up was queued: all 500 episodes, pooled PAUC-F 51.402116, fire precision 79.677419%, temporal recall 37.938719%, 494 useful fires out of 620, and 1328 gold events. Inference wall time including model loading and merge: 1346.521 seconds (22.442 minutes). GPT/scoring: 41.131 seconds. Historical RL identity subsets: 400 prior-train videos score 51.439953 and 100 prior-heldout videos score 51.265027. This fresh SFT itself used no WEB training data. ESTP-Bench Explicit inference follows on the evaluation node. EGO remains complete at 30.163680 pooled PAUC-F. The follow-up watcher is verified waiting for the current three-suite completion, and has not selected a future checkpoint yet.

### 2026-11-18: stop evaluations, freeze a 200-case explicit subset and profile faster execution

The user requested stopping current evaluations and the queued latest-checkpoint run, reducing ESTP-Bench Explicit to approximately 200 episodes, and investigating faster GPU execution with a total per-checkpoint budget under three hours. Stopped evaluation step 38177.81, its monitor and the latest dispatch watcher. Preserve completed checkpoint-143 WEB/EGO scores and all partial benchmark traces. The separate SFT epoch continuation on 38226 was not stopped. Only the authorized one-GPU technical pilot subsequently ran; production evaluations remain paused.

The proposed fixed seed-42 sample contains 25 randomly drawn whole episodes per native explicit task: 200 episodes, 59 source videos, 56741 native decisions. It uses no quality, model-output or duration filtering. Full WEB remains 500 episodes / 8803 decisions; EGO remains 326 / 117684. The combined workload has 1026 episodes and 183228 decisions, balanced across eight ranks at 22897–22907 decisions/rank. Execution prioritizes WEB then EGO then ESTP within each rank, with independent reset histories. This is a fixed diagnostic subset and must not be reported as the official full 851-case headline.

Implemented a pinned direct-GPU MoE loader and a native-evaluator copy with guarded lightweight controller inputs. The latter transfers cached frozen visual features while avoiding redundant raw-patch concatenation/transfer, without changing IDs, grids, scores, argmax or generator requests. Model loading happens once per GPU across the combined three-set manifest. No padded controller batching or runtime decision heuristic was added.

A one-GPU A/B/A pilot at checkpoint 143 replayed four complete native reference episodes, 957 decisions and 12 generated answers, including full 32-frame histories and aged-out memory. All three passes had zero action mismatches, identical answers and maximum action-probability error zero. Comparing warm native A2 to cached B, episode time fell 246.304 to 190.623 seconds (22.606% less time; 1.292x throughput), controller time fell 210.447 to 154.511 seconds (1.362x throughput), and mean active-GPU utilization rose 58.765% to 65.880%. Resident GPU memory stayed 74.542 GiB. The generator's direct load took 166.339 seconds on one GPU; this is not a measured eight-GPU startup improvement. No GPT judging or weight updates were run during the pilot.

Using prior measured eight-GPU rates, the fixed decision counts and this one-GPU warm speedup projects approximately 113 minutes including conservative 15-minute loading and three-minute judging allowances, or 132 minutes with an additional 20% compute margin. Plan roughly 2–2.5 hours; this is not a measured complete newest-checkpoint run. The prepared runner gates on passing parity/source hashes, pins one checkpoint, records population and phase timings, and self-cancels its own Slurm step at 179 minutes, with a recommended --time=03:00:00 outer step limit. Timeouts preserve partial artifacts and never create a completed evaluation. No automatic over-budget restart or full production run has been launched.

Artifacts and reviewable plan: `outputs/jevany-qwen35-2b-pace-estp-sft/eval-speed-design-v1/README.md`; frozen data manifests and 200-case selection under `data/`; live parity/timing evidence under `pilot/`; prepared entrypoint `scripts/run_pace_sft_budget_eval.sh`.

### 2026-11-18: launch faster evaluation of the latest saved checkpoint, without a time cutoff

The user authorized running the newer validated faster implementation and explicitly removed the three-hour hard limit, requesting completion and actual measured times. At dispatch the latest committed checkpoint was SFT step 1494, while live training had reached 1500. Pinned step 1494 for the whole run. Launched full WEB 500, full EGO 326 and fixed ESTP-Bench Explicit 200 (59 videos, 25 episodes/task) as one persistent eight-GPU inference job on 38177, followed by unchanged native GPT-4.1 judging. The job uses the validated direct-GPU MoE loader and cached controller inputs, preserves native singleton raw argmax and histories, and does not resume any cancelled queue.

Removed the prepared runner's wall-time watchdog and self-cancel logic; no Slurm --time limit was supplied. The freeze explicitly records run-to-completion and hard_time_limit=false. A bounded technical recovery watcher pins the same checkpoint and preserves the original start time. The report records actual overall elapsed wall time, shared model loading and per-set worker processing/GPT timings, with no timeout-based truncation or missing-case imputation. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-001494`. Initial state verified as combined_inference, zero restarts. SFT continues separately on 38226.

### 2026-11-18: faster checkpoint-1494 evaluation completed, without a wall-time cutoff

Completed all 1026 episodes and GPT scoring at 20:20:47 UTC, with zero restarts. Actual complete-run elapsed time: 6068.242 seconds, or 101.137 minutes (1h 41m 8s). Combined inference including shared loading/merge: 6039.616 seconds. GPT/scoring stage times sum to 135.967 seconds, with native WEB/EGO scoring overlapped against remaining GPU inference. No hard limit or timeout was applied. All 183228 saved controller decisions passed raw-argmax and complete-trace checks. Evaluation GPUs were verified free after completion; the allocation was retained. SFT continues separately (step 1700 of 2275 at the final check).

| Set | Videos | Episodes | Max-worker processing (excludes shared loading) | GPT/scoring | PAUC-F |
|---|---:|---:|---:|---:|---:|
| WEB | 500 | 500 | 12.355 min | 68.554 sec | 53.659303 |
| EGO | 326 | 326 | 55.885 min | 53.367 sec | 32.297094 |
| ESTP-Bench Explicit fixed subset | 59 | 200 | 26.577 min | 14.046 sec | 19.118450 |

Per-set processing is the maximum sum of episode wall times over eight workers. Suite transitions and native GPT scoring can overlap, so rows are not additive; the 101.137-minute complete-run wall time is the authoritative overall measurement. Frozen generator loading took 392.016–392.880 seconds on the eight GPUs, once per GPU across all sets. The ESTP score is the standard eight-task macro on the fixed 25-per-task diagnostic subset, not the full 851-case headline.

WEB pooled score changed from 51.402116 at step 143 to 53.659303 at 1494 (+2.257187 points); EGO changed from 30.163680 to 32.297094 (+2.133414). These are descriptive checkpoint comparisons, not proof of continued improvement or statistical significance. Historical WEB identity subsets: prior-RL-train 400 videos 54.532771, prior-RL-heldout 100 videos 50.329484. Historical EGO identity subsets: prior-RL-train 145 videos 32.307393, prior-RL-heldout 181 videos 32.288643. This fresh SFT used only PACE/ESTP-IT. The 200-case ESTP score must not be directly treated as the same-population result as historical full-851 benchmark scores.

Report: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-001494/report.md`. The complete JSON, per-set results, raw predictions, all judging artifacts, source/input hashes and code snapshots are preserved. No additional checkpoint evaluation or RL run was enqueued.


### 2026-11-18: requested latest-checkpoint faster evaluation, step 1928


On 2026-11-18 the user explicitly requested evaluating the latest SFT checkpoint again on the other node with the efficient setup and the same three sets. Pinned the newest fully committed checkpoint, step 1928, after verifying head/adapter checksums and unchanged validated evaluator hashes. Launched one eight-GPU evaluation on allocation 38177 / ip-10-3-120-167: full WEB 500, full EGO 326, and the same fixed seed-42 ESTP-Bench Explicit 200 episodes (59 videos). Preserve raw argmax/margin zero, frozen promoted MoE generator, native GPT scoring, run-to-completion without a wall-time cutoff and bounded technical recovery at the same checkpoint. Outputs: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-001928`. SFT continuation on 38226 remains unchanged; no additional sweep or RL is authorized by this request.


### 2026-11-19: evaluate step 2275 and continue unchanged SFT for another epoch


On 2026-11-19 the user authorized evaluating the latest committed SFT checkpoint on the evaluation node and continuing SFT for another epoch on the training node. Step 2275 is pinned for the separate faster evaluation on 38177 / ip-10-3-120-167: full WEB 500, EGO 326 and the unchanged fixed ESTP-Bench Explicit 200, including native GPT scoring and actual timings, no hard cutoff. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-002275`.

A separate continuation resumes committed step 2275 through step 4550 (2275 additional updates, one additional prepared ESTP-IT row pass while keeping equal PACE presentations), on 38226 / ip-10-3-70-177. Run: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2`. Restore learned adapter/head, AdamW, eight RNG states and deterministic sampler cursor; retain physical batch 48/global 384, head LR 0.00035, LoRA LR 0.000007 and completed warmup. Data, native labels, sampled negatives, source-video splits, loss and sampling recipe remain unchanged; task balancing/hard negatives discussed earlier are not enabled. Save every 900 accumulated training seconds and at endpoint. Both jobs use NVMe caches and durable FSx artifacts, raw learned argmax/margin zero; independent bounded recovery monitors preserve pinned checkpoints. No automatic later evaluation sweep or RL. Earlier frozen scripts/runs remain intact.


### 2026-11-19: queued three more SFT epochs on the same eight GPUs


On 2026-11-19 the user explicitly authorized continuing the current SFT through five total epochs: after epoch two finishes at committed step 4550, resume that endpoint for three more prepared ESTP-IT epochs (6825 additional updates) through cumulative step 11375. The user reiterated using the same eight GPUs on this training node, ip-10-3-70-177 / allocation 38226 only. The queued continuation is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5`. Its persistent monitor waits for epoch-two training-complete, launcher exit and GPU release, verifies every committed endpoint file, then dispatches on allocation 38226 with bounded recovery. Preserve physical48/global384, 50:50 PACE/ESTP-IT decisions, head LR0.00035 / LoRA LR0.000007, AdamW, completed warmup, rank RNG states, learned weights and deterministic sampler cursor. No changes to native labels, negatives, task frequencies, loss or source-video splits. Save every900 accumulated training seconds and final endpoint with durable FSx artifacts and NVMe caches. This later authorization supersedes the prior epoch-two wrapper's generic no-further-training next-action text only for this explicitly requested continuation; do not modify the frozen running epoch-two sources. No new evaluations or RL are authorized by this request. About59 additional training hours after epoch two at recent31sec/update; retained allocation expires2026-12-06, sufficient for the planned work.


On 2026-11-20 the user explicitly requested the epoch-two endpoint evaluation. Launched pinned step4550 on allocation38177 / ip-10-3-120-167, using full WEB500, EGO326 and the unchanged fixed ESTP-Bench Explicit200 with the validated faster evaluator, frozen MoE, raw argmax/margin0, GPT judging and no hard cutoff. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2/evaluation/fast-step-004550`. SFT through five epochs continues separately on38226. Evaluations are explicitly requested rather than periodic; training checkpoints are saved about every900 accumulated training seconds.

The user subsequently authorized exactly one follow-up evaluation after step4550 finishes all inference and scoring. At dispatch select the newest fully committed checkpoint available in the active epoch-five continuation, skip it if already completely evaluated, and otherwise pin it for the same three-set faster suite and all technical retries on38177. Wait for the step4550 job to exit before acquiring evaluation GPUs. Do not turn this into a recurring sweep. Preserve all results and unchanged policies. Queue: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2/evaluation/latest-after-step4550`.


On 2026-11-20 the user amended the one-shot latest-after-step4550 follow-up: if the newest committed checkpoint is already evaluated, keep checking every600seconds for a new committed checkpoint instead of ending with a skip. Once a new checkpoint is found, pin and launch exactly one three-set evaluation, then stop dispatch polling. Preserve the active step4550 evaluation and five-epoch training. The waiting dispatcher was restarted with this policy; no GPU jobs were interrupted.


On 2026-11-20 the user explicitly required preserving the epoch-one SFT checkpoint at `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/train/step-002275`. All files listed in commit.json were verified by SHA256; learned head/adapter, optimizer/training state and all eight rank RNG states are intact. RETAIN.json marks the checkpoint for retention. Do not delete, prune or overwrite this checkpoint without explicit user authorization. Preserve its evaluation artifacts as well. This retention request does not change the active five-epoch training.


On 2026-11-20 the user explicitly requested evaluating the current latest checkpoint. Verified checkpoint and unchanged evaluator hashes, then pinned SFT step6249, the latest committed checkpoint at dispatch, for full WEB500, full EGO326 and the same fixed ESTP-Bench Explicit200. Launched on all eight GPUs of evaluation allocation38177 / ip-10-3-120-167 using the validated faster setup, frozen promoted MoE, raw argmax/margin0, native GPT scoring, no hard cutoff and bounded technical recovery. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5/evaluation/fast-step-006249`. Training through step11375 on38226 continues unchanged; protected epoch-one step2275 remains retained. No additional follow-up or periodic sweep is queued by this request.


On 2026-11-20 the user requested preparing a continuous latest-checkpoint evaluation script, explicitly requiring confirmation before launch. Created `scripts/watch_pace_sft_latest_eval.py`; IT IS NOT LAUNCHED. Default invocation is preview-only; --run is required for dispatch. After user confirmation it should serially evaluate the newest fully committed checkpoint from the active epoch-five continuation on allocation38177, full WEB500/EGO326/fixed ESTP-Bench Explicit200, using the unchanged validated fast runner, raw argmax/margin0, frozen MoE and GPT scoring. It pins each selection through completion/recovery, skips intermediate checkpoints rather than building a backlog, skips completed matching checkpoint/protocol results, polls every600seconds when caught up, waits for existing evaluations and GPU release, persists a bounded four-attempt budget and stops with needs-attention on technical exhaustion. It evaluates the final training endpoint and exits when caught up after training completion. A STOP file at the training run's evaluation/continuous-latest/STOP stops future dispatches while allowing an active suite to finish; restarting without STOP resumes a persisted active selection. State and shared file locks prevent concurrent copies. No running training/evaluation jobs were changed. Six temporary-artifact scheduler tests passed; preview-only validation selects the current newest commit without starting jobs. Await explicit user confirmation before starting this watcher or new inference/judging.


On 2026-11-20 the user explicitly confirmed launching the prepared continuous evaluation watcher. Started `scripts/watch_pace_sft_latest_eval.py --run`; state/logs under `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5/evaluation/continuous-latest`. This supersedes the previous preparation-only/no-launch instruction and authorizes serial latest-checkpoint inference and GPT scoring until the final training checkpoint has been evaluated. Use allocation38177 / ip-10-3-120-167 eight GPUs only for evaluation, unchanged full WEB500/EGO326/fixed Explicit200 faster protocol, raw argmax/margin0, frozen MoE and native scoring, no wall-time cutoff. Skip intermediate checkpoints while busy, poll every600seconds when caught up, preserve bounded technical recovery and all artifacts. Training on38226 continues unchanged; protected step2275 remains retained.
