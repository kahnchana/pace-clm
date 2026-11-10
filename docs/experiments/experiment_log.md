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

### 2026-11-09: Reward-aligned RLOO completed

- Reward-aligned RLOO stopped at round 16 for useful-response coverage below
  90% of SFT at consecutive checks. Full 500-video GPT-4.1 WEB: 54.5561
  PAUC-F, 605 useful / 187 wasted versus retained SFT 54.6651, 701 / 290.
