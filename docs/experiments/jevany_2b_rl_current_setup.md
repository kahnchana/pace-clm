# Current JevAny 2B controller: training setup and results through step 160

This is the consolidated human-readable record for the learned controller at
step 160. Its 32-update balanced continuation and development evaluations at
128/144/160 are complete. The earlier run's larger step-128 endpoint suite is
separate and remains pending until its official scoring and completion marker
are saved. Status and tables below are tied to saved artifacts, not launch intent.

## Checkpoint lineage and what changed

The original SFT controller is
`outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
It is Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`.

| Optimizer steps | Initialization | Episodes per update | Configured source weights |
|---|---|---|---|
| 1–32 | Fresh original SFT, then exact continuation of that same RL run | 4 original, 2 WEB, 2 native EGO | Original 0.50, WEB 0.25, EGO 0.25 |
| 33–128 | Restore step 32, including optimizer/RNG/sampler state | 4 original, 2 WEB, 2 ego-family episodes | Original 0.50, WEB 0.25, combined EGO/IT 0.25 |
| 129–160 | Restore step 128, including optimizer/RNG/sampler state | 2 original, 2 WEB, 2 native EGO, 2 ESTP-IT | Each source 0.25 |

Steps 33–128 contain 81 batches with one native EGO and one IT episode, and
15 batches with two IT episodes. The first 32 schedule entries remain unchanged.
Every phase uses four complete trajectories per episode and one optimizer update
per collection. The final phase therefore collects 32 trajectories per update,
1,024 trajectory presentations across 256 episode selections, and exactly 32
additional optimizer updates. It does not use GRPO or sixteen trajectories per
episode. No fresh LoRA or optimizer was initialized at either continuation.

Lineage and operational history:

- [Through step 32](jevany_2b_episode_balanced_mixed_results.md).
- [Step-32 to step-128 plan](../plans/jevany_2b_estp_episode_balanced_to128.md).
- [Step-128 to step-160 plan](../plans/jevany_2b_four_source_balanced_to160.md).
- [Generated continuation tables](jevany_2b_balanced_to160_results.md).

## Exact training recipe

| Setting | Value |
|---|---|
| Learned parameters | Decoder LoRA across 24 layers: 62,521,344 parameters; pointer head: 1,049,088 parameters |
| LoRA architecture | Rank 64, alpha 128; target modules and complete adapter configuration retained from SFT |
| Frozen parameters | Base model, vision encoder, original SFT reference adapter/head and promoted answer generator |
| Dropout during RL | Zero on every dropout module; inherited adapter metadata still records its SFT dropout of 0.05 |
| Optimizer | AdamW, pointer-head LR 1e-5, LoRA LR 1e-6, weight decay 0.01 |
| Gradient clipping | Global norm 1.0 |
| RLOO advantage | Each trajectory reward minus the mean reward of its other three trajectories; no reward-standard-deviation normalization |
| Credit assignment | The trajectory's scalar advantage is attached to all its controller decisions |
| Actor optimization | One on-policy pass and one globally synchronized optimizer step per collected batch; no PPO/GRPO likelihood-ratio clipping or extra passes |
| Training action sampling | Categorical sampling directly from the learned three-action probabilities, equivalent to temperature 1 |
| Event-containing reward | Official per-episode GPT-4.1 judged PAUC-F / 100, omega 0.5, native windows preserved |
| Event-free reward | -0.25 per emitted fire, retained from the audited reward contract |
| SFT KL coefficient | 0.02, exact categorical action KL against the original SFT reference |
| Entropy bonus | 0.001 |
| RL loss normalization | Sum decision losses divided by global trajectory count times constant 32; no per-episode-length normalization |
| Supervised replay coefficient | 0.0035140820988695176, restored unchanged; no recalibration for the balanced phase |
| Replay credits | 126 examples per 4,096 cumulative RL decisions, using integer cumulative credits; 5,927 new credits in the final phase, 21,916 cumulative |
| Replay bank | Cycle the original frozen rows in order, preserving every earlier row/credit selection and coefficient |
| Seed | 42; learner RNG states restored and saved per rank |
| Collection/replay batch caps | Logical microbatch, physical batch and collection caps 24; generator batch cap 8 |
| Memory/input batching | Frame cache 2,048; token bucket 512; replay token budget 262,144; exact original collection shapes retained |

For source `s`, each trajectory receives loss scale
`0.25 * global_trajectories / trajectories_from_s` in the final phase.
With eight trajectories from each source, all scales are 1. The configured
quarter weights describe source means under constant-32 normalization; they do
not make the four sources contribute equal numbers of action decisions. Longer
episodes still contribute more decision losses. EGO and IT together account for
86.5% of final-phase action decisions.

The loss is the source-scaled sum of `-advantage * selected_action_log_probability
+ 0.02 * SFT_KL - 0.001 * entropy`, divided by the global trajectory count times
32, plus the separately globally normalized supervised replay term. Saved
collection peers, row positions and padded shapes are reused for likelihood
replay; uneven work across seven learners is globally normalized.

## Inputs, generator and inference

Episodes run causally at 1 FPS with at most 32 recent frames per decision and
144 visual tokens per frame. J1 inputs retain actual self-generated action/answer
history, timestamps and the latest aged-out response memory. Decoder context is
rebuilt per decision. Gold answers, future frames and reward judgments do not
enter controller inputs.

The answer generator is the frozen promoted Qwen3.5-35B-A3B MoE with the existing
`stream-gemma/outputs/lora-qwen-w1-long-neg` adapter. It uses the existing greedy
forced-response path and a 96-new-token cap; controller training does not backpropagate
through answer tokens. Its model/adapter/input hashes are in the frozen manifests.

Evaluation takes the raw learned action argmax, with response margin zero.
No action offsets, score mixing, calibration, response thresholds, cooldowns or
heuristic suppression were introduced. The inherited v2 memory/system prompts
were held fixed; this experiment did not isolate their contribution to behavior.

## Data, exposure and evaluation boundaries

The allowed pools preserve 8,321 original episodes, 400 WEB training episodes and
145 native EGO training episodes. WEB heldout 100, EGO heldout 181 and original
validation 128 remain excluded from RL. These are reused historical evaluation
cohorts, not newly untouched tests.

The final phase selects 64 episodes per source without within-phase repeats.
All native EGO training episodes had already been sampled by step 128, so its
64 selections are explicitly accepted reuses. Original and WEB episodes are new
to this RL schedule; this does not claim that every original source video or SFT
training example is unseen. ESTP-IT adds 64 distinct training source videos that
were absent from the parent training phase. Exact selected identities are saved.

IT source-video splits remain 932 train / 117 development / 117 test, after
protected evaluation exclusions. This is a project-defined split. Training uses
eligible standing single questions with complete physical clip bounds, nonempty
answers, valid positive-width windows and reachable targets on the fixed 1-FPS
grid. It does not move the grid using gold annotations or widen answer windows.
Physical 2-FPS indices and clip-relative timestamps remain separate. Dynamic
questions and malformed records are rejected, with counts/task types in the audit.
The filtered IT sample emphasizes explicit visual tasks and does not establish
contextual or multi-question ESTP training.

The phase adds 192,656 decisions, reaching 712,452 cumulative decisions. Its
source totals are 21,444 original, 4,620 WEB, 92,416 native EGO and 74,176 IT
decisions (166,592 combined EGO/IT decisions).
The frozen schedule retains the entire parent prefix and replay cursor.

The 1,230-case development suite contains full WEB 500, full EGO 326, original
train/validation probes 128 each, selected perfect20, and fixed IT dev 128.
The same cases and judge rubric are used at steps 128/144/160. Global PAUC-F pools
precision counts and gold-weighted temporal recall before taking the harmonic
mean; it is not the mean of per-episode F scores. Mean episode training reward
and best-of-four diagnostics are different quantities.

`sampled` labels refer to the complete cumulative schedule through step 160;
`phase_sampled` labels refer to the complete planned 32-update phase. At baseline
128 or intermediate 144, some planned phase episodes have not yet been sampled.
`phase_unsampled` episodes may have been seen in the parent run. Use the schedule
prefix and committed batch reports for actual exposure at an intermediate step.
Full WEB/EGO metrics include RL training examples; the heldout rows are the
generalization check.

IT test 128 and all 1,212 ESTP-Bench single-question cases are reserved for the
earlier run's 2,570-case endpoint suite, comparing steps 32 and 128. Their official
ESTP-F1 uses its separate GPT-4.1 rubric and absolute native timestamps. Native
zero-width/out-of-clip benchmark targets remain unchanged; inference stops at
the final physical frame where media are shorter. IT PAUC-F is an explicitly
documented training/development proxy, not official ESTP-F1. No step-160 official
ESTP-Bench result is claimed by the completed development evaluation.

## Correctness, resources and completion

Train only on allocation 38177 / ip-10-3-120-167: seven learners and one frozen
generator. Evaluate the balanced continuation on allocation 38226 /
ip-10-3-70-177 using its eight GPUs. No other nodes were allocated. The earlier
endpoint recovery uses the original training node after training releases GPUs.
GPT judging uses the user-authorized gateway.

The controller runtime is PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 90100,
Transformers 5.5.0 / PEFT 0.18.1. The generator sidecar retains PyTorch
2.10.0+cu128 / Transformers 5.5.3 / PEFT 0.18.1 / Triton 3.6.0. Clear inherited
LD_LIBRARY_PATH to preserve the validated controller numerical libraries.

Temporary files, Triton, TorchInductor, CUDA and extension caches use each node's
`/opt/dlami/nvme/kranasinghe/pace-clm/node-cache` through the existing NVMe wrapper.
Checkpoints, logs, manifests, judge caches, predictions and results stay on FSx.
The earlier endpoint was interrupted by a full root disk; its existing jobs,
prediction shards and judgments are preserved and resumed with NVMe caches.

Thirteen focused CPU tests passed. Exact step-128 and step-129 restores passed
on all seven learners with zero extra optimizer updates. Cross-node replay of
the first collection batches passed. Across all 32 completed updates, maximum
rollout/replay probability error was zero and maximum reserved learner memory
was 74.67 GiB, below the unchanged 115-GiB gate. The continuation has no recovery
log. Quality/coverage/silence values did not stop or shorten the fixed budget.
Unchanged numerical parity tolerances were maximum probability error 0.005,
maximum KL 1e-4, mean KL 1e-5 and selected-action log-probability error 0.05.

Training completed step 160 at 2026-11-16 09:01 UTC. All three development
evaluations completed by 2026-11-16 11:14:58 UTC. Successful update time was
13.45 node-hours / 107.57 allocated GPU-hours, excluding loading, restore gates,
retries, idle time and evaluation. The overall workflow status remains pending
while the separate older endpoint suite finishes; this does not mean step-160
training or its development evaluation is unfinished.

## Results

### Global judged PAUC-F learning curve

| Checkpoint | Full WEB (500) | Full EGO (326) | WEB heldout (100) | EGO heldout (181) | Original validation (128) | IT dev PAUC-F proxy (128) |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 54.688 | 42.726 | 55.727 | 42.484 | 63.901 | — |
| 32 | 56.604 | 42.305 | 55.875 | 41.476 | 66.919 | 10.324 |
| 48 | 57.276 | 42.572 | 57.510 | 41.706 | 67.402 | 9.995 |
| 64 | 57.939 | 42.431 | 57.694 | 41.545 | 67.002 | 9.595 |
| 80 | 58.200 | 42.384 | 57.909 | 41.428 | 67.034 | 8.636 |
| 96 | 57.813 | 43.057 | 57.601 | 43.170 | 69.638 | 10.739 |
| 112 | 58.061 | 43.236 | 57.813 | 42.435 | 68.863 | 9.913 |
| 128 | 57.974 | 43.850 | 57.325 | 43.160 | 69.695 | 10.220 |
| 144 | 58.182 | 44.114 | 57.708 | 43.519 | 70.164 | 10.413 |
| 160 | 58.426 | 44.223 | 58.032 | 43.176 | 70.197 | 10.730 |

### Step 160 against SFT and step 128

| Cohort | SFT | Step 128 | Step 160 | Gain vs SFT | Paired 95% CI vs SFT | Gain vs 128 | Paired 95% CI vs 128 |
|---|---:|---:|---:|---:|---|---:|---|
| Full WEB (500) | 54.688 | 57.974 | 58.426 | +3.738 | [+2.574, +4.915] | +0.452 | [+0.031, +0.875] |
| Full EGO (326) | 42.726 | 43.850 | 44.223 | +1.497 | [+0.331, +2.719] | +0.373 | [-0.472, +1.220] |
| WEB heldout (100) | 55.727 | 57.325 | 58.032 | +2.305 | [+0.071, +4.510] | +0.706 | [-0.456, +2.008] |
| EGO heldout (181) | 42.484 | 43.160 | 43.176 | +0.692 | [-0.832, +2.271] | +0.015 | [-1.041, +1.050] |
| Original validation (128) | 63.901 | 69.695 | 70.197 | +6.296 | [+3.879, +8.909] | +0.503 | [-0.883, +2.013] |
| IT dev PAUC-F proxy (128) | — | 10.220 | 10.730 | — | — | +0.510 | [-0.248, +1.227] |

Intervals use 5,000 paired episode-bootstrap replicates, conditional on saved judgments and selected/reused cohorts. They do not include judge or training-seed uncertainty. No matched SFT IT-dev evaluation is available in this comparison.

### Precision, recall and firing tradeoffs: step 128 → 160

| Cohort | Fire precision (%) | Temporal recall (%) | Useful fires | Wasted fires |
|---|---:|---:|---:|---:|
| Full WEB (500) | 71.03 → 73.54 | 48.97 → 48.47 | 863 → 842 | 352 → 303 |
| Full EGO (326) | 51.45 → 54.90 | 38.21 → 37.02 | 516 → 482 | 487 → 396 |
| WEB heldout (100) | 72.53 → 76.39 | 47.39 → 46.79 | 169 → 165 | 64 → 51 |
| EGO heldout (181) | 48.94 → 51.49 | 38.60 → 37.17 | 300 → 276 | 313 → 260 |
| Original validation (128) | 69.01 → 68.58 | 70.39 → 71.90 | 314 → 323 | 141 → 148 |
| IT dev PAUC-F proxy (128) | 6.29 → 6.68 | 27.32 → 27.25 | 33 → 32 | 492 → 447 |

Every cohort, including training-pool and planned phase-sampled/unsampled subdivisions, is retained in the [generated continuation table](jevany_2b_balanced_to160_results.md) and original result JSON. The tables above select the main generalization and headline cohorts.


## Official ESTP-Bench checkpoint evaluation

The full step-32 RL checkpoint reference has completed all 1,212 ESTP-Bench SQ
cases and all twelve released SQ task categories. Its official EyeWO-protocol
task-macro ESTP-F1 is **30.6418 / 100**, with GPT-4.1 answer judging. Explicit-task
aggregate is 33.2914 and implicit-task aggregate is 25.3426. This single-question
suite has no reported contextual aggregate. These are ESTP-F1 values, not PAUC-F.

Step 32 predates the addition of ESTP-IT training, so it is an RL transfer
reference rather than evidence of benefit from ESTP-IT tuning. Its complete
result is in
[`final-reference-step32/result.json`](../../outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128/evaluation/final-reference-step32/result.json).

As checked on 2026-11-16 at 15:08 UTC, the step-128 endpoint suite is still in
progress (1,575 of 2,570 overall suite cases inferred; this is not a count of
completed ESTP-Bench judgments). Step 160 has completed IT-dev PAUC-F evaluation,
but has no official ESTP-Bench evaluation in the current completed comparison.
No matched SFT ESTP-Bench result is established by these endpoint artifacts.

## Interpretation and limitations

An additional user-authorized GPT-only rescore of the saved explicit ESTP streams
uses the unchanged stream-gemma `estp-pauc-f-v1` protocol. Step 32 scored
**25.1553 explicit task-macro PAUC-F** at omega 0.5 over all 851 explicit cases,
with 71 zero-duration gold turns excluded as prescribed by that separate PAUC
protocol. It preserves all 3,635 emitted fires and absolute timestamps. The
step-128 matching PAUC pass waits for the existing endpoint result; no new model
inference is authorized or required. See the
[explicit PAUC-F report and per-task breakdown](jevany_2b_estp_explicit_pauc_results.md).
These PAUC-F scores are separate from official ESTP-F1; its 1–5 judgments were
not rescaled or reused as PAUC's cumulative-answer judgments.

Step 160 improves over the cohort-matched saved SFT baseline on WEB and original
validation. The EGO heldout gain versus SFT remains uncertain. Most total gains
precede step 128. The balanced continuation adds only modest full-WEB/full-EGO
gains; EGO heldout is essentially unchanged versus step 128. Its EGO behavior
becomes more selective, with fewer useful as well as wasted fires and lower
temporal recall. Precision gains alone are not evidence of improved coverage.

This is one seed and one continuation arm. Changing the episode mixture and
configured source weights together does not isolate their individual effects,
and this run does not compare four versus sixteen trajectories. Bootstrap
intervals condition on saved judgments and reused episode selection; they do
not include judge uncertainty or seed-to-seed variance. Historical full-WEB
54.67 remains an audit reference; the cohort-matched saved SFT value below is
54.687766. Full-set improvements are not an untouched-test quality-floor claim.

## Artifact index and reproduction

The balanced run directory is
[`mixed-web-ego-estp-balanced-v4-to160`](../../outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/).

| Evidence | Artifact within that run |
|---|---|
| Complete CLI defaults, optimizer settings, package versions and hashes | `freeze.json` |
| Phase policy, nodes, schedule and source weights | `workflow-freeze.json` |
| Prepared input/code hashes and source checkpoint commit hash | `prepared.json` |
| Episode/source-video identities, filtering counts, timestamps and exposure | `data/episodes.jsonl`, `data/audit.json`, `data/schedule.json`, `data/video-hashes.jsonl`, `data/estp-video-splits.json` |
| Planned phase/cumulative membership and interpretation | `data/phase-sampled-identities.json`, `data/sampled-identities.json`, `data/cohorts.json`, `cohort-interpretation.json` |
| Frozen executable snapshots | `code/run_jevany_estp_rloo.py`, `code/eval_balanced160_queue.py`, `code/probe_estp_parity.py` |
| Per-update actual observations/actions/rewards, gradients, parity, timings and GPU memory | `batch-000128` through `batch-000159`, including `rollouts.json`, `scored.json`, `report.json` |
| Per-update model/optimizer/RNG/sampler checkpoints and hashes | `step-000129` through `step-000160`, including `commit.json`, `training_state.pt`, `rng.rank*.pt`, `rl_state.json` |
| CPU, restore and cross-node gates | `cpu-validation.json`, `resume-step128-verification.json`, `resume-step129-verification.json`, `cross-node-parity.json`, `continuation-gate-passed.json` |
| Complete predictions, action traces, judgments, cohort metrics and paired intervals | `evaluation/baseline-step128`, `evaluation/full-step144`, `evaluation/full-step160` |
| Completion and live orchestration state | `training-complete.json`, `completion-documentation.json`, `workflow-status.json`, `evaluation-status.json` |

The SFT comparison uses the original checkpoint above and its preserved
[`baseline-sft/result.json`](../../outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/evaluation/baseline-sft/result.json).
Earlier step-32 through step-112 evaluations remain in
[`mixed-web-ego-estp-episode-balanced-v3-to128/evaluation`](../../outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128/evaluation/).
That run also contains the pending official endpoint jobs and their recovery logs.

Entry points are `scripts/run_jevany_balanced160.sh prepare|train|eval`,
`scripts/prepare_jevany_balanced160.py`, `scripts/run_jevany_balanced160.py`,
`scripts/jevany_balanced160_control.py`, and `src/pace_clm/rl_balanced160.py`.
Run node-specific modes through `srun` in their assigned allocations with
`--cpu-bind=none`. Existing completed run directories must be inspected and
resumed through their frozen contracts; do not overwrite them for a new experiment.
