# JevAny Qwen3.5-2B RL E1: completed pilot and WEB results

Recorded: 2026-11-08. Training and both full WEB evaluations completed on
2026-11-07 at 23:12:05 UTC. Status: **complete; neither evaluated RL checkpoint
improved PAUC-F over the strong SFT starting checkpoint**. No follow-up RL run
has been launched as part of this report.

The [implementation and execution plan](../plans/jevany_2b_rl_eight_h200_plan.md)
contains the engineering history. This report records the completed experiment,
its actual objective, sample exposure, results and limitations.

## WEB result

Both evaluations covered all **500 unique WEB videos / 1,328 gold turns**.
They used the original SFT evaluator, greedy controller with Response margin
0, actual generated history, 32 causal frames, last-response v2 memory, and
the frozen promoted MoE answer generator with a 96-token cap. GPT-4.1 scoring
used **`pauc-f-v1`, omega 0.5**, not a timing-only proxy. Prediction IDs, judge
response coverage and scorer settings passed the workflow's completeness checks.

The step-16 and step-32 protocol manifests each match the stored SFT manifest
in every field except `controller_adapter_sha256` and `pointer_head_sha256`.
The SFT reference below is the existing completed evaluation, not a newly
generated baseline or an independent repeated judge measurement.

| Metric | SFT step 8,243 | RL step 16 | RL step 32 |
|---|---:|---:|---:|
| GPT-scored PAUC-F | **54.6651** | 53.3141 | 53.0861 |
| Difference from SFT, points | — | -1.3510 | -1.5791 |
| Fire precision | 70.74% | **79.66%** | 76.94% |
| PAUC recall component | **44.54** | 40.06 | 40.52 |
| Total fires | 991 | 649 | 707 |
| Useful fires | **701** | 517 | 544 |
| Wasted fires | 290 | **132** | 163 |

RL produced fewer wasted responses and higher fire precision, but lost useful
coverage. At step 32 it had 127 fewer wasted fires and 157 fewer useful fires
than SFT. The resulting PAUC-F remained 1.58 points below SFT. Step 32 recovered
some recall relative to step 16, while losing precision and another 0.23 PAUC-F
points. Neither evaluated RL checkpoint is supported as a PAUC-F improvement
over SFT by this pilot.

These are observed point estimates from one RL seed. No paired confidence
interval or repeated GPT judging was computed for these differences. The
result does not isolate whether reward design, judge mismatch, exploration,
credit assignment or training exposure caused the regression. WEB was already
inspected before this run and must not be described as an untouched test set.

## Starting point and trainable parameters

- Initial policy: strong 2B J1-S SFT checkpoint
  `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
- Base: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Trainable: decoder LoRA across all 24 layers, r64 / alpha128,
  **62,521,344 parameters**, plus **1,049,088** q/k pointer-head parameters.
- Frozen: base weights, vision, SFT reference, and promoted answer generator
  `../stream-gemma/outputs/lora-qwen-w1-long-neg` with its pinned
  `Qwen3.5-35B-A3B` base revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Actions: Silence, Hold/Standby, Response. Controller input remains 1 FPS,
  384px, 144 visual tokens/frame, up to 32 causal frames and J1 history.
- LoRA dropout is disabled consistently for rollout/replay likelihood parity;
  the saved SFT adapter's FP32 arithmetic is preserved.
- AdamW: pointer LR `1e-5`, LoRA LR `1e-6`, weight decay 0.01, global gradient
  clipping 1.0. RL sampling seed: 42.

## Actual data exposure

| Quantity | Completed exposure |
|---|---:|
| Prepared training-span pool | 4,096 spans |
| Distinct span starts consumed | **1,024** |
| Distinct source videos consumed | **911** |
| Rollouts per start | 4 |
| Sampled trajectories | **4,096** |
| Decisions per trajectory | 32 |
| Total sampled action decisions | **131,072** |
| Rollout batches / optimizer updates | **32 / 32** |

Each update used **32 starts × 4 rollouts × 32 decisions = 4,096 decisions**.
The run consumed the first 1,024 distinct starts from the frozen manifest,
without revisiting those span IDs in later updates. Distinct spans can overlap
in video/time; the 131,072 decisions are correlated trajectory observations,
not 131,072 independent examples or unique video frames.

Consumed starts comprised **534 single-event, 253 multiple-event and 237
no-event spans**. They came from the existing 11,899-video SFT cohort; the data
builder excluded 445 source records matching known evaluation identities.
WEB was not used as RL training data. The full pool was balanced at 2,048
single-event / 1,024 multiple-event / 1,024 no-event spans, so this pilot used
**25% of the prepared starts**, not the entire pool or cohort.

The corrected SFT-start correctness gate on November 6 supplied step 1.
The November 7 continuation supplied steps 2–32; these are 32 total updates,
not 32 updates added after the gate. Earlier engineering profiles are separate
and are not included in this policy's exposure.

## Objective actually used: RLOO-style policy gradient

**This was not standard clipped GRPO.** Earlier conversational use of “GRPO”
was imprecise. The code implements on-policy whole-trajectory REINFORCE with
a leave-one-out group baseline, closest to
[RLOO](https://aclanthology.org/2024.acl-long.662/). Reusing GRPO span/judge
utilities does not change the optimizer's identity.

For each group of four trajectories:

```text
A_i = R_i - mean(R_j for the other three trajectories)

L = mean over trajectories and their 32 actions of:
    -A_i * log pi(action_i,t | observation_i,t)
    + 0.02 * KL(pi_current || pi_SFT)
    - 0.001 * entropy(pi_current)
```

Every sampled action in a trajectory receives that trajectory's same detached
advantage. Gradients pass through the current decoder LoRA and pointer head;
no gradient passes through the generated answer, sampled history or judge.
There is **one optimizer step per fresh rollout batch**, no group-standard-
deviation normalization, no clipped old/current probability-ratio surrogate,
and no multiple optimizer passes over a rollout batch.

[Standard GRPO](https://arxiv.org/html/2402.03300v3) uses a clipped policy-ratio
objective and group-normalized advantages. With our single accumulated update
per fresh batch, a newly introduced PPO-style ratio would initially be 1;
clipping becomes relevant when parameters change across further updates on
the same collected batch. Implementing that behavior requires an actual loss
and update-schedule change, including appropriate old-policy bookkeeping.

### Training reward was a proxy, not PAUC-F

The chronological event ledger used complete four-second annotation windows:

```text
D = max(1, number of gold events in the span)
R = [sum(matched semantic score * max(0, 1 - delay_seconds/4))
     - 0.50 * missed events
     - 0.25 * unmatched fires
     - 0.35 * duplicate fires] / D
```

A match requires semantic score at least 0.5. Matching is one-to-one and gives
eligible distinct recurring events priority over repetition penalties.
Duplicate and unmatched penalties are mutually exclusive; silence has no
positive bonus. A wholly silent event span incurs missed-event penalties.

Semantic/repetition scores came from the **local frozen Qwen MoE base with
the response adapter disabled**, backend
`jevany-offline-judge-v8-native-grouped-mm`, with constrained scores
`0, 0.25, 0.5, 0.75, 1`. **GPT-4.1 was used for WEB evaluation, not RL rewards.**
Calibration and cross-runtime consistency checks during setup do not establish
independent agreement with GPT or prove reward alignment with WEB PAUC-F.

PAUC-F instead combines useful-fire precision with the official timing/semantic
AUC using their harmonic mean. Our linear delay reward, explicit penalties,
matching threshold and per-span normalization are not that metric. The
observed precision/coverage tradeoff is consistent with a possible reward
mismatch, but is not causal proof of one.

## Learning-signal diagnostics

Derived from the saved scored trajectories and per-update learner reports:

| Diagnostic | Early run | Late run |
|---|---:|---:|
| Four-rollout groups with identical rewards | 99/256 = 38.7% (updates 1–8) | 140/256 = 54.7% (25–32) |
| Event-containing groups with identical rewards | 67/202 = 33.2% (1–8) | 95/195 = 48.7% (25–32) |
| Mean categorical entropy | 0.05757 (step 1) | 0.02815 (step 32) |
| Mean KL(current policy || SFT) | 0 (step 1) | 0.03374 (step 32) |

Identical rewards make all four reward-based leave-one-out advantages zero;
the KL and entropy terms can still contribute gradients. This is expected for
some no-event spans, but also occurred in almost half of late event-containing
groups. Lower entropy and more identical-reward groups warrant an exploration
audit. They do not prove that every equal-reward group had identical actions
or that insufficient exploration alone explains the benchmark result. Standard
GRPO also has zero reward-based group advantages when all rewards are equal.

## Execution, efficiency and validation

The full continuation and both scheduled WEB evaluations used **only the eight
H200s on `ip-10-3-120-167`**, allocation 38177 / Slurm step 38177.1. Training used
seven controller/learners plus one resident frozen generator/local judge.
Full WEB evaluation used eight local ranks, each with its controller and frozen
answer generator. Training paused for the step-16 evaluation/scoring and
resumed automatically; step-32 evaluation/scoring ended the workflow.

The three-warm-update engineering benchmark is retained separately in
`profile-7p1-v9/`; it preceded the startup-only exact-reference-loading fix.
The final run used corrected source and checkpoint lineage in `profile-7p1-v10/`.
Do not use the older benchmark checkpoint as this experiment's starting point.

Actual continuation measurements below cover **updates 2–32**, including the
restart after the step-16 evaluation; initialization and WEB evaluation are
excluded from the update windows:

| Measurement | Observed |
|---|---:|
| Complete-update wall time, sum | 19,575.39s = 5.44h |
| Mean complete-update time | 631.46s = 10.52min |
| Decisions / complete-update second | 6.49 |
| Learner utilization during learning | 92.23% |
| Learner utilization across complete updates | 79.02% |
| Whole-node utilization across complete updates | 70.36% |
| Peak learner allocated / reserved memory | 44.77 / 48.43 GiB |
| Peak learner / generator device memory | 55.23 / 81.83 GiB |
| Collection / replay real-row fraction | 89.89% / 97.37% |

Utilization is averaged from phase-filtered one-second GPU samples. Throughput
depends on the sampled spans and policy firing rate; these are pilot workload
measurements, not a controlled speedup claim over the earlier benchmark.
All 32 complete-update timings sum to 5.64h, including the prior-day step 1.
The November 7 workflow ran from its GPT preflight at 16:19:40 to completion at
23:12:05 UTC, about **6h52m**, including the one-video canary, initialization,
checkpoint reloads and both full WEB evaluations, but excluding the prior-day
correctness gate and earlier engineering profiles.

Across all 32 updates, the maximum recorded rollout/replay log-probability
error was **zero**. LoRA/head gradient and frozen-parameter guards passed.
The reference loader checks all 300 SFT adapter tensors exactly at startup,
avoiding PEFT's BF16 intermediate rounding when adding a second adapter.
Final restore verification passed on all seven learners at **step 32 / sampler
cursor 1,024**, with zero additional optimizer steps. Resume restores saved
model/optimizer/RNG/sampler state; asynchronous BF16 generator batching does
not promise bitwise-identical future answer text.

## Follow-up experiments proposed, not run

Do not infer that RLOO failed generally or that GRPO will necessarily improve
this task from this single pilot. Separate the reward and optimizer questions:

1. Audit the current reward's ranking of training trajectories against
   PAUC-F-based scoring, including a judge-agreement check. Do not assume
   that changing the reward formula alone removes local-judge mismatch.
2. Restart from the same strong SFT checkpoint for a small comparison of
   **RLOO + PAUC-F reward** and **clipped GRPO + PAUC-F reward**, alongside
   this completed RLOO + proxy baseline. Match sampled-data budgets and
   report optimizer passes and compute separately.
3. Scale a promising variant to more distinct starts and longer training on
   held-out development evidence, rather than blindly extending this checkpoint.

Policy gradients can optimize nondifferentiable scalar metrics; see
[self-critical sequence training](https://arxiv.org/abs/1612.00563). A PAUC-F
reward should use training-only annotations and the intended timing/judge
semantics. Mean short-clip PAUC-F is not equal to the benchmark's aggregate
PAUC-F, so episode boundaries and aggregation require an explicit design.
None of these suggestions authorizes using WEB episodes as training data.

Using all 4,096 prepared starts at the current 32 starts/batch and four
rollouts/start would require **128 rollout batches / 524,288 decisions**.
Under the present one-step-per-batch implementation that is 128 optimizer
steps; GRPO with multiple passes would have a different optimizer-step count
and compute budget. More exposure is available, but this run alone does not
establish that longer training under the existing proxy improves PAUC-F.

## Evidence and implementation

Paths in the ignored `outputs/` tree are local experiment artifacts.

| Evidence | Artifact |
|---|---|
| SFT reference result | [report](qwen35_2b_j1s_web.md), [score](../../outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json) |
| Step-16 WEB | [score](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step16/pauc_f_summary-gpt-4.1-v1.json), [predictions](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step16/predictions.jsonl), [protocol](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step16/manifest.json) |
| Step-32 WEB | [score](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step32/pauc_f_summary-gpt-4.1-v1.json), [predictions](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step32/predictions.jsonl), [protocol](../../outputs/jevany-2b-rl/profile-7p1-v10/evaluation/web_step32/manifest.json) |
| Final policy | [step 32](../../outputs/jevany-2b-rl/profile-7p1-v10/step-000032) |
| Frozen configuration and data audit | [freeze](../../outputs/jevany-2b-rl/profile-7p1-v10/freeze.json), [span audit](../../outputs/jevany-2b-rl/data/audit.json), [spans](../../outputs/jevany-2b-rl/data/spans.jsonl) |
| Workflow completion and launch | [status](../../outputs/jevany-2b-rl/profile-7p1-v10/full_run/status.json), [launch plan](../../outputs/jevany-2b-rl/profile-7p1-v10/full_run/launch_plan.json), [launcher log](../../outputs/jevany-2b-rl/profile-7p1-v10/full_run/launcher.log) |
| Resume and initial readiness | [final restore](../../outputs/jevany-2b-rl/profile-7p1-v10/resume_verification.json), [initial readiness](../../outputs/jevany-2b-rl/profile-7p1-v10/readiness.json) |
| Per-update evidence | [first report](../../outputs/jevany-2b-rl/profile-7p1-v10/batch-000000/report.json), [final report](../../outputs/jevany-2b-rl/profile-7p1-v10/batch-000031/report.json), [final scored trajectories](../../outputs/jevany-2b-rl/profile-7p1-v10/batch-000031/scored.json), [GPU samples](../../outputs/jevany-2b-rl/profile-7p1-v10/gpu_samples.csv) |

Implementation: [loss and reward](../../src/pace_clm/rl.py),
[collection/update orchestration](../../scripts/run_jevany_rl.py),
[model workers](../../scripts/jevany_rl_workers.py),
[local judge](../../scripts/jevany_rl_judge.py),
[full-run/evaluation workflow](../../scripts/run_jevany_rl_full_current_node.py).
