# Next Qwen3.5-2B experiment: stabilize the data mixture as training scales

Date: 2026-11-05. Status: proposed; no training launched. Updated for two
parallel nodes and full WEB evaluation every 100K training presentations.

Completion update (2026-11-06): the superseding two-node mixture study has
completed under user-amended fixed step budgets, with all eight full WEB
evaluations judged. See the
[completed results](../experiments/qwen35_2b_mixture_12h_results.md).
The proposed schedules and "next experiment" wording below are historical;
they are not the executed budgets or a new launch authorization.

Superseded schedule: the user subsequently selected **12 training hours per
arm, plus four WEB evaluations after 3/6/9/12 training hours**. Follow the
[current two-experiment specification](qwen35_2b_12h_mixture_experiments.md).
The fixed-dose and 100K schedules below are retained as earlier planning only.

This is the immediate next experiment, narrowing the broader
[controlled scaling plan](qwen35_2b_controlled_scaling_plan.md). Defer its
batch-size sweep, fresh-versus-replay branches, and LR interventions. First
test whether representative class exposure prevents quality deterioration
as the training dose increases.

## Question and evidence

Does distributing the natural Silence/Response/Hold mixture throughout
training improve closed-loop quality over the existing data ordering?

The full manifest is 81.96% Silence, 15.64% Response, 2.40% Hold. The earlier
successful training manifest was already similar: 81.78%, 15.63%, 2.59%.
However, the stopped run's first 426,464 presentations were 63.61%, 25.50%,
10.90%. Every rolling 20-update window through step 505 contained 8.44–13.32%
Hold. This suggests fixing exposure scheduling before changing the overall
objective or class weights.

On the identical 500-video WEB protocol, the earlier final controller used
3.00% Hold and scored 54.67 PAUC-F; step 505 used 30.81% Hold and scored
47.12. Step 25 produced 1,006 fires versus the earlier final's 991, but only
321 useful fires versus 701. Thus recovering useful event selection matters
more than merely matching predicted action frequencies. These observations
motivate the experiment; they do not establish causality.

Sources: [consolidated findings](../experiments/qwen35_2b_full_data_findings.md)
and [recomputed class/exposure diagnostic](../../outputs/jevany-qwen35-2b-full/reports/web_class_mix_diagnostic.json).

## Two matched training arms

Build one frozen approximately 1.055M-row subset from the full J1-S manifest,
then train two fresh controllers on exactly those rows:

| Arm | Ordering | Purpose |
| --- | --- | --- |
| Control | Existing source/episode order, then existing bounded length grouping | Measure the current scheduling behavior |
| Proposed | Class-aware interleaving across the subset, with constrained length grouping | Keep exposure representative as training progresses |

Use 1,055,040 presentations, four times the earlier 263,760-presentation run.
Freeze the subset before generating either order. Sample across the entire
eligible full manifest, approximately preserving its natural action and
source/context/history distributions. Do not take the first million rows.
Reserve development/confirmation videos and exclude all known benchmark and
validation overlap before sampling. If exclusions change the eligible pool's
class mix, document that and use its audited natural proportions.

Both arms retain exactly the same row multiset, including existing repeated
positions/history variants. Audit position repetition, exact-input repetition,
source/task mix, video concentration and history coverage, but do not introduce
deduplication, new per-video caps or new negative selection in this first
comparison. Those would change which data are learned, beyond its ordering.

Use global batch 128 for both arms: 16 examples per rank per update across
eight GPUs, with physical chunks reduced as needed. Use these fixed doses:

| Dose | Presentations | Optimizer updates | Historical exposure multiple |
| --- | ---: | ---: | ---: |
| Early scale | 263,808 | 2,061 | approximately 1× |
| Intermediate scale | 527,616 | 4,122 | approximately 2× |
| Final | 1,055,040 | 8,243 | exactly 4× |

The stages add 2,061 updates, then 2,061, then 4,121. All updates are full except the final
64-example update. Dose checkpoints are aligned to full batches; do not insert
extra partial optimizer updates merely to hit historical presentation counts.

Batch 128 is a candidate throughput compromise, held constant here; this
experiment does not establish its quality equivalence to batch 32. A/B quality
differences can be attributed to the scheduling intervention under this recipe.
Comparisons to historical 54.67 still include optimization/data differences.
Do not add a batch sweep to this experiment. If a bounded preflight finds batch
128 unusable, freeze one smaller batch for both arms and regenerate the dose
table before launch.

Start both arms from the same saved fresh official 2B initialization, including
identical initial LoRA and pointer weights for seed 42. Keep pointer LR 1e-4,
LoRA LR 2e-6, AdamW defaults, clipping 1.0, no new scheduler, and the established
model/input/history/generator contracts. Do not resume step 505: a fresh start
tests exposure during initial learning as well as later scaling.

## Implement representative exposure

Use deterministic class queues over the **whole frozen subset**, interleaving
videos, sources, contexts and memory states within each class. Every row is
consumed once; no replacement sampling or class-dependent loss weights.

Construct 4,096-row planning windows (32 global updates at batch 128). Allocate
class counts by cumulative fractional quotas, carrying rounding residuals
forward. At the full pool's proportions, a window averages approximately
3,357 Silence, 641 Response, and 98 Hold rows. Exact counts come from the frozen
subset proportions, not these rounded examples.

Distribute each window's class quotas across updates rather than exhausting
one class at a time. A typical update averages roughly 105 Silence, 20 Response
and three Hold. Prefer near-natural update counts when feasible, but allow
small deviations to preserve context and frame-cache locality. Do not require
each GPU or physical chunk to contain all three classes.

Choose comparable context lengths from the class queues when filling updates.
Balance short/long contexts and memory/history strata across planning windows;
do not globally sort all rows by length afterward. The existing length sorter
may be reused only within constraints that retain the exposure audits. Shuffle
video/episode groups within classes using a frozen seed; preserve each row's
gold causal history and do not use earlier sampled rows to construct its input.

Audit the final emitted order, after every grouping operation:

- Every full 4,096-row window: each class within one percentage point of the
  target; Hold within 0.5 points. Report the final partial window separately.
- Cumulative prefixes at 1%, 5%, 25%, 50%, 100%, and all evaluation doses:
  the same limits, allowing unavoidable integer rounding.
- Report per-update class ranges and longest runs without Response/Hold,
  plus per-rank distributions, context/memory/source distributions and padding.
- Verify exact row-multiset equality between arms, no loss or duplication,
  causal-input preservation, held-out exclusion and deterministic checksums.

Report the control's actual skew before training. If it does not reproduce
meaningfully biased exposure on the selected subset, label the experiment an
ordering comparison with weak class-skew contrast. Do not manufacture a bad
control or claim it tests the previously observed 10.9% Hold exposure.

If the constrained sampler cannot satisfy coverage and acceptable padding,
revise window size or queue selection before launch, then freeze both manifests.
Never solve feasibility by duplicating rare-class rows silently.

## GPU execution

Keep CPU preparation prefetch, cross-episode frame caching, parallel decode,
packed gradient reduction, expandable allocation and established kernels.
Profile both orders on the same node/backend with representative short and
long contexts, warmup and at least 100 measured updates each. Record examples/s,
tokens/s, padding, cache hit rate, preparation/synchronization wait, utilization,
peak memory and GPU-hours. Retain at least 10% measured peak-memory headroom.

The sampler may trade some frame-cache locality or padding efficiency for
representative batches. Measure that cost explicitly; do not compensate by
giving one arm a different global batch or fewer frames. The prior CPU-prefetch
81% utilization benchmark does not predict this experiment's utilization.

## Two-node execution, evaluation cadence and time budget

Assign one eight-GPU node to control and the other to the proposed sampler.
Run the same seed and fresh initialization on both. Verify model revisions,
dependencies, kernels, hardware, loss normalization and collective backend
before launch. The prior hosts differed in NCCL functionality; use a common
validated backend for both arms if possible. If backends differ, record that
additional implementation difference and avoid attributing every quality
difference exclusively to scheduling. Compare quality by presentations seen,
not wall-clock time; representative ordering may have a different cost.

On each node, save a complete checkpoint at the first optimizer boundary
after each cumulative 100K threshold, pause/release training GPU allocations,
run all 500 WEB videos with that immutable checkpoint and the promoted MoE,
then resume the exact training state. These pauses must not restart data order,
optimizer state, RNG state or the sampler. Do not try to fit eight-rank training
and eight-rank WEB inference onto the same occupied GPUs concurrently.

At batch 128, the first checkpoint is update 782 / 100,096 presentations;
the next are update 1,563 / 200,064, update 2,344 / 300,032 and update 3,125 /
400,000. Use cumulative thresholds, not 100K measured from the last rounded
checkpoint. Alignment overshoot is at most 127 presentations. Final training
remains 1,055,040 presentations / 8,243 updates.

Evaluate at 100K, 200K, ..., 1,000K and the final 1,055,040 dose: **11 full
WEB evaluations per arm, 22 across the pair, 11,000 video evaluations total**.
Keep the earlier approximately 1×/2×/4× checkpoints for development analysis;
their additional development-evaluation cost is outside the WEB budget below.

Measured historical wall times, from the last rollout-start marker to the
fully judged completion marker (including startup and scoring):

| Prior WEB evaluation | GPUs | Wall time |
| --- | ---: | ---: |
| Full-data step 505 | 8 | 30 min 11 s |
| Full-data step 10 | 4, alongside step 25 | 25 min 45 s |
| Full-data step 25 | 4, alongside step 10 | 31 min 47 s |

Sources: [step-505 rollout log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_eval.log)
and [completion log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_score.log),
with corresponding step-10/25 logs in that directory. Startup/cache state,
response count, rank load and judge latency differ, so these are not GPU-count
speed comparisons. Budget **30–45 minutes per full WEB evaluation**, plus
checkpoint/training-reload overhead; judge service delays can exceed this.
Eleven evaluations add approximately **5.5–8.25 hours per arm**. The two nodes
run concurrently, so paired elapsed time follows the slower arm rather than
the sum. GPU-free judging can overlap resumed training once rollouts/merge
finish, with results tagged to their checkpoint; the conservative budget above
waits for the judged result before resuming.

Training time must be profiled on the actual node and ordering. Existing
measurements bracket useful planning scenarios, not a guaranteed ETA:

| Training rate | Time per 100K | Time for 1.055M + 11 WEB evaluations |
| --- | ---: | ---: |
| 4.28 presentations/s | 6.49 h | 74.0–76.7 h |
| 8 presentations/s | 3.47 h | 42.1–44.9 h |
| 12.32 presentations/s | 2.25 h | 29.3–32.0 h |

The 4.28/s rate comes from an older, pre-prefetch global-batch-128 benchmark
with substantial padding ([raw report](../../outputs/jevany-qwen35-2b-util/reports/nccl_b16_throughput.json)).
The 12.32/s rate comes from the newer 20-step mixed-plan/Gloo training test,
not fixed batch 128. The 8/s row is an illustrative planning assumption.
The proposed sampler, full context distribution and node/backend pairing have
not been timed, so these rates are not bounds. Do not use the fixed-batch-48
19.61/s benchmark as the new run's promised rate. Measure both arms before
launch and publish separate node-specific ETAs, updating after the first 100K.
These totals exclude additional development evaluations, initial preflight,
checkpoint/reload overhead and failures.

## Evaluation and success criteria

Freeze a video-disjoint development cohort before training, proposed size 500
subject to eligibility and adapter audit. Use the same closed-loop generator,
margin 0, 96-token cap and GPT-4.1 PAUC-F protocol in both arms. Evaluate the
three predetermined development dose checkpoints, plus full WEB at every
100K threshold as specified above. Persist timestamped
action logits, selected actions and memory/history status; align them with
gold event windows and judged response outcomes.

Track judged PAUC-F, precision, PAUC recall, useful/wasted fires, repeats,
action rates and misses within gold windows. Separate Hold-heavy misses from
fires with semantically unsuccessful answers when the traces support it.
Teacher-forced CE/confusion and action rates are diagnostics, not selection
objectives. Do not force predicted Hold toward 3%: that is a comparator's
behavior, not a ground-truth WEB class prior.

Primary comparisons are proposed-minus-control PAUC-F at each matched dose
and the 1×-to-4× quality change within each arm. Compute 1,000 paired-video
bootstrap replicates, including a paired difference of those changes. Report
inconclusive intervals explicitly. A mixture correction is promising if it
improves useful-fire selection over control and its final score retains the
early-scale score within a proposed one-point noninferiority tolerance.
A lower Hold rate alone is not success.

Finish both predetermined final checkpoints unless there is a runtime or
data-integrity failure. Report every scheduled WEB result, including the
finals, and avoid checkpoint/margin selection on WEB. Treat the frequent WEB
curve as monitored benchmark evidence, not an untouched final test; keep recipe
changes and selection on development/confirmation data. If development
supports the intervention, repeat the paired experiment with seeds 43/44 and
use frozen confirmation data before a larger rollout. Report seed variation
separately from paired-video uncertainty.

If both arms deteriorate, report that scheduling was insufficient under this
recipe. Then investigate optimization or a separate data-content intervention
(history quality, redundant exposure or negative selection) using the broader
plan. If control and proposed perform similarly, first check whether their
actual class exposure differed before concluding mixture does not matter.

## Concrete deliverables

Implement a deterministic class-aware manifest planner and its audits, matching
control/proposed batch plans, exact checkpoint-dose support, the two-node
100K checkpoint/evaluate/resume orchestration, per-decision
development traces and a paired results report. Test row coverage, class quota
rounding, final partial batches, deterministic ordering, causal input identity
and checkpoint/resume continuity. Existing training code can retain its model
and optimizer logic; the new behavior belongs primarily in offline planning.

Suggested output root: `outputs/jevany-qwen35-2b-data-mixture/`. Preserve original
manifests/checkpoints. Save source hashes, cohort IDs, both emitted orders,
coverage/mixture audits, recipes, profiles, checkpoints and judged predictions.
Implement and audit the planner first; launch neither training arm until its
concrete manifests, class-exposure contrast and cost estimate are reviewable.
