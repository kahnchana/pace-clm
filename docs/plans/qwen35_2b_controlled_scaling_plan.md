# Qwen3.5-2B: controlled SFT scaling with fresh-data and replay controls

Date: 2026-11-05. Status: proposed experiment; no training launched.

Follow-up status (2026-11-06): the preliminary mixture study is complete;
see its [results and limitations](../experiments/qwen35_2b_mixture_12h_results.md).
The A/B reference, fresh/replay, LR and multi-seed branches proposed in this
document have not been launched or established as successful. Current evidence
supports bounded diagnosis before committing to unchanged full-manifest scaling.

Scope update: the next experiment now focuses on data mixture and scheduling
first; follow the [dedicated mixture plan](qwen35_2b_data_mixture_plan.md).
The additional batch/replay/LR branches below are deferred follow-up options.

Run a **1× → 2× → 4× data ladder at fixed global batch 128**, paired with a
repeated-data control and a global-batch-32 reference. Start from fresh official
weights. Keep CPU preparation prefetch and safe physical batching. The next
decision is whether roughly one million presentations of more diverse data can
retain or improve judged quality, before committing to all 8.34M presentations.
Batch 128 is a proposed compromise, not a measured quality or utilization winner.

## Evidence and hypotheses

The [latest consolidated findings](../experiments/qwen35_2b_full_data_findings.md)
do not establish that more data hurts this model:

| Factor | Earlier completed run | Stopped full-data run |
| --- | --- | --- |
| Presentations at evaluated checkpoint | 263,760 | 426,464 at step 505 |
| Optimizer updates | 8,243 | 505 |
| Global batch | 32 | 416–2,048, context dependent |
| Overall / exposed Hold fraction | 2.59% overall | 2.40% overall / 10.90% exposed |
| WEB PAUC-F | 54.67 | 47.12 |

The full-data step-10/25/505 scores were 34.89/32.27/47.12. Step 505's
gap to the earlier final checkpoint was -7.55 points, with a paired-video
95% interval of [-9.20, -5.95]. These are checkpoint effects from one training
seed, not an estimate of seed variance. No completed full-data epoch exists.

Prioritize three explanations: nonrepresentative exposure order, changed
optimization from much larger batches/fewer updates, and redundant exposure
rather than useful new coverage. Gold-history training versus self-generated
history at inference remains another possible limitation. Do not change that
contract in the first experiment, because it would confound this scaling test.

An additional audit finding comes directly from the manifest summaries:

| Manifest | Presentations | Distinct video/question/frame positions | Videos |
| --- | ---: | ---: | ---: |
| Earlier J1-S | 263,760 | 213,735 | 11,446 |
| Full J1-S | 8,343,088 | 4,694,565 | 158,218 |

The full source is 31.63× larger by presentations, but 21.96× by positions.
Repeated positions can carry different histories; these counts do **not** prove
identical model inputs. Audit exact inputs and histories separately. Source:
[earlier summary](../../outputs/jevany-j1s/manifests/train.summary.json) and
[full summary](../../outputs/jevany-qwen35-2b-full/manifests/j1s.summary.json).

Separating example and update budgets is also motivated by
[Hoffer et al.](https://arxiv.org/abs/1705.08741), who found update count mattered
in their large-batch vision experiments. That result motivates this control;
it does not establish the cause of our AdamW/LoRA controller behavior.

## WEB-grounded refinement: control class exposure before changing class weights

Follow-up analysis on 2026-11-05 streamed the actual first 505 planned updates
and recomputed action totals from the four merged WEB prediction files. The
machine-readable [diagnostic](../../outputs/jevany-qwen35-2b-full/reports/web_class_mix_diagnostic.json)
contains training-prefix counts, all 505 per-update class counts, official
scores, and per-video action-count comparisons.

| Training exposure | Silence | Response | Hold |
| --- | ---: | ---: | ---: |
| Earlier complete training manifest | 81.78% | 15.63% | 2.59% |
| Full manifest overall | 81.96% | 15.64% | 2.40% |
| Prefix through step 10 | 61.74% | 28.74% | 9.52% |
| Prefix through step 25 | 64.38% | 24.63% | 10.99% |
| Prefix through step 505 | 63.61% | 25.50% | 10.90% |

This is sustained exposure skew, not just a few anomalous batches: every
contiguous 20-update window in the first 505 updates has 8.44–13.32% Hold.
The observed range across individual updates is 0–24.80% Hold. The whole
manifest's class ratio is already close to the successful run's ratio.

| WEB controller | Hold decisions | Response decisions | Useful / wasted fires | PAUC-F |
| --- | ---: | ---: | ---: | ---: |
| Earlier final | 3.00% | 11.26% | 701 / 290 | 54.67 |
| Full step 10 | 64.06% | 3.42% | 136 / 165 | 34.89 |
| Full step 25 | 38.23% | 11.43% | 321 / 685 | 32.27 |
| Full step 505 | 30.81% | 9.80% | 514 / 349 | 47.12 |

All four runs cover the same 500 videos and 8,803 decisions. Step 505 uses
more Hold on 345 videos, less on six, and the same amount on 149. The issue is
widespread across videos. However, training-prefix Hold stays around 10–11%
while predicted Hold falls from 64% to 31% with training. Exposure ratios
cannot be the sole explanation; optimization progress and rollout histories
can affect the learned decision boundary.

Step 25 is particularly informative: its 1,006 fires nearly match the earlier
final's 991, but useful fires fall from 701 to 321 and wasted fires rise from
290 to 685. Step 505 fires 128 fewer times than the earlier final, yet loses
187 useful fires and adds 59 wasted fires. Matching aggregate fire frequency
would not establish recovery of useful event selection or response content.

**Recommendation:** distribute the existing natural mix through training;
do not start with equal thirds, inverse-frequency class weights, or target
WEB prediction proportions. Equal thirds would expose Hold about 13.9× as
often as its full-manifest rate, while our bad checkpoints already overuse
Hold. This is a reason to avoid that untested intervention, not proof of
how an equally sampled model would behave. With unweighted cross-entropy,
oversampling changes classes' contribution to the objective; class weights
are another explicit objective change ([PyTorch loss definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)).

At batch 128, the natural mix averages approximately 105 Silence, 20 Response,
and three Hold examples per **global optimizer update**. Use rotating rounded
quotas or sampling without replacement across a short multi-update window so
rare classes are spread out without changing total row multiplicities. Do not
require the same ratio in each GPU's physical chunk. Allow small-count noise
in individual batches and audit cumulative/rolling exposure. Keep context and
history diversity within each class; three near-identical Hold rows are not
equivalent coverage to three distinct examples.

Add a seed-42 **sampler ablation before the 4N commitment**. After B passes,
fork its complete state into two N-row fresh-data continuations:

- C uses the representative, approximately natural-ratio ordering above.
- S uses those exact same rows in the legacy source order followed by the
  existing bounded length grouping, at the same fixed batch 128.

Keep row multiplicities, labels, LR, optimizer, physical chunk policy and
evaluation identical. Audit each sampler's actual rolling class exposure;
if S has no meaningful skew, it is not a test of the skew hypothesis. Save
and evaluate at quarter-block intervals as well as the endpoint. The final
checkpoint has identical total data and updates, isolating the scheduling
intervention as a whole; it does not separate class ordering from correlated
source/context/history ordering. Profile padding/cache effects in both arms.
The C continuation is reused in the main ladder, so S adds one N-row branch.
This tests mitigation during continued scaling; a null result would not rule
out damage from skew during initial head/LoRA learning.

Use frozen development outcomes to decide: does C improve useful-fire
precision/recall and PAUC-F over S, and does it reduce pathological Hold use
without increasing wasted fires? Treat a causal claim about class ratio alone
as unresolved until source/context/history ordering is also controlled. Do not
force the earlier final's 3% predicted Hold rate: it is a comparator's behavior,
not a WEB ground-truth class prior.

Existing WEB files store per-video action totals, not timestamped Hold/Silence
decisions. They cannot establish which missed gold events were suppressed by
Hold. For new development rollouts, record frame/time, three raw logits,
chosen action, memory status and history provenance. Align those traces with
gold event windows and judged fires to separate missed opportunities, redundant
fires and semantically unsuccessful answers. Keep WEB for the frozen final
comparison; do not select sampler quotas by tuning its observed action ratios.

## Freeze the data and evaluation contracts first

Use `N = 263,760` presentations as one exposure unit, not “10%.” Preserve the
old J1-S manifest as the anchor, checksum
`89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222`.
Do not regenerate its J1-S negatives under a new policy.

Before training, reserve video-disjoint development and confirmation cohorts
from eligible training-source videos outside the anchor. Proposed sizes are
500 development videos and 500 confirmation videos, subject to eligibility
audit. Exclude all known prior validation and benchmark videos from training.
Freeze cohort IDs, coverage strata, and hashes before examining new predictions.
Use development for choices and confirmation only after freezing the recipe.
WEB has already been inspected repeatedly and is a historical comparison,
not a newly untouched test set.

Create three disjoint expansion blocks of N rows from the full J1-S pool,
excluding held-out videos. Prioritize videos absent from the anchor and avoid
reusing a video/question/frame position across expansion blocks or the anchor.
Check video aliases/content duplicates as well as literal IDs. Within each
eligible position, select history variants with a frozen, seeded rule; report
memory/history coverage. This is controlled sampling, not relabeling.

Match each block approximately to the anchor's action, context-length,
memory-present/absent, post-response-negative, and source/task distributions.
Use the anchor mix (81.78% Silence, 15.63% Response, 2.59% Hold), not equal
class weights. Audit feasibility before fixing block sizes; do not silently
fill a “fresh” block by repeating rows if a stratum is short. Report relaxed
strata or a smaller preregistered dose if required. This tests diversity at
roughly fixed mixture; it does not yet test the full manifest's raw weighting.

Generate representative order by seeded interleaving of video/episode groups
across the whole pool **before** bounded length grouping. Preserve each row's
causal frames, gold history, memory, label, and input text. Existing rows are
self-contained; row order must not supply history. Verify that explicitly.
Retain short episode-local groups where useful for frame-cache locality.

Audit cumulative prefixes at 1%, 5%, 25%, 50%, and 100% of every block and
rolling windows of about 8,192 rows. Proposed limits: action proportions within
1 percentage point of block targets, Hold within 0.5 points; context and
memory-bin proportions within 3 points. These are planning constraints, not
claims about optimal class proportions. Fail manifest preparation if ordering
recreates a Hold-heavy prefix. Audit all ranks together and individually.

## The minimum informative experiment

All arms use fresh official 2B initialization, the same initial head/LoRA for a
given seed, and the same frozen anchor order. Use seed 42 for screening, then
43 and 44 for independent confirmation of the selected comparison.

| Arm | Global batch | Data | Evaluations | Purpose |
| --- | ---: | --- | --- | --- |
| A: reference | 32 | Anchor once | N | Reestablish small-batch quality with representative order and the batched engine |
| B: shared anchor | 128 | Anchor once | N | Test batch increase at identical exposure; common starting state for C/R |
| C: fresh scaling | 128 | B, then three fresh blocks | 2N and 4N | Test more diverse data |
| R: replay control | 128 | B, then three new shuffles of the anchor | 2N and 4N | Test extra updates/exposure without new coverage |

C and R fork B's **complete** state: adapter, pointer, optimizer, and all RNG
states. Their continuation lengths, LR policy, batch size, class targets, and
checkpoint doses match. R is an experimental control, not the recommended
production curriculum. Preserve a separate output root for every branch.

At N, A has 8,243 updates; B has 2,061. At 4N = 1,055,040 presentations,
C/R each have 8,244 updates if each N-row block ends in its own partial update.
The one-update difference from A is explicit. Each batch-128 block consists of
2,060 full updates and one 80-example update. Do not pad by duplicating rows.

Interpret the comparisons separately:

- **B versus A at N:** same rows/exposure, different batch and update count.
- **C versus R at 2N/4N:** matched batch, updates and exposure; different data
  diversity. This is the primary scaling comparison.
- **C/R at 4N versus A at N:** nearly equal update budget, with four times the
  exposure. This is a practical recipe comparison, not an isolated batch effect.

First run A and B. If B fails the development quality gate, do not spend on
batch-128 continuations: use batch 32 for the fresh/replay ladder, accepting
the additional updates and measuring its actual cost. If A itself fails to
recover the historical policy, investigate engine/order differences first.
Use the original trainer and original row order for a targeted reproduction
only if needed; do not attribute a failed reproduction to new data.

## Fixed optimization and GPU execution

Keep the established J1-S model contract: official resolved 2B revision,
fresh r64/alpha128/dropout-0.05 LoRA on the existing language projections,
256-wide pointer q/k head, frozen base/vision, action cross-entropy, gradient
clipping 1.0, AdamW defaults, 1 FPS, up to 32 causal frames, 384px and 144
vision tokens/frame, existing memory and candidate descriptions.

For the primary experiment retain constant pointer LR `1e-4` and LoRA LR
`2e-6`, with no new scheduler or automatic linear LR scaling. This keeps LR
policy out of the first comparison. Match all optimizer defaults, including
weight decay. A longer run with a new scheduler would be a separate arm.

At global batch 128, use 16 examples per rank per optimizer update. Split
physical forward/backward work further for long contexts if necessary, while
accumulating to the same 128-example update. At batch 32, use four examples
per rank, with a safe physical chunk. Context length changes physical chunks,
never the effective global batch or examples' loss weights. Sum per-example
loss, normalize by actual global example count, reduce gradients once, clip
once, and step once. Test partial updates as well as full updates.

Use the [GPU reference](../reference/gpu_optimizations.md) together with the
newer [current-node follow-up](../experiments/qwen35_2b_current_node_gpu_util_followup.md):

- Bounded, one-batch CPU preparation prefetch; frame cache; parallel decode;
  expandable allocator; packed gradient reduction; existing Conv1d chunking.
- Preserve length grouping within representative windows. Check decode/cache
  hit rate after changing order; unrestricted row shuffling can erase locality.
- Use NCCL only after a same-node collective preflight. The follow-up host
  required packed Gloo reduction; hold backend and host constant across arms.
- Retain existing model kernels. The faster-kernel probe had unexplained
  loss/gradient differences; cost-balanced rank assignment was slower.
- Target at least 10% peak-memory headroom in sustained mixed-context tests.
  Do not increase effective batch merely to fill memory or hit a utilization
  target. Log allocator and sampled device peaks; either can reveal a limit.

Before the quality runs, profile batch 32 and 128 on identical representative
rows, covering all context tiers and the longest inputs. Warm up, then measure
at least 100 updates per recipe and repeat the timing window once. Report
presentations/s, visual/model tokens/s, optimizer updates/s, per-rank phase
times, padding, cache hits, GPU utilization/idle samples, memory, and GPU-hours.
Follow with a save/resume continuity check.

CPU prefetch reached 19.61 presentations/s and 80.98% utilization in the
fixed batch-48/Gloo benchmark, but the mixed-plan 20-step test measured
12.32/s and 51.09%. Neither establishes the rate for this batch-128 plan.
High utilization is a useful diagnostic; quality gained per GPU-hour is the
decision metric. Do not promise 80% utilization or reuse a different node's ETA.

## Evaluation and decision rules

Primary quality is **judged, closed-loop PAUC-F**, with precision and recall
reported separately. Use the frozen promoted MoE answer generator, greedy
generation, 96-token cap, actual generated history, response margin 0, and the
same GPT-4.1 `pauc-f-v1`, omega=0.5 protocol. Keep generator, judge settings,
software, and benchmark manifests fixed across arms. Cache judge results for
identical items and retain judge/version provenance.

At N/2N/4N, evaluate the full frozen development cohort. Alongside PAUC-F,
record useful/wasted fires, fires/video, repeats, Hold/Silence/Response rates,
response delays, and per-video/source/context/memory strata. Log teacher-forced
action CE, confusion, and calibration as diagnostics; do not use lower CE or
timing-only scores as a substitute for judged quality. Inspect gold-history
versus rollout behavior on training-disjoint development videos.

Predeclare **1.0 PAUC-F point** as the practical noninferiority tolerance; this
is a proposed decision threshold, not a measured noise floor:

1. B passes if the paired-video 95% interval for B−A on development has lower
   bound above -1.0. An inconclusive interval is not a pass. A large precision
   or Hold-rate shift triggers diagnosis even if the aggregate passes.
2. Continue C/R through 4N only after checking the 2N development result.
   Stop expansion if C is clearly worse than B beyond the tolerance. A single
   noisy point warrants checking judged completeness and uncertainty first.
3. The selected C dose must be noninferior to B and A. A positive C−R interval
   supports benefit from fresh diversity; a positive C−B interval supports an
   absolute gain from scaling. Noninferiority alone supports retention, not
   improvement. C≈R leaves the diversity explanation unresolved.
4. Freeze dose and recipe using development; repeat A and the selected C
   training trajectory for seeds 43/44. Evaluate the frozen confirmation cohort
   for all three seeds. Report each seed and the mean, standard deviation, and
   worst seed; use paired-video intervals within seeds. With three seeds,
   avoid strong population-level claims about seed variance. Report a
   video/seed hierarchical bootstrap as descriptive uncertainty if used.
5. Finally evaluate A and the chosen C checkpoint for all three seeds on all
   500 WEB videos, even if the result disappoints. Do not tune margins or choose
   another checkpoint on WEB. Compare against 54.67 as a historical seed-42
   result and against the new paired A runs. A desired stable-scale outcome is
   mean WEB degradation under 1 point, no seed losing over 2 points, and no
   systematic precision collapse; report uncertainty, including inconclusive
   outcomes, rather than forcing a pass/fail claim.

For final WEB comparison, require 500 unique expected IDs, complete judging,
matching protocol hashes apart from controller identity, and exact scorer
reproduction before 1,000 paired-video bootstrap replicates. Rejudge a frozen
10% development sample once to estimate judge repeatability separately from
video-bootstrap and training-seed uncertainty. Never count repeated decisions
from the same video as independent samples.

## What to do with the outcome

| Outcome | Next action |
| --- | --- |
| B loses to A | Keep batch 32; test fresh/replay there. Diagnose optimization before expanding batch again. |
| C retains/improves quality and beats R | Confirm across seeds, then expand unique coverage. |
| C and R both deteriorate | Test a separate continuation from B with both LRs reduced 4×; compare at identical doses. Do not silently change the primary arms. |
| C loses while R stays stable | Audit new-source labels, histories, per-video concentration and strata; run a separate 25% anchor-replay / 75% fresh continuation at the same budget. |
| Teacher-forced metrics improve while rollouts worsen | Investigate rollout-history mismatch; propose a separate rollout-negative/relabeling study. |
| Quality is retained but no improvement is measurable | Report stable scaling and its cost; do not claim a diversity gain. |

Do not start with replay weighting, class weighting, margin tuning, new kernels,
or an RL loss all at once. Those remain distinct, conditional experiments.

If 4N succeeds, expand to 8N and 16N with the frozen batch/optimization recipe,
representative ordering, and the same diversity audits. Reserve additional
confirmation data before using development repeatedly. Only then test the full
raw manifest's natural duplicate/mixture weighting as an explicit change. A
full pass at batch 128 would take about 65,181 updates, versus the previous
variable-batch plan's 10,134; budget it from measured throughput and quality,
not from the original five-to-six-day projection.

## Implementation and deliverables before launch

This document is a plan, not an executable runner. The current
`train_jevany_2b_batched.py` supports physical batching, CPU prefetch, plans,
and resume, but its learning rates are hard-coded and resume requires the same
manifest/plan hashes. Implement these narrow additions before execution:

1. A deterministic manifest builder/auditor for held-out IDs, anchor/fresh/replay
   blocks, representative order, overlap, prefix distributions, and checksums.
   Record position identity separately from full-input/history identity.
2. Dose-aware fixed-global-batch plans with exact partial-update accounting.
   Pin C/R manifests before B starts. Support an explicit parent-checkpoint
   branch operation that validates the shared prefix, model/optimizer recipe,
   and cursor, while preserving parent artifacts; do not bypass resume hashes.
3. Configurable LR groups and a safe optional physical-chunk limit independent
   of global batch. Record LR, clipping, gradient norms, actual counts, label
   counts, and all data/model/software hashes. Persist Python/NumPy/Torch/CUDA
   RNG states where used and prove interrupted/resumed cursor continuity.
4. Meaningful synthetic checks for loss/gradient normalization across physical
   chunks and partial batches, exact row coverage, prefix audit failures, and
   branch/resume invariants. Real-model parity probes should disable dropout
   for numerical comparisons, then exercise training dropout separately.
5. A sequential eight-GPU runner, save every 100 updates and at exact dose
   boundaries, plus complete evaluation/provenance and paired-bootstrap reports.

Suggested new output root: `outputs/jevany-qwen35-2b-controlled-scale/`.
Save the frozen plan, cohort IDs, manifests, batch plans, recipes, profiler
reports, all dose checkpoints, predictions, judged results, and a decision
table there. Estimate duration after the representative profile: each 4N
trajectory costs `1,055,040 / measured_presentations_per_second`, plus evaluation
and checkpoint overhead. C/R share the N-prefix training cost per seed.

The first authorized follow-up should be implementation and the bounded
profile/A-versus-B quality screen. This proposal does not restart the stopped
step-505 run or launch a multi-day full-manifest job.
