# Qwen3.5-2B mixture experiments: completed results

Date: 2026-11-06. Status: **both runs complete; all eight scheduled full WEB
evaluations are judged and verified**. No continuation or new training was
launched as part of the analysis or documentation.

## Decision

Do not commit to a full-manifest run with the current recipe unchanged.
Both trajectories have substantial changes in firing behavior, and neither
final checkpoint exceeds the historical 54.67 PAUC-F reference. This does not
establish that action cross-entropy is fundamentally unsuitable or that more
data inherently harms quality. Optimization, exposure scheduling, and the
gold-history versus generated-history mismatch have not been isolated.

The completed experiment is **not an equal-training-time or equal-exposure
A/B test**. The original design proposed twelve training hours per arm, but
the user explicitly replaced timed stopping with preflight-estimated, frozen
step counts. Actual training durations and consumed prefixes differed. The
intermediate maxima below are diagnostic observations, not a newly selected
checkpoint or a change to the preregistered final-checkpoint outcome.

## Design and actual execution

The intervention was exposure scheduling: existing source/episode order with
bounded length grouping versus class-aware interleaving with natural class
quotas. Neither arm added class weights, rare-class duplication, relabeling,
replay, deduplication, or a new per-video cap.

| Setting | A: existing-order control | B: representative mixture |
| --- | --- | --- |
| Node | `ip-10-3-120-167`, eight H200s | `ip-10-3-70-177`, eight H200s |
| Effective global batch | 128 | 128 |
| Examples per GPU per optimizer update | 16 | 16 |
| Physical forward/backward chunk | 8, two accumulated chunks | 16, one chunk |
| Gradient backend | Packed NCCL | Packed CPU-buffer Gloo; NCCL preflight failed with error 802 |
| Frozen total optimizer updates | 3,608 | 1,776 |
| Frozen WEB checkpoint steps | 902 / 1,804 / 2,706 / 3,608 | 444 / 888 / 1,332 / 1,776 |
| Total presentations | 461,824 | 227,328 |
| Estimated training hours at launch | 12.004 | 12.026 |
| Actual cumulative training hours | **13.533** | **8.914** |
| Training-only allocated GPU-hours, eight GPUs | 108.27 | 71.31 |
| Completion, including all scheduled judging | 2026-11-06 14:06:26 UTC | 2026-11-06 08:52:58 UTC |

Evaluation, checkpoint I/O and model-loading time are additional to the recorded
training-loop elapsed time. A reached approximately 5.65% of the eligible pool;
B reached approximately 2.78%. Neither completed an epoch or established
convergence. Matching checkpoint ordinal does not match hours, updates, rows,
or unique positions. Backend, physical chunk size, node, cache/padding behavior
and consumed source/context/history distributions also limit causal attribution
to class scheduling alone.

The original design and amendment are preserved in the
[execution plan](../plans/qwen35_2b_12h_mixture_experiments.md).
The authoritative frozen budgets are
[A launch plan](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/launch_plan.json)
and [B schedule](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/schedule.json).

## Model, objective and data contract

- Fresh official `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, seed 42. Both initial LoRA/head
  states are byte-identical; neither resumed the old step-505 controller.
- Language LoRA r64/alpha128/dropout 0.05 and the existing 256-wide q/k pointer
  head. Base model and vision weights remain frozen.
- Existing three-action cross-entropy on gold-history examples; no class
  weighting or sequence-level reward. Sum per-example losses, normalize by
  actual global batch, average gradients across ranks, clip once, and take
  one optimizer step. The two physical chunks in A still form one 128-example
  global update; gradients are reduced once after accumulation.
- AdamW pointer LR `1e-4`, LoRA LR `2e-6`, weight decay 0.01, betas
  (0.9, 0.999), eps `1e-8`, gradient clip 1.0, constant LR without a scheduler.
- Existing candidates and gold memory/history inputs; 1 FPS, up to 32 causal
  frames, 384px and 144 vision tokens per frame. Rows remain self-contained.
- Both use CPU preparation prefetch, four decode workers, a 1,024-frame cache
  per GPU, expandable CUDA allocation and Conv1d chunking at 128. No unvalidated
  faster-kernel replacement was adopted.

The shared eligible pool contains **8,173,782 presentations, 4,595,758 unique
video/question/frame positions and 158,103 videos** after excluding 169,306
presentations with benchmark/validation video overlaps, including alias checks.
Its class proportions are **81.9121% Silence / 15.6593% Response / 2.4285% Hold**.
Original row contents, gold histories and presentation multiplicities are
preserved. Repeated positions are not automatically identical inputs.

A retains stable estimated-token sorting within 8,320-row source-order
windows. B interleaves natural class quotas in 4,096-row windows, with
fractional rounding carried forward, constrained length grouping and rotating
rank ownership. Full B updates have 104–105 Silence, 19–21 Response and 3–4
Hold rows. Full-window and cumulative-prefix mixture audits pass. The terminal
86 rows remain in the full plan but cannot form a complete 128-example update;
these bounded runs do not reach that tail.

A's first 4,096 rows contain 2,478 Silence / 1,256 Response / 362 Hold
(8.84% Hold). Its four successive training segments contain 10.37%, 10.82%,
11.19% and 11.35% Hold. The consumed control prefix therefore remains far from
the full-pool prior. This is a measured control property, not an injected skew.

Exact full A/B row-multiset equality was verified by hashing the original
source bytes in the representative offset plan's source order and matching
the eligible-pool hash. Initial-state and data hashes:

| Artifact | SHA256 |
| --- | --- |
| Source J1-S manifest | `363279af125eaea823208817ef6a82afccb46e13cc4d463c264e99d487974e69` |
| Eligible pool in source order | `57547c3cf6d8f988456af5728c28a193a9b50dad8c83502d55d05951bb349248` |
| Emitted A order | `6ed39b1cf13db9331db9f7b1cb1c74152120fe9365160468462711c33dddb703` |
| B offset order used in training | `672c3a3cd8356d08d4b748821b2cd771864ba2d51e9ff85b4218754e13f77d13` |
| Initial LoRA weights | `4dbe28e6eeb4c3817852017b576e0f971ce142f3eee43596f046beeb5ccfd54d` |
| Initial pointer head | `53acb56d179c2c938d7c8865c97f774cc5458c16a2ea34b6de43aa63ee4d5fed` |

See [common pool audit](../../outputs/jevany-qwen35-2b-mixture-12h/shared/manifests/audit.json),
[control exposure audit](../../outputs/jevany-qwen35-2b-mixture-12h/shared/manifests/control_exposure_audit.json),
[supplemental TV/VAD zero-overlap proof](../../outputs/jevany-qwen35-2b-mixture-12h/shared/manifests/supplemental_exclusions.json)
and [B plan audit](../../outputs/jevany-qwen35-2b-mixture-12h/plans/audit.json).

## Evaluation protocol and integrity

Every scheduled checkpoint evaluated the same **500 WEB video IDs** with the
frozen promoted MoE generator, greedy generation, 96-token cap, response
margin 0, generated rollout histories and GPT-4.1 `pauc-f-v1`, omega 0.5.
The generator adapter SHA256 is
`eb11e600de59edf48ef2d71bb632ae5944c0474441f264bb3ce6ceae8bb71da9`;
WEB annotation SHA256 is
`91f348a9f1dc6823277d5bf7bdd741a8b0a38e65655f0d4a494b9f0f0ed7b783`.

All eight evaluations were rechecked during documentation: 500 unique expected
prediction IDs and score IDs each, complete unique judge-request coverage,
valid judge responses, correct scorer/version/omega and agreement between
saved scores and reports. Normalized WEB protocols match across arms and
checkpoints after omitting controller adapter/head hashes and controller
identity. This is 4,000 video-evaluations over the same benchmark cohort.
The [documentation audit](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/reports/documentation_runtime_audit.json)
records the checks and control runtime summaries.

## All scheduled WEB results

Precision and recall below are percentages. "Useful" uses the scorer's
earliest positive judged fire per gold turn; "wasted" includes remaining
physical fires. PAUC recall is the official time/quality-weighted component,
not ordinary event recall.

### A: existing-order control

| Step | Presentations | Training hours | PAUC-F | Fire precision | PAUC recall | Useful / wasted fires |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 902 | 115,456 | 3.948 | 45.44 | 45.66 | 45.22 | 795 / 946 |
| 1,804 | 230,912 | 7.468 | 49.47 | 61.69 | 41.29 | 615 / 382 |
| 2,706 | 346,368 | 10.637 | 43.20 | 45.40 | 41.21 | 641 / 771 |
| 3,608 | 461,824 | 13.533 | **48.70** | 57.44 | 42.28 | 641 / 475 |

Primary final score: 48.70. The observed intermediate maximum is 49.47 at
step 1,804. The historical completed J1-S reference scored 54.67 with global
batch 32, 263,760 presentations and 8,243 updates. A's final gap is −5.97
points, but those historical recipe/exposure/update differences prevent
interpreting it as an isolated data-scaling effect.

### B: representative mixture

| Step | Presentations | Training hours | PAUC-F | Fire precision | PAUC recall | Useful / wasted fires |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 444 | 56,832 | 2.460 | 40.49 | 83.82 | 26.69 | 57 / 11 |
| 888 | 113,664 | 4.768 | 47.74 | 74.80 | 35.06 | 383 / 129 |
| 1,332 | 170,496 | 6.907 | 51.73 | 77.81 | 38.75 | 512 / 146 |
| 1,776 | 227,328 | 8.914 | **47.61** | 50.44 | 45.08 | 738 / 725 |

Primary final score: 47.61. The observed intermediate maximum is 51.73 at
step 1,332. The last segment adds useful fires and recall but substantially
more wasted fires. Natural class scheduling does not by itself establish
stable closed-loop quality. These unequal-budget outcomes do not establish
a causal A/B winner.

### Action and repeat diagnostics

Each evaluation contains 8,803 controller decisions. Repeat counts below are
within-video answer-text repeats: A lowercases and collapses whitespace;
B lowercases and trims leading/trailing whitespace. These descriptive counts
are not a semantic repeat classifier; use judged useful/wasted fires for
quality conclusions.

| Arm / phase | Hold decisions | Response fires | Silence decisions | Repeated identical answers |
| --- | ---: | ---: | ---: | ---: |
| A1 | 2,897 | 1,741 | 4,165 | 251 |
| A2 | 2,389 | 997 | 5,417 | 112 |
| A3 | 1,362 | 1,412 | 6,029 | 181 |
| A4 | 1,820 | 1,116 | 5,867 | 113 |
| B1 | 0 | 68 | 8,735 | 0 |
| B2 | 293 | 512 | 7,998 | 36 |
| B3 | 7 | 658 | 8,138 | 46 |
| B4 | 1,296 | 1,463 | 6,044 | 323 |

## Checkpoint fluctuations and uncertainty

A's step-1,804 → step-2,706 decline is primarily a precision deterioration:
useful fires rise slightly (615 → 641), wasted fires double (382 → 771),
and PAUC recall remains almost unchanged (41.29 → 41.21). Its final recovery
retains 641 useful fires while reducing wasted fires to 475. Aggregate Hold
or fire frequency alone is not a success criterion.

The diagnostic paired bootstrap uses the same 500 video IDs, 1,000 replicates
and RNG seed 42. Per-video contributions reproduce the official point scores
before resampling. These are exploratory, unadjusted comparisons selected
after observing the curves.
The calculation uses `per_video` and `resampled_scores` from
[bootstrap_exp10_web_ego.py](../../scripts/bootstrap_exp10_web_ego.py), sorted
question IDs, and shared `numpy.random.default_rng(42)` index draws of shape
`(1000, 500)` for all checkpoints. This preserves pairing within each replicate.

| Comparison | PAUC-F difference | Paired-video 95% interval |
| --- | ---: | --- |
| A step 2,706 − A step 1,804 | −6.27 | [−7.79, −4.79] |
| A final − A step 1,804 | −0.77 | [−2.18, +0.70] |
| A final − A step 902 | +3.26 | [+1.76, +4.73] |
| B final − B step 1,332 | −4.12 | [−5.82, −2.34] |

The sharp drops persist under video resampling. The final A checkpoint is
better than A's first checkpoint, but not clearly better than A's middle
checkpoint. Intervals do not estimate training-seed variance, numerical
trajectory variation or judge repeatability. Only one training seed and one
judged evaluation per checkpoint were used. No clean matched-dose or
matched-time A/B effect has been measured.

Machine-readable uncertainty:
[checkpoint bootstrap](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/reports/checkpoint_fluctuation_bootstrap.json).

## Runtime observations, validation and deviations

Control production GPU samples were filtered to each phase's training-loop
start through its final completed update, excluding evaluation/loading windows.
Sampling is every ten seconds; it is a utilization diagnostic, not an exact
GPU duty-cycle profiler.

| A phase | Mean sampled GPU util. | Samples at ≤5% util. | Peak device memory, MiB |
| ---: | ---: | ---: | ---: |
| 1 | 30.88% | 57.60% | 38,612 |
| 2 | 35.43% | 54.35% | 38,272 |
| 3 | 38.02% | 52.46% | 38,154 |
| 4 | 42.26% | 48.84% | 37,878 |

H200 capacity is 143,771 MiB. Memory was underused relative to capacity.
The control's physical chunks of eight were a conservative implementation
choice, not a measured throughput optimum. Full chunks of 16 were not
benchmarked for this control; no speedup factor is established. Accumulation
does not double mathematical work or change effective batch, but can add
launch overhead and reduce GPU efficiency. The recipe was not changed mid-run.

Per-phase mean rank-0 CPU prefetch waits were under 0.006 seconds per update. Mean
gradient-sync-boundary times ranged from 6.53 to 4.91 seconds per update;
those boundaries include waiting and potentially outstanding GPU work, and
do not prove NCCL bandwidth is the bottleneck. The exact slow-rank/root cause
was not established with a per-rank profiler. Earlier ~81% utilization came
from a different, global-batch-384 benchmark and is not a guarantee for this
global-batch-128 experiment. See the
[GPU reference](../reference/gpu_optimizations.md).

Control software versions were Torch 2.5.1+cu121, CUDA 12.1, Transformers
5.5.0, PEFT 0.18.1, Triton 3.1.0 and FLA 0.3.2. See the
[preflight report](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/preflight/report.json)
and each arm's recipe for exact configuration.

A's validation included 16 passing tests, eight-rank NCCL scalar and gradient-size
collectives, representative and maximum-context capacity checks, exact loaded
LoRA/head/optimizer/Torch-CUDA-Python-RNG/cursor/time restoration on every rank,
and a resumed update advancing the expected optimizer and data counters.
Independent uninterrupted repetitions still differed numerically despite
matching RNG; two-update head differences were about 1.6e-4. Checkpoint state
restoration is exact; bitwise-identical training trajectories are not promised.
This is not proof that numerical variation caused the observed WEB swings.
B's separate continuity check compared adapter/head/optimizer/RNG tensors
with rtol `1e-5`, atol `1e-6`; its validation should not be described as the
same byte-exact restoration check used by A.

A synthetic 500-video silent fixture verified official build/process/score and
zero-fire PAUC-F without judge requests. The scorer wrapper skips an empty API
batch while retaining the official metric. The fixture is a preflight test,
not an experiment checkpoint. A real tiny judge connectivity check passed for
A; B's separate optional historical-request connectivity check was rejected
by automatic approval review and skipped. All scheduled experiment judging
subsequently completed successfully.

Control preparation had two recovered implementation issues: a live-edited
shell driver stopped after a completed preflight pass, and an initial
import-only production launch lacked snapshot YAML configs. Completed
preflight artifacts were retained, configs were added and hashed, snapshot
imports were checked, and production restarted from the shared fresh state.
Neither issue consumed production optimizer updates. Archived attempt logs
remain in A's `logs/`. Production used a private source/config snapshot;
subsequent repository edits did not change its four training phases.

## Interpretation and next experiment

Evidence supports unstable closed-loop firing behavior; it does not separate
optimization/calibration, training exposure, label quality or objective/history
mismatch. Moving training-loss means over the last 100 updates of A's phases
were approximately 0.450 / 0.420 / 0.458 / 0.404. They are moving training
statistics on different rows, not held-out CE, calibration or a monotonic
objective-to-quality curve.

The action-CE objective supervises gold per-frame actions and gold histories.
WEB measures useful, timed generated answers over a rollout with the model's
own histories and memory. That mismatch is a plausible limitation, not a
demonstrated cause in this experiment. The general sequential imitation-learning
distribution-shift issue is described by
[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html).
There is no evidence here establishing a need to replace SFT with RL.

Before a substantially larger run, use bounded, separately controlled studies:

1. Measure fixed, video-disjoint development action CE/confusion/calibration
   together with closed-loop quality. Compare gold versus generated histories.
2. Test a lower head LR or an LR schedule separately from a batch-32 reference;
   retain matched data and explicitly account for update/exposure differences.
3. If teacher-forced performance is good but rollouts fail, test model-generated
   histories with verified corrective supervision as a distinct intervention.
4. Freeze choices using development data and confirm the chosen comparison
   across additional seeds and a reserved confirmation cohort before scaling.

WEB has already been repeatedly inspected. Do not use its observed checkpoint
maxima to tune margins, LR, sampling or checkpoint selection and then present
the same WEB cohort as an untouched confirmation test. The older
[controlled scaling plan](../plans/qwen35_2b_controlled_scaling_plan.md)
contains proposed follow-ups; those branches are not completed results.

## Artifact map

| Purpose | Artifact |
| --- | --- |
| A configuration and source/config snapshot hashes | [recipe](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/train/recipe.json), [snapshot metadata](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/code_snapshot.json) |
| A complete checkpoint reports | [report.json](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/report.json) |
| A raw WEB rollouts, per-decision traces and judged scores | [evaluation directory](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/evaluation/) |
| A runtime logs and GPU samples | [logs](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/logs/) |
| A immutable scheduled/recovery checkpoints | [train](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/train/) |
| A launch/preflight details | [control record](qwen35_2b_control_12h.md) |
| B configuration | [recipe](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/train/recipe.json) |
| B complete checkpoint reports | [results.json](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/reports/results.json), [detailed results](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/reports/detailed_results.json) |
| B raw WEB rollouts, per-decision traces and judged scores | [evaluation directory](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/evaluation/) |
| B scheduled/recovery checkpoints | [train](../../outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/train/) |
| B launch/preflight details | [representative record](qwen35_2b_representative_12h.md) |
| Completion notification delivered back to this chat | [watcher state](../../outputs/jevany-qwen35-2b-mixture-12h/control-seed42/completion_watch.json), [watcher implementation](../../scripts/watch_qwen35_control_completion.py) |

The eight full WEB runs, immutable checkpoints, scorer inputs/outputs and frozen
plans are preserved. Documentation and diagnostic audits did not restart or
modify either training experiment.
Earlier autogenerated B reports retain a generation-time statement that the
control was pending; this consolidated completion report supersedes that status
text without rewriting the original run artifacts.
