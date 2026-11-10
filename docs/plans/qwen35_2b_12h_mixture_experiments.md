# Qwen3.5-2B mixture experiments: original twelve-hour design

Design date: 2026-11-05. Updated: 2026-11-06. Status: **both arms complete;
all eight full WEB evaluations judged and verified**. See the authoritative
[completed results and limitations](../experiments/qwen35_2b_mixture_12h_results.md),
[control record](../experiments/qwen35_2b_control_12h.md) and
[representative record](../experiments/qwen35_2b_representative_12h.md).

Execution amendment (2026-11-05): the user requested a **fixed optimizer
step budget estimated from preflight throughput**, rather than a training-time
stop. Freeze that estimate before launch, divide it into four equal step
segments, and run WEB after each segment. Record actual training hours; the
result is an estimate, not an enforced equal-time comparison. A froze
3,608 updates and completed 461,824 presentations in 13.533 training hours;
B froze 1,776 updates and completed 227,328 presentations in 8.914 training
hours. WEB ran at each arm's quarter-step milestones. Their final PAUC-F
scores are 48.70 and 47.61, respectively. The actual runs are neither
equal-time nor equal-exposure. Frozen parameters and results live under
`outputs/jevany-qwen35-2b-mixture-12h/{control,representative}-seed42/`.

The sections below preserve the original proposed time-budget specification,
superseding the fixed-dose and
100K-evaluation schedules in the [earlier mixture plan](qwen35_2b_data_mixture_plan.md).
Original constraint, replaced by the amendment above: **12 hours of training per run, plus evaluation time**;
four complete WEB evaluations after 3, 6, 9 and 12 cumulative training hours.

## The two experiments

| Setting | Experiment A: existing order | Experiment B: representative mixture |
| --- | --- | --- |
| Node | First eight-GPU node | Second eight-GPU node |
| Question | How does the existing order behave under a 12-hour training budget? | Does representative class exposure yield better quality under the same budget? |
| Data pool | Full eligible J1-S pool | Exactly the same pool and row multiplicities |
| Ordering | Existing source/episode order followed by existing bounded length grouping | Class-aware interleaving with constrained length grouping |
| Target class weighting | Existing natural weighting | Same natural weighting, spread throughout training |
| Training budget | 12 cumulative hours | 12 cumulative hours |
| WEB evaluations | After 3/6/9/12 training hours | After 3/6/9/12 training hours |

The intervention is exposure scheduling. Do not add equal-class sampling,
inverse-frequency loss weights, relabeling, hard-negative mining, replay,
deduplication or a new per-video cap in either arm. Audit their relevance, but
defer those separate changes until this comparison is complete.

## Why this is the first mixture intervention

The full manifest is approximately **81.96% Silence / 15.64% Response / 2.40%
Hold**, close to the earlier successful run's 81.78% / 15.63% / 2.59%.
Yet the stopped run's first 426,464 presentations contained 10.90% Hold and
25.50% Response. Every rolling 20-update window through step 505 contained
8.44–13.32% Hold. On WEB, that checkpoint predicted Hold on 30.81% of decisions
versus 3.00% for the earlier final controller, and PAUC-F was 47.12 versus 54.67.
This supports testing scheduling; it does not prove that label proportions
caused the quality gap.

Matching action frequency is insufficient: step 25 fired 1,006 times versus
the earlier final's 991, but produced only 321 useful fires versus 701. Judge
the new recipe by useful-fire precision/recall and PAUC-F, not proximity to a
chosen Hold or fire rate.

Sources: [consolidated findings](../experiments/qwen35_2b_full_data_findings.md)
and [recomputed diagnostic](../../outputs/jevany-qwen35-2b-full/reports/web_class_mix_diagnostic.json).

## Shared training contract

- Fresh official `Qwen/Qwen3.5-2B`, pinned resolved revision; seed 42. Save
  one initial LoRA/head state and copy it into both experiments. Do not resume
  the Hold-heavy step-505 checkpoint.
- Existing r64/alpha128/dropout-0.05 language LoRA and 256-wide q/k pointer
  head; frozen base and vision; existing three-action cross-entropy.
- AdamW defaults, pointer LR `1e-4`, LoRA LR `2e-6`, gradient clip 1.0,
  constant LR and no new scheduler. Record all defaults, including weight decay.
- Fixed global batch **128** across eight ranks: 16 examples per GPU per
  optimizer update. Reduce physical chunks for memory while retaining the
  same global batch and per-example loss normalization.
- Existing J1-S labels and gold-history inputs; 1 FPS, 32 causal frames,
  384px, 144 vision tokens/frame, existing candidates and memory contract.
- Same software, kernels and validated collective backend where possible.
  The prior nodes differed in NCCL functionality; record any backend or other
  runtime difference as a limitation rather than assuming node equivalence.

Batch 128 is a fixed candidate for this experiment, not a known quality match
to historical batch 32. Preflight both nodes and orders; if it is unusable,
freeze one smaller effective batch for both before either budget begins.
Do not sweep or change effective batch during the experiment.

## Data preparation and mixture algorithm

Use the full eligible J1-S pool, approximately 8.34M presentations before
exclusions, so neither run exhausts a small subset and silently starts a new
epoch. Exclude all known WEB/other benchmark and validation overlaps using
video identity, checking aliases. Reserve development/confirmation videos
before planning, but do not add their GPU evaluations to this 12-hour trial.
Hash the common eligible pool and both emitted orders. If exclusions change
the natural class proportions, freeze and report the resulting proportions.

For A, retain the existing ordering policy and quantify its emitted exposure.
For B, build deterministic per-class queues across the whole pool, with
source/context/memory diversity and short video-local groups for cache reuse.
Fill **4,096-row windows** with natural class quotas, carrying fractional
rounding residuals forward. Spread the quotas over the window's 32 global
updates rather than emitting class-contiguous runs. A typical update averages
about **105 Silence / 20 Response / three Hold**, with rounding variation.

Choose nearby context lengths across class queues to reduce padding; never
apply an unconstrained global length sort afterward. Do not require every
rank's physical chunk to contain all three classes. Preserve row contents,
causal history and labels exactly; rows must be self-contained, with history
independent of sampling order. Every presentation is used at most once during
these runs, retaining the source pool's original multiplicities of positions.

Before launch, audit full-window and cumulative-prefix action proportions:
within one percentage point of target, Hold within 0.5 points, allowing
unavoidable rounding. Report per-update class ranges, runs without rare
classes, source/context/history distributions, rank distributions and padding.
Audit exact row-multiset equality of the full A/B plans. Never pass an audit
by duplicating rare-class rows. If A does not show meaningful exposure skew,
report weak contrast rather than manufacturing a bad control.

## Schedule: evaluations are additional to 12 training hours

Track cumulative training-loop elapsed time, including ordinary CPU input
preparation, GPU work and synchronization. Exclude evaluation, model-loading,
checkpoint I/O and deliberate pauses. Persist this counter across resumes.
At each three-hour threshold, finish the current optimizer update, save a
complete checkpoint and release training GPU allocations. Run full WEB with
that immutable checkpoint, then reload and resume exactly where training left
off. Small timing overshoot from completing an update is acceptable.

| Phase | Training clock | Expected elapsed completion, excluding reload overhead |
| --- | --- | --- |
| Train 1 + WEB 1 | 0–3 h, then pause | 3 h 30–45 min |
| Train 2 + WEB 2 | 3–6 h, then pause | 7 h–7 h 30 min |
| Train 3 + WEB 3 | 6–9 h, then pause | 10 h 30 min–11 h 15 min |
| Train 4 + WEB 4 | 9–12 h, then stop | 14–15 h |

Four 500-video WEB evaluations per arm means **eight evaluations / 4,000 video
evaluations across both nodes**. Run the nodes independently; one need not wait
for the other. This budget assumes scoring completes before training resumes;
GPU-free judging may overlap resumed training later if implemented without
changing checkpoint association or evaluation completeness.

Historical step-505 eight-GPU WEB launch-to-judged-completion time was
30 min 11 s: [rollout log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_eval.log),
[completion log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_score.log).
Budget 30–45 minutes per evaluation, with **14–15 hours elapsed per experiment,
plus preflight, checkpoint/reload overhead and any judge delays**. Both run in
parallel, so paired elapsed time follows the slower experiment. Twelve training
hours on eight GPUs costs 96 allocated GPU-hours per arm, plus evaluation and
other allocation overhead.

## Expected exposure and fair interpretation

Training throughput for this exact recipe is unknown. Illustrative counts:

| Presentations/s | Presentations per 3 training hours | Presentations after 12 training hours |
| --- | ---: | ---: |
| 4.28 | ~46K | ~185K |
| 8 | ~86K | ~346K |
| 12.32 | ~133K | ~532K |

The low rate is an older pre-prefetch global-batch-128 benchmark; 12.32/s was
a newer short mixed-batch training test. Neither predicts these exact runs.
Profile representative rows on both nodes before starting their training
clocks and update projections after the first segment. Retain CPU preparation
prefetch, parallel decode, frame cache, packed reduction and established
kernels; measure cache/padding costs of the new sampler. Retain at least
10% measured peak-memory headroom in the preflight.

**This is a comparison at equal training-time budgets.** Different sampler or
node speeds may yield different exposures and optimizer counts. The full data
pool is identical, but the consumed prefixes need not contain identical rows.
Report PAUC-F versus both training hours and presentations/updates. Do not
interpolate sparse WEB points and call them a measured equal-exposure result,
or attribute all quality differences exclusively to class ratio. Save extra
lightweight checkpoints near common presentation milestones where practical;
evaluating those later is separate follow-up, outside these four WEB evaluations.

Depending on throughput, 12 hours may only reach approximately 0.7–2.0 times
the historical 263,760-presentation exposure and substantially fewer than its
8,243 optimizer updates. Treat this as a bounded mixture diagnostic, not proof
of full-data scaling or convergence. Historical 54.67 is a reference, not an
automatic pass/fail threshold for these intermediate checkpoints.

## WEB protocol, outputs and decision

At every scheduled checkpoint run all 500 identical WEB videos with the
promoted frozen MoE generator, response margin 0, greedy generation, 96-token
cap and unchanged GPT-4.1 `pauc-f-v1`, omega=0.5 scoring. Verify exactly 500
expected unique IDs, complete judging and matching protocol hashes apart from
controller identity. Record frame/time, raw action logits, chosen action and
memory/history status to diagnose missed opportunities and wasted fires.

For each checkpoint report training hours, actual presentations, unique
positions/videos, optimizer updates, cumulative and recent training label mix,
throughput, padding/cache/memory measurements, WEB PAUC-F, fire precision,
PAUC recall, useful/wasted fires, repeats and Hold/Response/Silence rates.
Compare A/B on the same videos at each training-time checkpoint with 1,000
paired-video bootstrap replicates; intervals quantify video uncertainty, not
seed variance or judge repeatability.

Primary outcome: **12-training-hour PAUC-F and useful-fire precision/recall**,
with the four-point curves showing whether quality improves or deteriorates.
B is promising if its actual mixture audits pass and its quality/cost curve
improves over A without merely changing firing frequency. Report ties,
inconclusive intervals and failures. Hold rates alone are not success criteria.
Do not change recipe, thresholds or checkpoint selection after viewing WEB.
Complete both budgets and report all four scheduled scores unless a runtime
or integrity failure prevents it. Single-seed results motivate replication;
they do not establish stable scaling across seeds.

Suggested output roots:
`outputs/jevany-qwen35-2b-mixture-12h/control-seed42/` and
`outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/`.
Preserve existing artifacts. Implementation must add deterministic mixture
planning/audits, persistent training-time budgets, four-phase checkpoint/WEB/
resume orchestration and the paired report. Test quota rounding, coverage,
causal-input identity, timed stopping, full-state resume and no training-clock
reset after evaluation before launching either experiment.
