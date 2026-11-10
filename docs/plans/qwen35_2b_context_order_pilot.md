> RETIRED (2026-11-08): the user prohibited heuristic runtime decision margins. Margin-adjusted evaluations and associated selection/scaling gates in this historical plan must not be used. Checkpoints trained without offsets remain available for clean raw-score evaluation. See AGENTS.md and pace_clm.evaluation_policy.

# Candidate next experiment: isolate context-order effects

Status: authorized and launching on 2026-11-07. All work uses
`ip-10-3-70-177` and its eight H200 GPUs. The user also authorized GPT scoring
as needed. Frozen launch metadata and progress are under
`outputs/jevany-qwen35-2b-context-order-pilot/`.

## Why test this before changing the loss

The representative planner preserves natural action quotas, but sorts each
class's selected rows by length before emitting 32 global updates in a
4,096-row window. This creates a repeated short-to-long input-context cycle.
The completed run's checkpoints land at phases 28, 24, 20 and 16 of that cycle.
Rank-0 mean context lengths over the previous eight updates are 14.97, 9.86,
7.41 and 5.66 frames respectively. Gold-memory prevalence also varies.
Checkpoint age is consequently confounded with recent context/history exposure.
Four WEB points do not prove that this caused the quality changes.

The matched gold-history and development-margin diagnostics should first check
whether the final checkpoint has worse ranking/generalization, shifted
calibration, or mainly worse behavior with generated history. Semantic scores
and confidence intervals remain necessary for a quality conclusion.

## Single intervention

Shuffle the order of the already formed 128-row global batches **within each
4,096-row window**, using a frozen deterministic seed. Preserve the contents
and rank assignment of every batch. Preserve all rows and multiplicities,
per-update class counts, window quotas, image/history inputs, losses, model,
optimizer defaults and learning rates. Do not add sampling weights, replay,
new labels, new negatives, a new generator or a larger effective batch.

This breaks the systematic short-to-long progression while retaining the same
length grouping inside each update. Do not select a shuffle after seeing
validation results. Audit full-window row-multiset identity and both class and
context exposure before launching.

## Bounded continuation design

1. Reconstruct shared complete state at step **1,344**, a window boundary, by
   replaying 12 original updates from the saved step-1,332 checkpoint. Save
   adapter, head, optimizer, all rank RNG states and exact row cursor.
2. Compare original order with shuffled update order over **448 updates**,
   finishing at **1,792**. Both arms consume exactly the same 14 full windows:
   57,344 presentations. Their cumulative exposure is 229,376 presentations.
3. Keep global batch 128, physical batch 16 per GPU, the original seed and
   pinned base, original r64 LoRA/head, pointer LR 1e-4 and LoRA LR 2e-6,
   constant LR, clip 1.0 and the supported Gloo backend.
4. Run both the shuffled-order and original-order continuations fresh from
   the shared state, sequentially on this same node. Each takes 448 updates.
   Both use the same validated fixed launch geometry for the existing FLA
   normalization kernels. Tested forward/backward outputs were bit-identical
   across launch geometries; checkpoint restoration is verified on all ranks.
   This common runtime avoids the normalization-autotuning crashes observed
   during the earlier diagnosis.
5. Save lightweight states every 16 or 32 updates and measure matched-input
   loss/ranking across context phases. Evaluate predeclared closed-loop
   checkpoints, including the final state, with a fixed validation policy.

Forking the sampler at step 1,344 requires explicit branch provenance and a
verified unchanged consumed prefix; do not disable resume checksum safeguards.
Record wall time, cache hit rate, padding and utilization separately. A larger
batch or new kernels would confound this experiment.

## Evaluation and decision

Choose any response margin on development only, freeze the rule, then use a
separate video-disjoint confirmation set. For the new training pilot, draw fresh
videos from the remaining reserved cohorts rather than repeatedly optimizing
against the 32+32 diagnostic videos. Report judged PAUC-F@4s, precision, recall,
useful/wasted fires, repeats, paired-video intervals and actual training cost.
Label timing-only results as optimistic proxies.

The hypothesis is supported if the shuffled schedule reduces context-phase
sensitivity while retaining or improving confirmed quality at matched exposure.
Stable Hold/fire rates alone are not success. If it does not help, use the
ranking/history diagnostic to decide between a lower-LR continuation and an
explicit generated-history training intervention. Do not change several of
these factors in the same next run. Replication is needed before claiming a
scaling result.

## Frozen evaluation implementation

Use 64 fresh reserved development videos (4,487 decisions, 268 events) and 64
separate confirmation videos (4,207 decisions, 216 events). These exclude the
32+32 videos used in the preceding diagnosis. Primary comparison uses response
margin zero for both final checkpoints. Secondary comparison chooses each
arm's margin from -2, -1, 0, +1 and +2 on development only, freezes those
choices, then evaluates them on confirmation. Use the existing promoted MoE
answer generator and GPT-4.1 MarkII PAUC-F@4s scorer at the approved gateway.
No WEB tuning is performed.

Track matched-input loss/ranking and Response/Hold decision behavior at steps
1,744, 1,760, 1,776 and 1,792 on the same frozen 512-row gold-history probe.
Those classification measurements describe context-phase sensitivity; they
are distinct from semantic rollout quality. All checkpoint selection is fixed
in advance. Report paired-video 1,000-replicate intervals for the primary and
secondary confirmation comparisons.

The parent step-1,332 checkpoint was chosen after inspecting the prior WEB
curve. This is a targeted single-seed continuation ablation, not an independent
scaling validation. GPU samples, exact fork provenance, immutable data-plan
hashes, training counters, requests and judged scores are retained.

Evaluation recovery on 2026-11-07: the original Torch 2.5.1 / Triton 3.1
answer-generation process segfaulted before completing any experimental video.
The rerun uses the existing local `../lmms-eval/.venv/bin/python` runtime
(Torch 2.10.0+cu128, Transformers 5.5.3, PEFT 0.18.1, Triton 3.6.0), requires
native grouped MoE matrix multiplication, and retains the same checkpoints,
cohorts, response margins, promoted answer weights, greedy 96-token generation,
and GPT-4.1 rubric. A single-video closed-loop and GPT scoring smoke test passed,
but the full run also segfaulted in the newer runtime with FLA generator kernels.
Its three completed video rows were archived and excluded. The subsequent rerun
uses Transformers' PyTorch fallbacks for the frozen generator's gated-delta
recurrence, convolution and gated norm, retaining native grouped MoE GEMMs and
the existing dense controller kernels. Periodic 180-second stack dumps were also
disabled: both failures coincided with truncated dumps inside Triton's native
launcher. A trivial GPU-copy reproducer subsequently segfaulted with timed
stack dumps enabled; the matching timer-disabled control completed 485,999
launches with identical outputs. This reproduces a diagnostic failure without
model weights. Fatal-signal diagnostics remain enabled. The in-flight common
PyTorch-generator evaluation is retained after this diagnosis.
An eight-row controller comparison agreed on all actions but was not bit
identical (maximum probability difference 0.0201), so both arms' complete
gold-history probes are rerun in `evaluation/phase/torch210/` as well. The final
report compares arms under this common repaired runtime, with original-runtime
diagnostics retained separately. Runtime provenance is recorded in
`preflight/evaluation_runtime/runtime_repair.json` beneath the pilot output.

## Completed outcome

Both 448-update continuations and all GPT-4.1 evaluation conditions completed.
On the separate 64-video confirmation cohort, fixed-zero-margin PAUC-F@4s was
61.13 for reference and 61.34 for shuffled: delta +0.21, paired 95% interval
[-2.30, +2.79]. Both arms selected +1 on development. At that frozen margin,
confirmation scores were 64.14 and 69.00: delta +4.86, interval [+1.26, +9.61].
Useful fires were nearly equal (122 reference, 121 shuffled), while wasted fires
fell from 57 to 37 and identical repeats from 29 to 17.

This is a null primary result with a promising secondary quality signal.
Matched gold-history loss and ranking differences were inconclusive, and
checkpoint-to-checkpoint Response-probability variation was not reduced.
Replicate with another continuation seed, unchanged learning rates, a
predeclared +1 primary margin, and a fresh confirmation cohort before allocating
a larger scaling budget. Keep a lower pointer-head learning-rate ablation
separate so its effect can be identified.

Full results, paired intervals, runtime provenance and integrity checks are in
`outputs/jevany-qwen35-2b-context-order-pilot/reports/results.md`,
`reports/results.json`, and `reports/verification.json` under the same output.
