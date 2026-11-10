> RETIRED (2026-11-08): the user prohibited heuristic runtime decision margins. Margin-adjusted evaluations and associated selection/scaling gates in this historical plan must not be used. Checkpoints trained without offsets remain available for clean raw-score evaluation. See AGENTS.md and pace_clm.evaluation_policy.

# Qwen3.5-2B: continuation replication and gated full-source scaling

Authorized 2026-11-08. All training and GPU evaluation remain on
`ip-10-3-70-177`, using exactly its eight H200 GPUs. Every GPU launch uses
`torchrun --standalone --nnodes=1 --nproc_per_node=8`; hostname and GPU-count
checks run before work starts. GPT-4.1 judging uses the previously authorized
gateway from this node.

The previous pilot's primary margin-zero result was null, while its secondary,
development-selected +1 result favored shuffled order. The next primary
comparison is therefore fixed at +1 before any new predictions.

## Replication

Fork the same complete step-1344 state into reference and shuffled-order arms.
Keep base checkpoint seed 42 and all model, optimizer, LR and batch settings.
Explicitly reset continuation RNG streams to `43 + local rank` in both arms;
use seed 43 for the whole-global-batch permutations inside each 4096-row
window. Train 448 updates per arm to step 1792, saving every 16 updates.
Record parent and new RNG hashes on all eight ranks and verify that the two
arms start with matching new streams. Resume restores the saved streams and
never reseeds. Later scaling forks retain these streams.

This is another **continuation seed conditional on the same trained parent**,
not an independent from-scratch initialization. Pointer LR stays `1e-4`, LoRA
LR `2e-6`, effective global batch 128, 16 examples per GPU, unweighted CE,
Gloo reduction, frame cache 1024, four decode workers, preparation prefetch,
and the validated fixed FLA normalization launch geometry.

Before training, verify exact adapter, head, optimizer and RNG restoration on
all eight ranks. After training, compare the same four late checkpoint probes
and run actual generated-history rollouts at margin +1 on 64 new heldout videos.
Use the repaired evaluation runtime: Torch 2.10, native MoE grouped GEMMs,
standard PyTorch generator recurrence/convolution/gated norm, and no periodic
native-stack dump timer. Training retains its established Torch 2.5 runtime;
the two Triton cache directories remain separate.

Replication passes only if shuffled-minus-reference PAUC-F is at least +1.0
point and the paired-video 95% interval has lower bound strictly above zero.
A null or inconclusive replication ends the job with an explicit report; it
does not authorize the larger scaling branches as a demonstrated success.

## Frozen evaluation cohorts

Reserve three disjoint 64-video cohorts before viewing predictions: replication,
scaling development, and final scaling confirmation. All come from the existing
400 heldout videos already excluded from training. Exclude the previous
32+32 diagnostic and 64+64 pilot videos. Of the remaining 208 videos, only
21 have medium lengths; the original 16/32/16 cohort allocation cannot be
reproduced. Each new cohort has 42 short, 6 medium, and 16 long videos, chosen
by a frozen hash rule. Absolute scores must not be compared across old and new
cohorts as if their distributions were identical.

## Full-source scaling, conditional on replication

The immutable source has 8,343,088 rows. Retaining all prior holdout exclusions
leaves 8,173,782 eligible rows. The existing fixed-batch plan uses 8,173,696 rows
in 63,857 global-128 updates; reserve the final 86 rows for batch alignment,
without relabeling, resampling, or duplicating them into a final batch.

Use the replicated shuffled endpoint as the shared starting state for fresh
and replay scaling. The fresh plan preserves the consumed prefix and shuffles
whole global batches through the remaining full-source windows. The replay
control reuses only batches already seen by step 1792, selecting batches with
exactly matching per-update class counts. It is update- and label-exposure-matched
through step 8192. The fresh plan is an exact permutation of the full-batch
eligible pool; rank assignments within each batch remain intact.

Predeclare cumulative doses at steps 4096, 8192, 16384, 32768, and 63857
(524,288; 1,048,576; 2,097,152; 4,194,304; 8,173,696 presentations).
Evaluate fresh and replay at the first two doses, using the frozen scaling
**development** cohort at +1. Stop for a clear regression if the paired interval's
upper bound is below -1 point against replay or the step-1792 anchor.
At step 8192, require a positive fresh-minus-replay interval and a
fresh-minus-anchor lower bound above -1 before committing to larger doses.
Inconclusive evidence is reported explicitly and stops expansion at this gate.

If that gate passes, extend fresh only through the remaining doses. Stop if a
later dose is clearly worse than the step-8192 fresh model beyond the same
1-point tolerance. Final confirmation compares full-dose fresh, step-8192 fresh,
step-8192 replay and the replication anchor on the untouched final 64 videos.
Larger-dose comparisons have unequal update budgets; they do not isolate data
diversity as the first fresh/replay comparison does. Repeated development looks
are budget gates, not independent confirmation results.

Use fixed step budgets and measured local timings. The full ladder is a
multi-day run on this node, not a 12-hour full-epoch claim. Do not increase the
effective batch to fill memory. Save every 256 updates during scaling and at
every dose endpoint, retaining at least 10% peak reserved-memory headroom.

Entry point: `scripts/run_qwen35_context_replication.py`.
Frozen plans, status, logs, checkpoints and reports:
`outputs/jevany-qwen35-2b-context-replication-seed43/`.
