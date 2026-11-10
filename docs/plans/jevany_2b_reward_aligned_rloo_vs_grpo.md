# Next two experiments: reward-aligned RLOO versus clipped GRPO

Date: 2026-11-08. Repository: `/fsx/home/kranasinghe/repo/pace-clm`.
Scope: prepare the reward audit, then run experiments E2-R and E2-G below.
No third experiment or automatic extension beyond the pilot is included.

## Motivation and current state

The previous E1 run is `outputs/jevany-2b-rl/profile-7p1-v10`, completed at
32 collection rounds. Its full WEB result at step 32 was 53.0861 PAUC-F,
544 useful / 163 wasted fires, versus the SFT baseline's 54.67, 701 / 290.
It improved selectivity while losing coverage; reward alignment is a hypothesis,
not a proven explanation. Do not continue its trained weights.

The current implementation uses leave-one-out, unnormalized advantages and
one on-policy optimizer update per batch. It is RLOO-style policy gradient,
not standard clipped GRPO. A GRPO comparison must add explicit old-policy ratios,
group normalization and optimization passes, rather than merely rename it.

## Stage 0: reward-ranking audit before new training

Reuse `profile-7p1-v10/batch-*/{rollouts.json,scored.json,report.json}`.
Select approximately 128 complete four-rollout groups from early, middle and
late batches, stratified across single-event, multiple-event and no-event spans.
Only training trajectories are eligible. Preserve IDs and selection seed.

Rescore the **same generated answers** with GPT-4.1 using the intended scorer's
answer aggregation/rubric. Do not generate new trajectories. Cache by complete
request, model and rubric identity; retain raw judgments. Existing local scores
are not automatically equivalent to official judge scores.

Compare the current and PAUC-F-aligned reward formulas under local and GPT
judgments where a documented score conversion permits it. Record:

- Pairwise ranking inversions and top-trajectory disagreements, with ties
  handled explicitly; do not break ties by arbitrary trajectory order.
- Formula-only disagreements with judge held fixed, and judge-only disagreements
  with formula held fixed. Report unavailable comparisons rather than silently
  equate incompatible rubrics or scales.
- Equal-reward groups split into identical action histories, different histories
  with equal outcomes, and ties caused by coarse/noisy judging. Report per stratum.
- Action diversity, response-count variation, useful-event coverage and reward
  range. GRPO does not create reward signal in equal-reward groups.

Write `outputs/jevany-2b-rl/reward-audit-v1/report.md` and machine-readable
group-level results. Freeze a reward specification and judge backend before E2-R.
If the audited reward cannot rank meaningful behavior differences reliably,
repair/calibrate scoring or investigate exploration before launching either run.

## Reward contract to freeze

Preferred event-containing-span reward: judged **clip PAUC-F / 100**, reusing
the existing MarkII four-second-window scoring semantics and omega=0.5.
Reuse the scorer/request builders; test wrapper parity rather than implementing
another approximate PAUC/F formula. Use all sampled fires inside the scored span,
including out-of-window and duplicate fires. Prefix responses are context only.

- Include complete annotated windows only; reject boundary-censored spans.
  Preserve absolute timestamps. Do not invent shorter windows for clip edges.
- Define zero-fire precision as zero, as the existing scorer does. Preserve and
  document PAUC's initial-score convention; report raw coverage alongside F.
- For **no-event** spans, PAUC-F cannot distinguish all firing behaviors usefully.
  Proposed explicit reward: `-0.25 * number_of_fires`. Validate this in the audit
  and freeze its scale. Call the combined objective PAUC-F-aligned, not pure
  PAUC-F on every span. No positive reward for waiting.
- Start without extra miss/duplicate penalties on event-containing spans;
  precision and temporal recall already enter F. Any audit-driven modification
  must be written down and applied identically to both runs.
- Per-clip F is a surrogate: its average is not benchmark-global PAUC-F.
  Keep global judged PAUC-F for development selection and final reporting.
- Use GPT-4.1 training judgments if the local judge misranks trajectories and
  calibration cannot resolve it. Otherwise freeze the calibrated local judge.
  Record the decision; never silently use different judges across E2-R/E2-G.

## Shared experiment settings

| Item | Both runs |
|---|---|
| Fresh initialization | `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243` |
| Controller | Qwen3.5-2B, original pinned revision and SFT architecture |
| Trainable weights | Existing decoder LoRA across 24 layers + pointer head |
| Frozen weights | Original base, vision, SFT reference, MoE answer generator |
| Generator | Promoted PACE MoE, existing greedy forced-response path, 96-token cap |
| Inputs | J1 history; actual emitted answers; 1 FPS; 32 causal frames; 144 visual tokens/frame |
| Span pool | Existing audited 10% training cohort; same persisted draw schedule |
| Rollout batch | 32 starts × 4 rollouts = 128 trajectories/collection round |
| Pilot exposure | 32 collection rounds; 1,024 starts; 4,096 trajectories; 131,072 nominal decisions at 32 steps/trajectory |
| Sampling | Same seed 42, temperature 1.0, categorical three-action policy |
| AdamW | Pointer LR 1e-5; LoRA LR 1e-6; preserve existing weight-decay setting |
| Regularization | Exact three-action SFT KL coefficient 0.02; entropy bonus 0.001; global grad clip 1.0 |
| Infrastructure | Preserve working 7-learner / 1-generator configuration from v10; do not rebuild serving for this comparison |

Pin the actual source revision/diff, input/checkpoint hashes and all numerical
defaults. Keep rollout/replay dropout disabled consistently. Do not change
sampling, frame rate, resolution, input history, supervised replay or data
mixture while testing the reward/optimizer changes. No supervised CE term in
either of these two runs.

### E2-R: corrected reward with the existing RLOO update

Output: `outputs/jevany-2b-rl/reward-aligned-rloo-v1/`.

- Replace the current reward ledger's scalar return with the audited reward.
- Keep `A_i = r_i - mean(r_j, j != i)`; no standard-deviation normalization.
- Give each sampled controller action the trajectory's scalar advantage.
  No delayed/return-to-go credit in this comparison.
- One pass over each batch; accumulate the complete batch gradient and perform
  **one optimizer step per collection round: 32 optimizer steps total**.
- Retain the existing exact-state replay and SFT-reference KL.

### E2-G: clipped GRPO with the same corrected reward

Output: `outputs/jevany-2b-rl/reward-aligned-grpo-v1/`.
Start afresh from SFT, not from E2-R's trained checkpoint.

- Group advantage: `(r_i - group_mean) / (group_std + 1e-6)`; specify population
  standard deviation (`correction=0`). Equal-reward groups have zero reward
  advantage but retain defined KL/entropy contributions and denominator weight.
- Snapshot the behavior policy once per collection round. Store its selected
  action log-probabilities and policy version. Keep these **fixed across all
  minibatches and passes**; do not refresh them after optimizer updates.
- On exact saved observations, compute per-action ratio
  `rho_t = exp(log_pi_current(a_t|h_t) - log_pi_behavior(a_t|h_t))`.
  Use `-min(rho_t*A_i, clip(rho_t,0.8,1.2)*A_i)` plus the same SFT KL/entropy.
  Apply the loss to controller actions only; no MoE answer-token gradients and
  no product of ratios across the whole trajectory.
- Two optimization passes per collected batch. Each pass has two **global
  minibatches of 64 trajectories**, giving **four optimizer steps per round,
  128 total**. Physical GPU microbatches only control memory; accumulate each
  global minibatch before synchronizing/stepping.
- Shuffle whole trajectories reproducibly per pass; preserve each trajectory's
  observations/actions and original group advantage. Mask padding and retain
  the existing fixed-horizon loss normalization (32), without per-rank weighting
  errors. Handle uneven seven-rank work with synchronized schedules.
- The behavior policy is the ratio denominator; the frozen original SFT policy
  is the KL reference. They are different objects after training starts.

This compares a conventional normalized/clipped multi-pass GRPO package with
RLOO, not clipping alone. Both collect identical exposure; GRPO performs two
actor passes and four times as many optimizer steps. Report extra learner time,
replayed decisions and GPU-hours explicitly. Exact trajectories diverge once
policies differ, even with the same starting-span schedule/seeds.

## Implementation, gates and run order

Existing entry points are `scripts/run_jevany_rl.py`,
`scripts/jevany_rl_workers.py` and `src/pace_clm/rl.py`, with associated judge,
broker and reference helpers. The current runner **does not expose the reward
switch or GRPO controls above**. Implement explicit versioned configuration and
launch wrappers first; do not claim an existing command already runs E2-G.

Create separate launch/resume wrappers for E2-R and E2-G that serialize the
settings above, preserve v10 runtime/kernel configuration, and use the same
`--starts 32 --rollouts 4 --updates 32 --full-run` collection budget. Ensure
`updates` means collection rounds in both modes, with a separate optimizer-step
counter. Use new output paths; never overwrite E1 or SFT artifacts.

Before either full pilot, pass reward-wrapper parity, likelihood replay parity,
trainable/frozen parameter checks, one complete round and save/resume tests.
For GRPO additionally test: ratios start near 1; clipping activates after policy
changes; negative advantages use the correct clipping branch; zero-advantage
groups are finite; behavior log-probs survive resume unchanged within a batch.
Checkpoint optimizer/minibatch/pass cursors so a resume cannot repeat an update.

Run order: **audit → E2-R gate/pilot → E2-G gate/pilot**. Profile E2-G's complete
round before committing to its larger replay cost. Keep rollout/replay parity
checks active; reject nonfinite ratios/rewards and incomplete judge batches.

## Development evaluation and acceptance

Freeze a video-disjoint, non-training development cohort before training. Use
the same cohort and GPT-4.1 rubric for SFT and rounds **8/16/32**, with greedy
controller decisions, zero response margin and actual MoE-generated history.
Label any previously inspected cohort as reused development, not untouched test.

Report global judged PAUC-F, precision, temporal recall, useful-event coverage,
useful/wasted fires, and Hold/Silence diagnostics. Also log reward/tie rates by
stratum, action diversity, entropy, SFT KL, gradient norms, and GRPO ratio/clip
statistics by pass, alongside end-to-end timing and cost.

Pre-register coverage protection: pause for near-silence immediately, or useful
coverage below 90% of SFT at two consecutive checks. Do not call a gain from
fewer fires alone. Extend neither pilot beyond 32 rounds automatically.

Select the better recipe/checkpoint on judged development evidence. Only then
run its full 500-video WEB evaluation with the unchanged GPT-4.1 `pauc-f-v1`,
omega=0.5 scorer and matched SFT control; report paired video-bootstrap intervals.
If neither run improves, stop and diagnose reward/exploration instead of
automatically spending on a third run or another WEB sweep.

Deliver per-run frozen config, audit report, reward/judge specification, learning
curves, checkpoints, dev predictions/judgments and a side-by-side result report.
Separate conclusions about reward alignment (E1 vs E2-R) from optimizer package
(E2-R vs E2-G); qualify historical E1 comparisons if any other setting changed.
