# JevAny 2B E2-R: reward-aligned RLOO

Requested on 2026-11-08, following
[the reward-alignment plan](../plans/jevany_2b_reward_aligned_rloo_vs_grpo.md).
Scope: the first experiment, E2-R, including the reward audit, correctness and
resume gates, 32-round pilot, and GPT-scored development checks. E2-G remains
a separate subsequent experiment.

The user explicitly requested the eight GPUs on the current node only,
`ip-10-3-120-167`, allocation 38177. This session instruction supersedes the
different node named in AGENTS.md. No other node is used.

## Recipe

- Fresh SFT: `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
- Seven controller/learners and one frozen promoted MoE generator; v10 batches,
  frame cache, exact reference loading, replay and kernels retained.
- Official clip PAUC-F / 100 on complete event windows; -0.25 per fire on
  no-event spans. GPT-4.1 official cumulative-answer rubric for training and
  evaluation, through the existing authorized gateway. No local-score conversion.
- Four rollouts at each of 32 starts per round. One RLOO optimizer step per
  round. The first correctness-gate round counts toward 32 total rounds,
  4,096 trajectories and 131,072 nominal action decisions.
- Raw learned categorical sampling in training and raw learned argmax in
  evaluation; response margin zero. LoRA and pointer learning rates 1e-6 and
  1e-5, KL .02, entropy .001, weight decay .01 and gradient clip 1.

## Development and gates

A deterministic 128-video subset of previously inspected WEB is frozen before
training and checked against SFT-manifest and RL-pool video aliases. It is reused
development, not untouched test. Evaluate fresh SFT and rounds 8/16/32 using
actual generated history and GPT-4.1 `pauc-f-v1`, omega .5. Stop for fewer than
10% of SFT fires at a check, or useful fires below 90% of SFT at two consecutive
checks. Zero fires across all event-containing training trajectories also stops
the run before that batch's optimizer step.

The pilot never extends beyond 32 rounds. If a checkpoint exceeds development
SFT PAUC-F while preserving at least 90% useful coverage, select the highest
PAUC-F checkpoint and evaluate it on all 500 WEB videos. Compare against the
retained matched historical SFT control with paired video-bootstrap intervals.
Do not launch another full-WEB sweep if none qualifies. Since WEB is reused for
development, full-WEB results are not an independent generalization estimate.

## Artifacts and operation

Completed preparation evidence on 2026-11-08:

- The stratified audit covered 128 groups / 512 saved training trajectories.
  The new reward distinguishes 56 of 58 groups with differing response counts
  or useful coverage; 15 groups have a changed top-trajectory set. Comparisons
  combine formula, aggregation and judge changes. Of 67 equal-reward groups,
  63 have identical action-and-answer histories.
- GPT audit usage: 245 unique calls, 54,988 prompt tokens and 245 completion
  tokens. Every request and raw response is retained. Offline replay of all
  512 trajectories reproduced every reward exactly with zero additional calls.
- Fresh 128-video SFT development baseline: **54.1001 PAUC-F**, 63.6678%
  precision, 47.0323 temporal-recall component, 184 useful / 105 wasted fires,
  333 gold turns. This is a subset result, distinct from full WEB's 54.6651.
  The consecutive-check coverage floor is therefore 166 useful fires.
- On the same frozen subset, retained historical GPT judgments give E1 step
  16 **54.8209 PAUC-F / 135 useful fires**, and E1 step 32 **53.0783 / 134**.
  Both retain only about 73% of fresh SFT useful coverage. Retained historical
  SFT evaluation gives 53.3764 / 183; fresh generated history and GPT evaluation
  give the primary 54.1001 / 184 baseline above. These are distinct evaluations,
  and small point-estimate differences should not be treated as decisive.
  Historical subset artifacts require no additional generation or GPT calls and
  are saved under `development/historical-*` in the new run.
- Focused reward, gradient/loss, reference, schedule and workflow tests passed;
  the eight-GPU communication preflight passed before the first training round.
- Round 1 completed in 707.53 seconds: 128 trajectories / 4,096 decisions,
  one optimizer step, zero replay log-probability/probability error and zero
  initial SFT KL. LoRA and pointer gradients were nonzero on every rank.
  GPT scoring used 48 new calls and took 13.32 seconds. All seven learners
  restored step 1 / sampler cursor 32 with zero additional optimizer steps.
- Generated Triton cache files were moved to node-local RAM scratch to recover
  root-disk space. All 137,816 files / 3,588,732,379 bytes passed SHA-256
  comparison. The logical `/tmp/pace-rl-triton` path and kernel contents are
  unchanged; `cache-scratch-migration.json` records the physical location.

- Audit: `outputs/jevany-2b-rl/reward-audit-v1/report.md`, `groups.json`,
  `selection.json`, raw GPT requests/responses and complete-request cache.
- Run: `outputs/jevany-2b-rl/reward-aligned-rloo-v1/`.
- Live state: `workflow-status.json`; per-phase logs: `logs/`.
- Frozen inputs/code: `freeze.json`, `workflow-freeze.json`, `source_snapshot/`,
  `recipe.json`, `development/freeze.json` and `development/selection.json`.
- Final products: `report.md`, `result.json`, `learning-curves.json`,
  `learning-curves.csv`, checkpoints and evaluation predictions/judgments.

Launch or resume using the same wrapper, within the existing allocation:

```bash
srun --jobid=38177 --overlap --nodelist=ip-10-3-120-167 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=none \
  bash scripts/run_jevany_e2_rloo.sh
```

The workflow enforces source hashes on resume. Its JSON status and result files,
rather than this preparation note, record actual execution and outcomes. E1's
historical comparison changes formula, answer aggregation and judge together;
it cannot isolate reward-formula alignment alone.

## User-requested full WEB evaluations, 2026-11-09

The pilot stopped at round 16 after useful development coverage was below 90%
of SFT at rounds 8 and 16. Round 16 scored 54.7754 development PAUC-F versus
SFT's 54.1001, with 153 versus 184 useful fires. The coverage stop remains in
effect for training.

The user subsequently requested full 500-video WEB evaluations. The separate
evaluation-only workflow `scripts/run_jevany_e2_full_web.py` evaluates round 16
first and round 8 second on the original node's eight H200s. It retains each
checkpoint's 128 complete development predictions and cached GPT judgments,
generates the remaining 372 videos, and computes the unchanged official
GPT-4.1 PAUC-F over all 500 videos. Checkpoint tensors and generation/scoring
sources are hashed; prediction reuse validates model identity and repartitions
complete videos into the correct full-WEB shards.

Artifacts are under `outputs/jevany-2b-rl/reward-aligned-rloo-v1/full_web_v1/`:
`status.json`, `freeze.json`, `web-step16/`, `web-step8/`, `comparison.json` and
`report.md`. Comparisons use the retained matched 500-video SFT control and
5,000 paired video-bootstrap resamples, seed 42. The resulting intervals
condition on saved outputs/judgments. WEB includes the reused development
subset and is not an independent test. This request does not resume training
or launch E2-G.
