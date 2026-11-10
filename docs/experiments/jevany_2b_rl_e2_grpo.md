# JevAny 2B E2-G: normalized, clipped, two-pass GRPO

User-authorized on 2026-11-09, to start only after both full 500-video E2-R
evaluations finish. The [experiment plan](../plans/jevany_2b_reward_aligned_rloo_vs_grpo.md)
defines this second optimizer comparison. The original current-node instruction
continues to apply: eight H200s on `ip-10-3-120-167`, allocation 38177. No other
node is used.

## Frozen comparison

- Fresh original SFT: `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
- Train the existing 24-layer decoder LoRA and pointer head. Keep base, vision,
  SFT reference and promoted answer generator frozen. Seven learners and one
  resident generator; retain v10 input/replay kernels and numerical settings.
- Same persisted training-span order, seed 42, categorical temperature 1,
  32 starts × 4 rollouts × 32 decisions per collection round. Same GPT-4.1
  cumulative-answer judging and audited PAUC-F-aligned reward as E2-R.
- Group advantages use population standard deviation plus 1e-6. Exact
  equal-reward groups have zero reward advantage and retain KL/entropy terms.
- Fixed selected-action behavior likelihoods for the whole collected batch;
  per-action ratio clipping at 0.8/1.2. The original SFT policy remains the
  separate frozen KL reference. No answer-token gradients or supervised CE.
- Two passes, each with two global 64-trajectory minibatches: four optimizer
  steps per collection round. Shuffle complete trajectories deterministically;
  preserve original group advantages and exact observations/actions. Weight
  uneven seven-rank gradients by the global minibatch and fixed horizon 32.
- Maximum 32 collection rounds, 128 optimizer steps, 131,072 sampled decisions
  and 262,144 actor-replayed decisions. Report verification forwards and cold
  restore overhead separately from actor replay and core training time.

## Gates and operation

The dependency gate requires both prior full-WEB results and complete GPT
judgment coverage; a failed prerequisite does not start training. CPU tests
cover clipping signs, policy-ratio changes, population normalization, zero
advantages, global rank weighting, deterministic whole-trajectory schedules,
optimizer restoration, immutable behavior data and committed-update recovery.

The GPU gate deliberately commits optimizer step 1 inside round 0, restores
all seven learners with zero extra updates, and checks the saved behavior,
observations, scores, shuffle and parity hashes. It then resumes the remaining
three updates and verifies a completed-round restore at round 1 / optimizer
step 4. This first round counts toward the pilot. A full behavior likelihood
parity forward precedes any updates in every new collected batch; subsequent
optimizer passes measure ratios against the same saved behavior probabilities.

Every optimizer step saves model/AdamW/RNG state and a pass/minibatch cursor
in a staging directory. The driver publishes a hash-verified commit by atomic
rename, then advances the latest pointer. Resume adopts a committed update if
the pointer write was interrupted; incomplete staging directories are ignored.

Evaluate all 500 WEB videos with GPT-4.1 at rounds 8/16/24/32, greedy raw actions,
response margin zero and actual generated history. Derive the same frozen
128-video development cohort from these predictions and judgments, using its
fresh SFT baseline for coverage checks. Stop for near silence or useful coverage
below 90% of SFT at two consecutive checks. Primary exposure-matched comparisons
with E2-R are rounds 8 and 16 because RLOO stopped at round 16. No extension
beyond 32 rounds is automatic.

Launch/resume within the original allocation:

```bash
srun --jobid=38177 --overlap --nodelist=ip-10-3-120-167 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=none \
  ../stream-gemma/.venv/bin/python -u scripts/run_jevany_e2_grpo.py
```

The original v1 stopped at its first-round GPU gate because shuffled replay
changed the visual inputs and initial action ratios. Its artifacts are retained.
The fixed v2 starts fresh from SFT, retains each trajectory's original learner,
and persists the exact frozen visual embeddings used during collection for all
replay and checkpoint restores. An initial-ratio check now runs before the first
optimizer update as well as at the completed-round gate.

Active artifacts: `outputs/jevany-2b-rl/reward-aligned-grpo-v2/`. Live state is
`workflow-status.json`; the explicit recipe/source/input hashes are in
`workflow-freeze.json` and `freeze.json`. See `cpu-validation.json`,
`midbatch-resume-verified.json`, `gate-passed.json`, `optimizer_checkpoints/`,
`batch-*/report.json`, `evaluation/`, `learning-curves.csv` and `report.md` for
actual gate outcomes, checkpoints, cost, clipping diagnostics and results.
Preparation and launch do not imply that the GPU gates or pilot have completed.
