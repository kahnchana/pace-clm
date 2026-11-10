# E3: RLOO with a fixed supervised training anchor

Status: completed 2026-11-10 at 03:43:29 UTC, 32 rounds / 32 optimizer steps.
All four full 500-video WEB evaluations and GPT judgments completed.

Authorized 2026-11-09 after the coverage audit and E2 RLOO/GRPO comparison.
Use only the original node `ip-10-3-120-167`, allocation 38177, eight H200s.
Output: `outputs/jevany-2b-rl/rloo-sft-anchor-v1/`.

## Evidence and hypothesis

Full 500-video WEB: SFT 54.6651 PAUC-F, 701 useful / 290 wasted fires;
RLOO round 16 54.5561, 605 / 187; GRPO round 16 53.7149, 577 / 191.
Both pilots stopped at round 16 for lost useful coverage.

The training-only audit covers 512 four-rollout groups per run, with saved GPT
judgments and official distinct useful-fire assignments. Among pairs differing
in useful-response count, the more useful trajectory has lower reward in only
23/546 RLOO and 20/456 GRPO pairs. Nevertheless, 388/1704 and 405/1723 useful
responses inherit negative trajectory advantages. All four rollouts miss
131/650 and 140/650 annotated events respectively. No-event spans account for
about 8% and 6% of absolute response-logit score-function signal; this diagnostic
does not measure shared parameter gradients or establish causal attribution.

The reward is mostly ordered sensibly, but shared trajectory credit and limited
exploration can erode useful behavior. Test whether a small training-only
supervised anchor preserves useful responses while RL reduces waste. This is
a hypothesis, not a demonstrated remedy. Audit artifacts are in
`outputs/jevany-2b-rl/coverage-audit-v1/`.

## Frozen pilot

- Fresh original SFT `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`.
- Original RLOO reward, population of spans/draw order, GPT-4.1 judging, seed 42,
  temperature 1 categorical training actions, seven learners / one frozen MoE.
- Same decoder LoRA and pointer head; frozen base/vision/generator/SFT KL target.
  Pointer LR 1e-5, LoRA LR 1e-6, weight decay .01, KL .02, entropy .001,
  gradient clip 1, dropout zero. No answer-token gradients.
- 32 starts × 4 trajectories × 32 decisions = 4,096 on-policy decisions/round.
  One actor pass, one globally accumulated optimizer update per round.
- Add controller-action CE on 126 original SFT examples per round, 42 per
  action. Select 4,032 unique manifest rows deterministically by seeded SHA256,
  restricted to existing RL training-video paths and excluding all registered
  evaluation/validation video aliases. Preserve their original labels/history.
  The class-balanced replay sampling is part of this auxiliary objective.
- Objective: original RLOO + KL + entropy + lambda × mean supervised CE.
  Before the first optimizer step, synchronize RL-objective and unweighted CE
  gradients separately on the first training round. Freeze
  `lambda = 0.15 * norm(g_RL) / norm(g_CE)` for the entire run. Invalid/zero
  calibration fails. This fixes the initial gradient-norm ratio, not AdamW's
  parameter-update ratio. No WEB data selects lambda; no later adaptive tuning.
- Save lambda, optimizer, model, RNG and absolute replay cursor with checkpoints.
  Hash checkpoint files and publish the latest pointer only after all ranks
  save and the round report is complete. Interrupted uncommitted rounds restart
  from the last committed checkpoint; completed updates are not repeated.

**Inference uses raw learned action argmax.** Response margin is zero. No score
offsets, inference temperature calibration, policy mixing, thresholds, cooldowns
or other runtime behavioral heuristics are introduced. All behavioral changes
must come from learned weights.

## Gates and evaluation

Before the pilot, test global loss/reduction equivalence, calibration, deterministic
train-only replay, useful-fire audit attribution and zero-margin policy guards.
The first GPU round checks original behavior replay parity, finite gradients,
zero initial SFT KL, 4,096 RL / 126 supervised decisions, and the 15% calibration.
Restore every learner with zero extra updates and verify the exact frozen lambda.

Maximum 32 rounds / 32 optimizer steps. Evaluate all 500 WEB videos at rounds
8, 16, 24 and 32 with the unchanged GPT-4.1 scorer and generator protocol. Derive
the original frozen 128-video development subset from the same predictions and
judgments. Preserve near-silence stops and two-consecutive-check coverage stops
below 90% of SFT; check both the development subset and full WEB. Full WEB must
retain at least 631 useful responses; the subset must retain at least 166.

Report global PAUC-F, useful/wasted responses, precision, temporal recall, paired
video-bootstrap intervals, RL entropy/KL, CE loss/accuracy by action, calibrated
weight, supervised overhead and core/end-to-end GPU cost. Require higher PAUC-F
and >=90% coverage to label a point improvement. Report whether its paired 95%
interval also excludes zero. One seed and repeatedly inspected WEB are development
evidence, not an untouched test or a training-seed uncertainty estimate.

## Scaling path

The steady-state learner adds CE gradients locally and performs the same single
packed gradient all-reduce as RLOO. Only the initial calibration requires a
second reduction and gradient copy. Supervised replay adds 3.08% observations,
no GPT calls, no extra reference forward, and no inference compute. Measure its
actual token/runtime overhead rather than assuming observation count equals cost.
Keep existing token buckets, frozen visual caching, batched generator brokerage
and complete-request GPT cache. RLOO uses one optimizer step and one actor pass
per collection round versus GRPO's four steps and two passes.

Quality gates precede more compute. If this pilot improves with coverage, repeat
the frozen recipe with additional seeds, then grow the training-video cohort and
collection exposure in stages. Profile generator/learner/GPT time before increasing
batch sizes; preserve global normalization and explicit policy versions. Stay on
this node's eight GPUs for the authorized pilot. If coverage still degrades,
return to event-level credit/exploration diagnosis rather than automatically
expanding this recipe, changing runtime logits, or sweeping WEB repeatedly.

Launch/resume:

```bash
srun --jobid=38177 --overlap --nodelist=ip-10-3-120-167 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=none \
  ../stream-gemma/.venv/bin/python -u scripts/run_jevany_e3_anchor.py
```

Live state: `workflow-status.json`. Frozen recipe: `workflow-freeze.json`.
Training-only selection: `anchor/freeze.json`; measured weight:
`anchor-calibration.json`; quality/cost: `report.md` and `result.json`.
Preparation/launch do not imply that GPU gates passed or quality improved.

## Observed first update

Launched 2026-11-09 at 19:56:55 UTC. The first update completed with 4,096
RL decisions and 126 supervised examples (42/action), zero replay probability
error, zero initial SFT KL and finite gradients. The fixed CE coefficient is
0.0035140820988695176, from RL-objective gradient norm 0.3766456372 and
unweighted CE norm 16.0772696816; their gradient cosine is -0.2731293817.
The supervised phase, including initial calibration, took 48.85 seconds of
a 751.86-second round (6.5%). This single-round phase measurement is not a
steady-state scaling benchmark. Completed WEB quality results follow below.

At 20:17:57 UTC, all seven learners passed restore verification at round 1
with zero additional optimizer updates and the identical frozen anchor
calibration. The complete GPU gate passed, and training resumed toward round 8.

## Completed full-WEB results

| Checkpoint | PAUC-F | Delta vs SFT | Useful / wasted fires | Useful coverage |
|---|---:|---:|---:|---:|
| Retained SFT | 54.6651 | — | 701 / 290 | 100% |
| Round 8 | 52.9117 | -1.7534 | 533 / 165 | 76.0% |
| Round 16 | 54.2602 | -0.4049 | 674 / 279 | 96.1% |
| Round 24 | 54.8260 | +0.1608 | 701 / 280 | 100.0% |
| Round 32 | 54.3782 | -0.2869 | 628 / 212 | 89.6% |

Round 24 is the best observed checkpoint: it preserved baseline useful fires
and removed ten wasted fires. Its paired 95% video-bootstrap delta interval,
[-0.6665, +0.9796], includes zero, so the point gain is not conclusive. Round
32 regressed and fell below the 90% coverage floor at that last check; the run
finished its preset budget, not a two-consecutive-failure stop.

Supervised replay helped retain useful coverage compared with unanchored RLOO
and GRPO, but this one-seed, reused-development result does not establish a
reliable improvement or justify scaling the recipe unchanged. All evaluations
used raw learned argmax, response margin zero and the unchanged GPT-4.1 scorer.
No runtime behavioral heuristics were applied. No automatic extension launched.

Artifacts: `outputs/jevany-2b-rl/rloo-sft-anchor-v1/report.md`, `result.json`,
`workflow-status.json`, and `evaluation/web-step{8,16,24,32}/`.
