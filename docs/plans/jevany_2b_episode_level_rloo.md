# Episode-level PAUC-F with continuous self-generated history

Date: 2026-11-10. Status: implementation and GPU update/restore gate passed; bounded pilot running.
Hardware: only the eight H200s on `ip-10-3-120-167`, original allocation 38177
while available. Do not allocate another node automatically.

## Motivation and controls

The current evaluator already generates continuously over full videos. Training
instead samples independent 32-second clips, uses a supplied prefix, and scores
each clip independently. Averaging clip PAUC-F is not equivalent to episode
PAUC-F; resets also omit the effects of the model's earlier decisions/history.

Full 500-video WEB retained SFT: 54.6651 PAUC-F, 701 useful / 290 wasted fires.
Reward-aligned RLOO round 16: 54.5561, 605 / 187. Fixed GRPO round 16: 53.7149,
577 / 191. Anchored RLOO's best observed checkpoint, round 24: 54.8260,
701 / 280; paired delta +0.1608, 95% interval [-0.6665, +0.9796]. Round 32
regressed to 54.3782, 628 / 212. The small round-24 gain is not conclusive.

Hypothesis: training on complete episodes with the same continuous history used
at evaluation improves useful coverage and precision accounting. Longer-horizon
credit assignment and rollout variance remain risks. Test this horizon change
before adding another optimizer or changing the official scorer.

## 1. Aggregation audit and episode manifest

- Use saved full-video predictions and GPT judgments to compare mean clip,
  episode and benchmark-global aggregation/rankings. Preserve complete event
  windows and all physical fires. This audit isolates aggregation effects; it
  cannot reproduce changes caused by resetting generation history or GPT's
  cumulative-answer context.
- Reconstruct one episode per video-request pair from original annotated
  training records. Use the complete trusted annotated interval, including
  intervening quiet sections. Never stitch independently generated old clips
  into a purported continuous trajectory or treat unannotated tails as negatives.
- Restrict the pilot to the existing training-video cohort. Reapply every
  registered evaluation/validation exclusion, deduplicate video/request identity,
  and record annotation, video, split and manifest hashes.
- Inspect lengths, event density, quiet duration and true event-free examples.
  Freeze eligibility and sampling before training. Do not silently sample only
  short or event-rich episodes to make throughput look better.

## 2. Continuous causal rollout and episode reward

- Begin each episode from the same causal initialization used by evaluation.
  Carry actual generated answers, recent history and last-response memory to
  the end. Retain the existing 32-frame visual context at 1 FPS / 144 visual
  tokens per frame. Longer reward horizon does not require longer model context.
- Process in bounded chunks for scheduling/memory, without resetting behavioral
  state. Preserve absolute timestamps, episode position and exactly the memory
  visible at every decision. Never expose future frames or annotations to policy
  inputs. All four rollouts for an episode use the same initial state and policy.
- Score all episode responses and gold events jointly with the unchanged
  GPT-4.1 cumulative-answer rubric and `pauc-f-v1`, omega .5. Event-containing
  episode reward is episode PAUC-F / 100. Keep every physical response in the
  precision denominator, including duplicates and responses in quiet regions.
- Quiet regions inside event-containing episodes need no separate firing penalty.
  Genuine event-free episodes retain the explicit existing `-.25 * fires`
  treatment for this initial pilot, documented as an exception to pure PAUC-F.
  Audit its duration-dependent scale before freezing the recipe. If full-episode
  scale is unsuitable, stop preparation and document the revised reward contract
  rather than silently changing it during training.
- Mean episode PAUC-F still differs from benchmark-global PAUC-F. Preserve the
  global benchmark scorer and report both; do not claim exact objective alignment.

## 3. Eight-GPU execution and learning

| GPUs | Work |
|---|---|
| 0–6 | Resident controller learners: original decoder LoRA and action head |
| 7 | Shared frozen promoted answer generator |

- Start fresh from the original SFT checkpoint,
  `outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243`. Use RLOO with four
  independent full-episode trajectories per group, one actor pass and one
  globally accumulated optimizer update per rollout batch. Do not update the
  policy partway through a collected episode/group.
- Preserve base/vision/generator freezing, dropout zero, pointer LR 1e-5,
  LoRA LR 1e-6, weight decay .01, SFT KL .02, entropy .001, grad clip 1 and
  categorical training sampling at temperature 1. Freeze exact loss scaling
  and variable-length normalization before the pilot. Target expected episode
  return; do not divide each trajectory's policy-gradient sum by its own length
  without explicitly acknowledging the resulting objective change.
- Keep the supervised training anchor. For the controlled comparison, retain
  its frozen coefficient 0.0035140820988695176 and original selected replay
  schedule, targeting 126 CE examples per 4,096 RL decisions rather than per
  variable-length episode batch. Specify deterministic replay-credit accounting
  before launch; do not tune this coefficient on WEB.
- Batch controller work by compatible shapes and generator requests across
  active episodes. Keep only rolling vision/history state on GPU; persist exact
  replay observations/features off GPU as needed. Do not retain an unbounded
  full-video autograd graph. Reuse the complete-request GPT cache.
- Choose episode-group concurrency from a measured GPU gate. Account for long
  episodes, uneven seven-rank work and synchronization without assigning extra
  weight to ranks with fewer observations.

## 4. Correctness and performance gates

- Verify continuous versus chunked execution preserves observations, memory,
  generated history and action probabilities within the existing numerical
  tolerances. Chunk boundaries must not change behavior or duplicate fires.
- Match the standalone official episode scorer exactly, including complete event
  windows, missed events, zero fires, duplicates and event-free handling.
- Require rollout/replay parity, finite gradients, expected trainable/frozen
  parameters and zero initial SFT KL. Test variable-length loss aggregation and
  unequal rank work against a single-process reference.
- Save model, optimizer, RNG, policy version, episode/group identity, frame
  cursor, pending generator state and behavioral memory. Either restore exactly
  at a supported boundary or explicitly restart an uncommitted batch from its
  last committed state. Never silently skip or repeat a committed update.
- Pass a complete small rollout/update and zero-extra-update restore before
  pilot training. Profile decisions/sec, vision/CPU preparation, generator wait,
  learner/GPT time, padding, memory and GPU-hours; freeze concurrency afterward.

## 5. Exposure budget, evaluation and stopping

- Maximum pilot exposure: approximately the previous 131,072 RL decisions,
  with full WEB evaluation near 32,768 / 65,536 / 98,304 / 131,072 decisions.
  Finish complete batches/episodes at evaluation boundaries and report actual
  exposure. Freeze a draw schedule that respects the total budget without
  truncating episodes. Record optimizer steps and episode count separately;
  do not compare episode rounds with old 32-second clip rounds directly.
- Use all 500 WEB videos and unchanged GPT-4.1 scoring at each milestone.
  Derive the same frozen 128-video development subset from those saved outputs
  and judgments. WEB remains reused development, not untouched test data.
- Compare against retained SFT, anchored clip RLOO round 24, and old checkpoints
  at matched sampled-decision exposure. Report global PAUC-F, precision, temporal
  recall, useful/wasted fires, episode returns, coverage and paired bootstrap
  intervals. Separate decision-matched from GPU-hour-matched comparisons.
- Preserve immediate near-silence stops and two-consecutive-check stops below
  90% of SFT useful coverage, on both full WEB and the development subset.
  Acceptance requires higher global PAUC-F plus at least 631 useful responses
  on full WEB (166 on the subset). Distinguish a point gain from a paired 95%
  interval excluding zero; additional seeds are needed before claiming scaling.
- No automatic extension beyond the frozen pilot exposure. If credit assignment
  remains the bottleneck, plan denser credit or value estimation separately;
  do not bundle it into this horizon comparison.

## Inference constraint and deliverables

Inference remains raw learned action argmax with response margin zero. No token
or action logit offsets, score mixing, thresholds, inference-temperature
calibration, cooldowns or other runtime behavioral heuristics are permitted.
All improvements must arise from training learned weights.

Deliver the aggregation audit, frozen episode/data/reward specification, source
snapshots, gate and resume results, exposure/cost curves, checkpoints, full WEB
predictions/judgments, paired comparisons and a concise result report. This plan
does not by itself establish that the correctness gates or pilot completed;
consult the execution record and run artifacts below.

## Execution record (2026-11-10)

The user authorized implementation and the bounded pilot on the original node.
The implementation now lives in `scripts/run_jevany_episode_workflow.py`, with
episode collection/learning in `scripts/run_jevany_episode_rloo.py` and
`scripts/jevany_rl_episode_workers.py`. Historical experiment sources are retained.
The active run directory is `outputs/jevany-2b-rl/episode-rloo-v2`; its
`workflow-status.json` is the current execution state, and its frozen manifests,
source snapshots, logs and reports are the evidence for subsequent results.

Preparation retained 8,321 video/request episodes (median 67 seconds, maximum
179), with all 8,321 video contents hashed. It rejected 3,132 records whose
complete event windows exceed the trusted annotated interval, excluded 445
evaluation records, and rejected one captioned record without retained events.
No genuinely event-free episodes remain, so the existing explicit no-event
penalty is inactive. No physical video tail is assumed to be annotated silence.

The frozen retry draw schedule has 29 complete-batch updates and 126,376
sampled decisions, with WEB evaluations at 34,636 / 69,744 / 99,832 / 126,376.
The first failed gate collected 4,224 decisions without committing an optimizer
update. Those decisions count against the original cap: total planned
collection across both attempts is 130,600, below 131,072. The retry starts
fresh from SFT; rejected work is not counted as learned exposure.
Supervised replay uses the original row order and coefficient
0.0035140820988695176, with cumulative credits
`floor(sampled_decisions * 126 / 4096)`. Policy-gradient and regularization sums
use the global trajectory count times the constant 32 as their denominator;
individual trajectories are never divided by their own lengths. This retains
the prior constant loss scale while targeting expected episode return.

The aggregation audit reproduced official global scores and preserved all
physical fires and complete event windows. Global PAUC-F ranks anchored clip
RLOO round 24 above SFT, whereas mean episode PAUC-F ranks SFT above it. Thus
episode reward is explicitly not an exact substitute for global model selection.
The 36 focused CPU tests passed. The corrected full-batch GPU gate committed
one update on 4,224 decisions with 129 supervised replay examples, zero replay
probability error, zero initial SFT KL, and a maximum learner reservation of
46.64 GiB. All seven learners restored the checkpoint without another update.
The measured batch took 1,470.48 seconds, suggesting approximately 12.2 core
node-hours for the remaining frozen recipe's entire committed exposure, before
evaluation and restart costs. This first-batch projection is not a completion
time guarantee. Pilot quality results remain pending.

The failed gate is retained in `outputs/jevany-2b-rl/episode-rloo-v1`. It stopped
on a replay probability mismatch of 0.0309185, above the unchanged 0.005
tolerance. A diagnostic reproduced the same discrepancy in evaluation and
training modes. Replay now preserves complete collected batches, row positions
and padding shapes, rather than regrouping observations. A no-update GPU proof
on 70 saved observations, including gradient-enabled replay and early and late
episode positions, measured zero probability error for the preserved layout.
The repeated full rollout/update and restore gate passed, and the pilot has
continued toward its first full-WEB milestone. Inference and raw learned action
scores are unchanged.

Postprocessing reports episode and benchmark-global scores separately, and
compares both decision exposure and GPU cost. The matched-control workflow
freezes all 32 available historical anchored checkpoints before new WEB results
are available. It selects the nearest checkpoint by exposure or cost, reports
signed differences, and evaluates missing historical controls after the pilot
using only the same original eight-GPU allocation. It never selects references
by their quality score or silently claims exact budget equality.
