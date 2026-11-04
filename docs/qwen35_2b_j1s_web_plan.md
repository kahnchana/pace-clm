# Qwen3.5-2B J1-S controller: matched 10% training, full WEB evaluation

Date: 2026-11-03. Repository: `/fsx/home/kranasinghe/repo/pace-clm`.

## Objective and scope

Train `Qwen/Qwen3.5-2B` using the existing JevAny/J1-S setup to choose
`Silence`, `Hold`, or `Response`. On `Response`, use the frozen promoted
PACE Qwen3.5-35B-A3B model to generate the answer. Complete training, evaluate
all 500 WEB videos, and report judged results. This is supervised J1-S
classification, not RL. Run one seed (42); EGO and model-size sweeps are outside
this experiment. Proceed through implementation, training, WEB judging, and
reporting without another routine confirmation.

## Exact data and training contract

- Reuse `outputs/jevany-j1s/manifests/train.jsonl` byte-for-byte, with its
  `train.summary.json` and `audit.json`. This already represents the existing
  10% training exposure; **do not take another 10% sample**.
- Expected SHA-256:
  `89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222`.
  Exactly 263,760 presentations: 215,705 Silence, 6,835 Hold, 41,220 Response.
  Preserve the J1-S negative replacement and row order; do not rebalance classes,
  relabel targets, add mined examples, or train on validation/WEB.
- Preserve J1 history, including prior response text and last-response memory;
  use the existing candidate descriptions, decision marker, 256-wide q/k pointer
  head, and action cross-entropy. `Standby` maps to `Hold`. Do not use J2's
  response-blind inputs or J1-S-R rollout-mined data.
- Keep 1 FPS, 32 causal frames, 384px / 144 visual tokens per frame, direct
  generation prefix, and last-response v2 memory with the existing aged-out
  selection. Training uses the manifest's gold history; evaluation uses actual
  controller decisions and generated MoE answers in history/memory.
- Initialize from the official post-trained 2B checkpoint and pin its resolved
  revision. Create a **fresh** LoRA (r64, alpha128, dropout 0.05, bias none) on
  the corresponding language-model projection modules used by J1-S; freeze
  vision and base weights. Record matched module names and trainable counts.
  Initialize the pointer head for the model's actual hidden width. Never load
  the 35B adapter or pointer weights into 2B.
- Match `scripts/sbatch_jevany_j1s_full.sh` and
  `scripts/train_jevany_streaming.py`: AdamW, pointer LR `1e-4`, LoRA LR `2e-6`,
  gradient clipping 1.0, one epoch, seed 42, eight GPUs, one example per GPU
  microbatch and accumulation 4 (effective global batch 32), 8,243 updates
  including the final partial batch. Preserve the original loss scaling and
  optimizer defaults. Save/resume every 100 updates. Use the final one-epoch
  checkpoint for WEB; do not select checkpoints or margins on WEB results.

## Required implementation and preflight

Adapt the existing trainer/evaluator with explicit model identity in checkpoint
metadata. Use a new output root, `outputs/jevany-qwen35-2b-j1s/`; preserve old
artifacts. Reference `configs/jevany_streaming.yaml` and
`docs/exp/jevany_response_blind_next_three.md` (J1-S completed result).

The current evaluator loads controller and generator adapters onto one MoE
backbone. **That cannot work across 2B and 35B.** Load separate runtimes:
2B plus its trained controller adapter/head, and the frozen promoted MoE
generator from `../stream-gemma/outputs/lora-qwen-w1-long-neg`. Preserve its
pinned base revision and forced-response behavior. Each runtime uses its own
processor and vision features; do not share model-specific embeddings. Both see
the same causal frames and actual conversation history. Only the 2B controller
chooses whether to fire; the MoE supplies answer text.

Before the full run, verify manifest hash/counts and train/eval disjointness,
model/module loading, three-class gradients and a tiny overfit, full 32-frame
forward/backward memory, checkpoint resume, and a few **training-video**
controller-to-MoE handoffs. Confirm vision is frozen and no future frames enter
either model. Check available H200 memory for both models during evaluation;
use existing FLA support as appropriate without shortening the input window.
Save source revision/diff, dependencies, model/adapter hashes, settings, and
manifest provenance. Then launch the matched full run on one eight-H200 node.

## WEB evaluation and deliverables

After training, run the existing WEB adapter on all 500 videos, with
`response_margin=0.0`, promoted MoE answers, greedy generation and the existing
96-token answer cap. Preserve benchmark sampling and scorer behavior. Adapt
`scripts/eval_jevany_streaming.py` for separate runtimes, then use
`scripts/merge_jevany_benchmark.py` and `scripts/score_exp10_web_ego.py web`.
Use the same GPT-4.1 judge/rubric and `pauc-f-v1`, omega=0.5, as J1-S.
Ensure exactly 500 unique expected IDs, all shards merged, and judging complete.
Report the full WEB result even if the model performs poorly; do not stop at
timing-only scores or use a performance gate to omit semantic judging.

Write `outputs/jevany-qwen35-2b-j1s/reports/web_result.md` and a machine-readable
summary with checkpoint/provenance, judged PAUC-F, precision and recall
components, total/useful/wasted fires, fires/video, decision latency, generator
latency, peak memory, and training duration. Compare against historical WEB
J1-S MoE (PAUC-F **56.51**, precision **65.36%**, recall **49.77**, 1,357 fires,
887 useful / 470 wasted) and promoted PACE token policy (PAUC-F **57.47**).
Report deltas; bootstrap paired global PAUC-F against stored J1-S predictions
if available. State whether 2B reaches 56.51 and show the measured quality/cost
tradeoff. The MoE J1-S controller started from a PACE-trained adapter, whereas
this 2B starts from official weights: record this initialization difference,
so results are not attributed solely to parameter count. Return WEB numbers
and links to the report, checkpoint, merged predictions, and judged summary.
