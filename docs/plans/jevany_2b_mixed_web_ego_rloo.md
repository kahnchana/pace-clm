# Mixed WEB/EGO episode RLOO pilot

Date: 2026-11-11. Fresh initialization from original J1 SFT step 8243.
Run: `outputs/jevany-2b-rl/mixed-web-ego-rloo-v1`.

Routing update, 2026-11-12: the user requested final step-17 evaluation on
allocation 38177 after training releases its eight GPUs. The second-node worker
was stopped while idle. `eval_jevany_final_local.py` waits for the trainer to
exit, then acquires the original training GPU lock and evaluates the final
checkpoint with the same protocol. `final-evaluation-routing.json` records the
authorization, entrypoint hash, and routing details; earlier results are retained.

This experiment tests whether the controller can learn WEB and EGO behavior
when those domains enter training. It uses four continuous causal trajectories
per episode and official episode PAUC-F rewards. Seen-domain improvement and
heldout transfer are reported separately.

## Frozen training recipe

- Decision exposure targets: 50% existing training data, 25% WEB, 25% EGO.
  Select whole episodes uniformly within each source, using deterministic
  deficit scheduling. Finish all four trajectories; never truncate to fill a
  budget. The frozen schedule contains 99,964 decisions in 17 updates:
  50,952 ours, 24,464 WEB, 24,548 EGO.
- Source objective: `.5 * mean ours episode loss + .25 * mean WEB episode loss
  + .25 * mean EGO episode loss`, with the existing constant-32 normalization.
  This avoids weighting short WEB episodes by their larger episode count.
- Original head/LoRA learning rates 1e-5/1e-6, KL .02, entropy .001, weight
  decay .01, gradient clip 1, no dropout, one update per collected batch.
  Original supervised replay manifest and coefficient .0035140820988695176;
  credits accrue at 126 examples / 4096 decisions. Replay affects weights only.
- Keep original collected peers, order and padding during likelihood replay.
  Freeze source/artifact hashes, schedule, split, media hashes, and runtime
  versions. Preserve committed model, optimizer, RNG and sampler state.
- Original ours trusted intervals retain their four-second answer windows.
  WEB/EGO preserve original reply/related timespans without clipping or replacing
  them. Sample their physical videos with canonical end-aligned 2/6 stride to
  1 FPS; use native timestamps for responses and response memory. Policy inputs
  never include future frames or gold answers.

## Data and overlap policy

Keep 8,321 existing training episodes. Source-video groups, seed 42, give WEB
400 train / 100 heldout and EGO 145 train / 181 heldout. EGO's larger heldout
fraction preserves existing protected evaluation/validation video exclusions;
only the explicit WEB/EGO training identities are newly allowed. Holdouts never
enter RL. The existing replay anchor remains unchanged.

Also freeze 128 ours training episodes, 128 complete trusted ours validation
episodes, and the previous 20 perfect episodes with the most annotated responses.
The validation source yields 138 eligible episodes; 62 incomplete event windows
are rejected. The perfect20 cohort is a selected diagnostic, not a test set.
All historical benchmarks have already informed earlier experiments; no
untouched-test claim is made. Combined full-WEB scores after this experiment
include training examples and are not a new heldout quality floor.

## Nodes and correctness gates

Train only on allocation 38177, `ip-10-3-120-167`, seven learners and one frozen
answer generator. Evaluate only on allocation 38226, `ip-10-3-70-177`, eight
evaluation workers. No additional allocations. GPT-4.1 judging uses the
previously authorized gateway, with distinct training/evaluation caches.

Before the pilot continues past update 1, require a complete mixed update,
finite gradients, memory below 115 GiB, immutable checkpoint verification,
exact optimizer/RNG restore, cross-node collected-batch replay, raw-argmax
inference smoke test, and full original-SFT cohort baselines. Replay tolerances
remain probability .005, max KL 1e-4, mean KL 1e-5, selected-logp .05.

## Evaluation schedule

| Stage | Evaluation |
|---|---|
| Before continuation | Original SFT on all seven cohorts; first-batch smoke |
| Reference | Previous episode-RL final step 23 on all seven cohorts |
| Every update | Per-domain exposure, reward mean/std, within-four range/flatness, nonzero advantages, fires, KL, entropy, parity, head/LoRA gradients, supervised replay, throughput and GPU cost |
| Update 1 | Training-node restore; independent evaluation-node replay and first-batch inference |
| Approximately every 10k decisions | Fixed 32-episode subsets of six main cohorts, plus all perfect20 |
| First update at/after 25k, 50k, 75k, final budget | Full WEB train/heldout, EGO train/heldout, ours train/validation, perfect20 |
| Finish or coverage stop | Final pending evaluations and JSON/CSV/Markdown report |

Every inference uses the raw learned argmax with response margin zero. Report
global PAUC-F, mean episode PAUC-F, useful/wasted/total fires, precision,
temporal recall, and paired bootstrap differences against SFT. Bootstrap
intervals are conditional on saved judgments and reused cohort selection.

Queue only atomically committed checkpoints. Never drop a full evaluation;
supersede stale fast probes. Wait at update boundaries once training is 10k
decisions ahead of an unresolved full evaluation (whole-update overshoot is
possible). Stop on heldout near-silence (<10% of SFT fires), or two consecutive
full evaluations with useful fires below 90% of SFT in any heldout domain.
Never repair a regression with inference-time behavioral heuristics.

## Entrypoints

Run `prepare_jevany_mixed_rl.py`, then
`run_jevany_mixed_workflow.py --prepare-only` to freeze the workflow and queue
baselines. Start `eval_jevany_mixed_queue.py --run <run>` on the assigned eval
node and `run_jevany_mixed_workflow.py` on the training node. The workflow runs
the first-update/restore gate and resumes the pilot automatically after the
remote checks and baseline complete. Inspect `workflow-status.json`,
`evaluation-status.json`, `latest.json`, and batch reports for live progress.
