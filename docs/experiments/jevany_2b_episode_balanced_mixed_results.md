# Episode-balanced WEB/EGO RLOO: results and experiment record

**PRELIMINARY — awaiting step 32 and final judging.** Updated 2026-11-13T14:38:43.204832+00:00.

PRELIMINARY: latest completed full evaluation is step 30.

Full WEB PAUC-F: 56.56 versus SFT 54.69 (+1.87 points).

Full EGO PAUC-F: 42.83 versus SFT 42.73 (+0.10 points).

WEB held out PAUC-F: 55.56 versus SFT 55.73 (-0.17 points).

EGO held out PAUC-F: 42.06 versus SFT 42.48 (-0.42 points).

Training/final judging are incomplete; this is not the final result.

## What was tested

The experiment tested whether changing 50/25/25 original/WEB/EGO sampling from decision counts to episode counts improves EGO learning while retaining WEB and original-data quality. It began fresh from Qwen3.5-2B J1-S SFT step 8243. The continuation restores the same learned run, rather than restarting from SFT.

Each update collects four original episodes, two WEB episodes and two EGO episodes, with four complete causal trajectories per episode. The full 32-update schedule contains 138,460 decisions and 128/64/64 distinct episodes. The original 23-update schedule prefix is preserved exactly.

This is a practical sampling comparison, not an isolated causal estimate of episode diversity: source decision shares, identities, optimizer-update counts and compute exposure differ from the earlier decision-balanced run.

## Completed full evaluations

| Checkpoint | WEB (500) | EGO (326) | WEB held out (100) | EGO held out (181) | Original validation (128) |
|---|---|---|---|---|---|
| SFT | 54.69 | 42.73 | 55.73 | 42.48 | 63.90 |
| 1 | 54.83 | 41.92 | 56.41 | 41.44 | 64.53 |
| 6 | 55.32 | 42.23 | 56.39 | 42.08 | 66.70 |
| 12 | 55.61 | 42.41 | 56.27 | 42.24 | 67.61 |
| 15 | 56.24 | 42.82 | 55.89 | 42.39 | 67.09 |
| 16 | 56.46 | 42.65 | 55.74 | 42.16 | 67.76 |
| 24 | 56.64 | 42.62 | 55.78 | 41.47 | 66.21 |
| 30 | 56.56 | 42.83 | 55.56 | 42.06 | 66.85 |

Scores use official GPT-4.1 judged PAUC-F, omega 0.5, and raw learned argmax with response margin zero. Combined domain scores pool precision counts and gold-weighted recall before calculating PAUC-F; they are not averages of split PAUC-F values.

## Latest checkpoint against SFT

| Cohort | PAUC-F | Change | Paired 95% interval | Useful: SFT → checkpoint | Wasted: SFT → checkpoint |
|---|---|---|---|---|---|
| Full WEB | 56.56 | +1.87 | [+0.92, +2.85] | 701 → 763 | 290 → 290 |
| Full EGO | 42.83 | +0.10 | [-0.95, +1.15] | 479 → 475 | 473 → 458 |
| WEB held out | 55.56 | -0.17 | [-1.84, +1.40] | 138 → 146 | 44 → 54 |
| EGO held out | 42.06 | -0.42 | [-1.80, +0.98] | 280 → 274 | 295 → 293 |
| Original validation | 66.85 | +2.95 | [+1.13, +4.91] | 293 → 309 | 196 → 173 |

Intervals are paired episode bootstraps conditional on saved judgments and reused cohort selection. They do not capture judge uncertainty. Selecting the best observed checkpoint after evaluation is exploratory; every evaluated checkpoint is shown above.

Best observed combined WEB checkpoint: step 24; best observed combined EGO checkpoint: step 30. These are posthoc observations on benchmarks that include training identities, not independent model-promotion evidence.

## Exposure and overlap

| Source | Distinct episodes committed | RL decisions committed |
|---|---|---|
| ours | 120 | 38572 |
| web | 60 | 4160 |
| ego | 60 | 86636 |

Allowed source pools are 8,321 original episodes, 400 WEB training videos and 145 EGO training videos. WEB heldout 100, EGO heldout 181 and original validation 128 are excluded from this RL run. Separate sampled/unsampled training-pool summaries are in the saved evaluation JSON. Those labels refer to the planned schedule; the committed exposure counts above are actual.

Combined full-WEB/full-EGO results include training examples and do not establish a new heldout quality floor. The historical 54.67 full-WEB floor remains a historical reference, not an untouched-test comparison for mixed training. Historical benchmark reuse also precludes an untouched-test claim for the heldout cohorts. Perfect20 is a selected diagnostic.

## Training and inference recipe

- RLOO with four complete trajectories per episode. Event-containing episodes use official PAUC-F/100 rewards; genuinely event-free episodes receive -0.25 per physical fire. Trajectory-relative advantages are shared across that trajectory’s decisions.
- Source objective: 0.5 mean original episode loss + 0.25 mean WEB + 0.25 mean EGO, with the existing constant-32 normalization.
- Head/LoRA learning rates 1e-5/1e-6; KL 0.02; entropy 0.001; weight decay 0.01; gradient clipping 1; dropout zero; one optimizer step per collection.
- Decoder LoRA and pointer head learn. Base/vision/reference and promoted answer-generator weights remain frozen.
- Supervised replay coefficient 0.0035140820988695176; 126 credits per 4,096 RL decisions. The finite replay bank was extended by repeating its original prefix for the final 227 credits.
- Native videos sampled at 1 FPS using canonical physical-frame mappings; native WEB/EGO reply windows retained. Original examples retain trusted four-second windows. No future frames or gold answers enter policy inputs.
- Every decision uses at most 32 recent frames, timestamps, self-generated action/answer history and the latest aged-out response memory. Decoder context is rebuilt per decision; an entire six-minute EGO video is not accumulated in one prompt.
- RL actions are sampled directly from learned probabilities. Evaluation uses raw learned argmax, margin zero. No action offsets, calibration, thresholds, cooldowns or inference-time suppression were added. Frozen answer generation is greedy, at most 96 new tokens.

The inherited v2 last-response memory prompt discourages repetition and explicitly favors silence later in videos. It was held fixed in this experiment; its contribution to EGO behavior remains unproven and was not isolated by this sampling experiment.

## GPU layout and evaluation workflow

Training: allocation 38177, ip-10-3-120-167, seven controller/learner GPUs and one frozen answer-generator GPU. Evaluation: allocation 38226, ip-10-3-70-177, eight independent video workers. Learner correctness checks stay on the training node. GPT judging uses the authorized gateway and separate training/evaluation caches. No additional nodes were allocated.

Immutable checkpoints include model, optimizer, per-rank RNG and sampler state. Evaluations consume committed checkpoints asynchronously. The original pilot evaluated full milestones 6/12 and its stopped final 15; the extension evaluates full 16/24/32 and fixed subsets 20/28. A full step-30 evaluation was also completed after the technical interruption. Full evaluations are retained; stale fast probes may be superseded. Training waits when pending full evaluation is more than 10k decisions behind, allowing one update’s overshoot.

## Operational history and repairs

1. Initial Slurm steps exposed a single-CPU task affinity despite eight assigned GPUs. They were restarted before any update with CPU binding disabled, exposing 96 CPUs.
2. At the first cross-node gate, identical weights, RGB pixels and token inputs produced different vision embeddings. The evaluation process inherited host CUDA libraries and loaded cuDNN 9.7.1 versus training’s 9.1. Clearing LD_LIBRARY_PATH restored exact replay on all seven batches. Startup now asserts PyTorch 2.5.1+cu121, CUDA 12.1 and cuDNN 90100. Affected v2 gate/baseline outputs were archived and regenerated; original v1 cache reuse retained its exact cross-node gate and unchanged model/input/scorer provenance.
3. The preset quality rule stopped the original pilot at update 15 after full steps 6 and 12 had EGO-heldout useful counts 239/235 versus SFT 280, below 90% twice. The user explicitly overrode quality stopping and extended the same learned run to update 32. Coverage/silence diagnostics became warnings; correctness checks were retained.
4. Update 31 hit a supervised-replay bank bound: the bank contained 4,032 rows, while the longer budget requires 4,259 credits. The initial extension missed this capacity check. The repair preserves the original bank and appends its first 227 examples in order, keeping all prior replay selections and coefficients unchanged. The failed uncommitted batch and previous freezes/source were archived, then checkpoint 30’s model/optimizer/RNG/sampler state was restored.

The host NVIDIA drivers differ (570.133.20 training / 595.91.07 evaluation). The confirmed mismatch was the loaded cuDNN library; the original unchanged probe became exact after matching that library. No driver change or relaxed numerical tolerance was used.

## Training rollout outcomes

Each row below contains two different EGO episodes, four trajectories each. Best-of-four is a sampled diagnostic, not raw-argmax evaluation. Episode difficulty changes between updates, so these rows cannot establish a learning curve on a fixed cohort. The accompanying CSV preserves all four scores for every source and episode.

| Source | Episode groups | Mean episode PAUC-F | Mean best-of-four | Flat reward groups | Nonzero-advantage trajectories |
|---|---|---|---|---|---|
| ours | 120 | 57.37 | 69.38 | 15.0% | 85.8% |
| web | 60 | 50.16 | 62.85 | 5.0% | 95.0% |
| ego | 60 | 33.83 | 42.66 | 5.0% | 95.0% |

Reward flatness uses actual training rewards. For event-free episodes, PAUC-F is zero while the negative-fire reward can still vary; the CSV preserves both quantities.

| Step | EGO episode 1 best | EGO episode 2 best | Mean best-of-four | Mean all eight |
|---|---|---|---|---|
| 1 | 50.05 | 52.75 | 51.40 | 39.22 |
| 2 | 38.16 | 69.75 | 53.96 | 42.52 |
| 3 | 40.26 | 56.70 | 48.48 | 40.15 |
| 4 | 30.95 | 48.48 | 39.72 | 31.83 |
| 5 | 56.55 | 45.77 | 51.16 | 30.78 |
| 6 | 26.06 | 25.62 | 25.84 | 18.22 |
| 7 | 71.75 | 56.70 | 64.22 | 53.33 |
| 8 | 25.76 | 39.74 | 32.75 | 23.11 |
| 9 | 0.00 | 48.83 | 24.42 | 16.73 |
| 10 | 51.03 | 48.70 | 49.86 | 35.38 |
| 11 | 38.52 | 42.89 | 40.70 | 35.55 |
| 12 | 44.76 | 45.87 | 45.32 | 39.45 |
| 13 | 55.72 | 32.12 | 43.92 | 38.42 |
| 14 | 61.20 | 58.25 | 59.72 | 51.09 |
| 15 | 47.68 | 36.31 | 42.00 | 35.35 |
| 16 | 36.17 | 59.07 | 47.62 | 29.26 |
| 17 | 62.24 | 42.00 | 52.12 | 41.10 |
| 18 | 33.56 | 58.39 | 45.98 | 37.49 |
| 19 | 55.62 | 0.00 | 27.81 | 25.22 |
| 20 | 51.79 | 63.11 | 57.45 | 40.20 |
| 21 | 0.00 | 56.76 | 28.38 | 25.58 |
| 22 | 40.22 | 34.92 | 37.57 | 32.36 |
| 23 | 30.34 | 24.64 | 27.49 | 20.49 |
| 24 | 27.70 | 19.77 | 23.74 | 14.66 |
| 25 | 46.30 | 45.93 | 46.11 | 40.18 |
| 26 | 14.73 | 58.04 | 36.39 | 33.13 |
| 27 | 40.71 | 36.22 | 38.47 | 31.34 |
| 28 | 45.05 | 31.75 | 38.40 | 33.79 |
| 29 | 59.99 | 35.99 | 47.99 | 33.34 |
| 30 | 40.24 | 61.67 | 50.96 | 45.59 |

## Runtime and correctness accounting

Committed updates: 30. Successful-update time: 12.05 hours; successful-update eight-GPU cost: 96.38 GPU-hours. This excludes model loading, restore gates, idle evaluation backpressure, failed attempts and standalone evaluation; it is not total allocation cost.

Maximum recorded rollout/replay probability error across committed updates: 0. Final checkpoint file hashes will be verified on completion. Source manifests, schedule, runtime contract, freezes, replay repair and checkpoint commits are retained.

## EGO response timing at the latest full checkpoint

| Video seconds | Decisions | Fires | Useful | Wasted | Mean Response probability |
|---|---|---|---|---|---|
| 0–120 | 39120 | 391 | 193 | 198 | 4.77% |
| 120–240 | 39120 | 291 | 162 | 129 | 3.92% |
| 240–360+ | 39444 | 251 | 120 | 131 | 3.44% |

These are descriptive temporal buckets; event density and episode content differ across video time. They do not by themselves prove a causal time-dependent suppression effect.

## Earlier decision-balanced mixed run

| Checkpoint | Full WEB | Full EGO |
|---|---|---|
| 5 | 55.15 | 41.88 |
| 10 | 56.53 | 42.86 |
| 14 | 56.63 | 42.24 |
| 16 | 56.74 | 42.35 |

This is a different training recipe and episode exposure, shown for context rather than pooled with the episode-balanced run.

## Artifacts and provenance

- [continuation.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/continuation.json)
- [freeze.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/freeze.json)
- [workflow-freeze.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/workflow-freeze.json)
- [runtime-contract.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/runtime-contract.json)
- [replay-bank-repair.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/replay-bank-repair.json)
- [commit.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/step-000030/commit.json)
- [result.json](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/evaluation/final-step30/result.json)
- [full-results.csv](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/final-documentation/full-results.csv)
- [training-episode-rollouts.csv](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/final-documentation/training-episode-rollouts.csv)
- [summary.txt](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/final-documentation/summary.txt)

Related records: [experiment plan](../plans/jevany_2b_episode_balanced_mixed_rloo.md), [original mixed plan](../plans/jevany_2b_mixed_web_ego_rloo.md), and [experiment log](experiment_log.md).
