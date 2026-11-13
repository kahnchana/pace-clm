# Episode-balanced mixed WEB/EGO RLOO pilot

Authorized and launched 2026-11-12, 15:05 UTC.
Output: `outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2`.
Training workflow: allocation step 38177.37 on `ip-10-3-120-167`, tmux
`pace-episode-balanced-v2-train`. Evaluation workflow: allocation step 38226.25
on `ip-10-3-70-177`, tmux `pace-episode-balanced-v2-eval`.
The first-update gate controls automatic continuation; launch is not a claim
that this gate or any quality comparison has passed. All 58 focused CPU checks
passed before launch. The frozen schedule matches the counts below.
Initial steps 38177.35 / 38226.23 were stopped before any training update to
correct Slurm's one-CPU task affinity. Both restarted at 15:08 UTC with
`--cpu-bind=none`; the available node CPU affinity is 96 CPUs. Allocations and
GPU assignments are unchanged.

At 15:41 UTC the first-update cross-node gate stopped the workflow. Step 1 and
seven-learner restore passed, with zero training-node replay error. Investigation
found identical model tensors, decoded RGB pixels, processor pixels and token
inputs, but different frozen vision embeddings on the evaluation node. The
training node loaded cuDNN 9.1 (`90100`); the evaluation node inherited host CUDA
paths through `LD_LIBRARY_PATH` and loaded cuDNN 9.7.1 (`90701`). Substituting saved
training vision embeddings in a diagnostic recovered exact decoder scores.
Clearing the inherited library override made the unchanged original probe pass
all seven batches with zero error despite the nodes' different driver versions.

At 16:50 UTC both workflows restarted through the run-local
`launch-pinned-runtime.sh`, which removes the host library override and asserts
PyTorch `2.5.1+cu121`, CUDA `12.1`, and cuDNN `90100` before launching. No learned
weights, scores, tolerances, source freezes, schedule, or optimizer settings were
changed. First-gate/baseline outputs computed under the mismatched v2 runtime
were archived under `audit/runtime-mismatch-20261112T165029Z/` and are regenerated.
Original v1 cached baselines remain reusable with their recorded exact seven-rank
cross-node gate and unchanged media/model/input/scorer hashes. The saved step-1
checkpoint is retained; continuation still waits for regenerated gate inference
and baseline records. Runtime evidence is in `runtime-contract.json`,
`pinned-runtime-*.json` and `diagnostics/pipeline-*.json` under the run directory.

## Question and comparison

Test whether allocating 50% original / 25% WEB / 25% EGO by episode count,
rather than decision count, improves EGO coverage and judged quality without
losing WEB or original-data behavior. Initialize fresh from original J1-S SFT
step 8243, not mixed-v1 step 16. Retain mixed-v1 artifacts unchanged.

Keep the existing source-weighted objective, constant-32 loss normalization,
official episode PAUC-F rewards, four trajectories per episode, optimizer,
supervised replay, frame sampling, prompt, memory, and inference contracts.
This is a decision-budget-matched practical sampling comparison; update count,
episode identities, decision shares, and training cost differ, so it does not
isolate episode diversity independently of all these effects.

## Sampling and budget

Use the existing frozen episode pool and train/heldout identities. Independently
of the old decision-deficit scheduler, construct queues in source order
`ours`, `web`, `ego`, using one Python `random.Random(42)` instance and shuffling
each source's existing ordered pool in that order. Consume each queue in order
without replacement. Each update contains exactly four original, two WEB and
two EGO episodes, with four complete causal trajectories per episode. Do not
truncate an episode or merge an extra-large final batch. Stop before a complete
eight-episode batch would exceed 100,000 decisions.

A CPU dry calculation against the current pool gives:

| Source | Episode share | Distinct episodes | Trajectories | Decisions | Decision share |
|---|---:|---:|---:|---:|---:|
| Original | 50% | 92 | 368 | 29,320 | 29.67% |
| WEB | 25% | 46 | 184 | 3,096 | 3.13% |
| EGO | 25% | 46 | 184 | 66,420 | 67.20% |
| Total | 100% | 184 | 736 | 98,836 | 100% |

There are 23 updates, 32 trajectories/update, and 3,640–5,248 decisions/update.
Persist and hash the exact ordered schedule before any GPU work; the preparer
must reproduce these counts or report the discrepancy before launch. Episode
counts are not response counts. Preserve original source loss weights
`.5 * mean ours episode loss + .25 * mean WEB episode loss + .25 * mean EGO
episode loss`, with the existing constant-32 normalization, not a new
normalization by episode length.

## Two-node execution

Training uses only `ip-10-3-120-167`, allocation 38177, all eight assigned GPUs:

- GPUs 0–6: seven resident Qwen3.5-2B controller/learner replicas, synchronized
  decoder LoRA and pointer-head updates, and frozen SFT reference computation.
- GPU 7: the existing frozen promoted answer generator with dynamic batching.
  Generated answers enter each trajectory's subsequent history. GPT-4.1 training
  rewards use the user-authorized gateway, with a separate training cache.
- Controller visual features are cached; decoder forwards rebuild the bounded
  causal prompt with `use_cache=False`. Each trajectory reaches the physical
  episode end. Every update saves a committed checkpoint, optimizer, all seven
  learner RNG states, policy version, queue cursor and exact replay layout.

Evaluation uses only `ip-10-3-70-177`, allocation 38226, all eight assigned GPUs:

- Eight independent video workers partition each evaluation cohort. Each worker
  loads the checkpoint controller and the frozen answer generator needed for its
  videos; no cross-node gradient synchronization or training occurs here.
- Consume atomically committed checkpoints from the shared filesystem through
  the existing durable evaluation queue. Run inference and GPT-4.1 scoring
  asynchronously with training, using the separate evaluation judge cache.
- All benchmark, validation, fast-probe and temporal diagnostic evaluations run
  here. Required learner restore/gradient/replay correctness checks remain on
  the training node, as required by AGENTS.md. Cross-node replay is a forward
  correctness check, not optimization on the evaluation node.
- Allocate no other nodes. Verify both hostnames, existing allocations and GPU
  availability before starting; an unavailable allocation blocks launch.

## Fixed recipe and input contract

Head/LoRA learning rates 1e-5/1e-6, KL .02, entropy .001, weight decay .01,
gradient clip 1, dropout zero, one optimizer update per collected batch.
Keep the original replay manifest and coefficient .0035140820988695176;
replay credits accrue at 126 examples per 4096 decisions.

Use physical frame mappings at 1 FPS, native WEB/EGO timestamps and annotation
windows, and four-second trusted windows for original data. At each decision,
show the latest 32 frames and self-generated action/answer history, plus the
latest answer aged out of that window. Answer generation remains greedy with
96 maximum new tokens. Training samples actions directly from learned
probabilities; evaluation takes raw learned argmax, response margin zero.
Do not add score offsets, temperatures, thresholds, cooldowns or suppression.

The inherited v2 memory prompt explicitly favors silence later in a video and
may discourage repeated EGO events. Preserve it for this sampling experiment;
record the limitation. Any prompt/memory change requires a separately specified
training experiment and protocol-matched baselines. Do not silently change the
prompt for checkpoint selection or benchmark evaluation.

## Preflight and recovery

Implement explicit episode-count scheduling and a versioned run configuration;
the current mixed-v1 entrypoints hard-code parts of the old recipe and must not
be used unchanged. Preserve old freezes and output paths. Add meaningful CPU
checks for deterministic sampling, exact 4:2:2 composition, no duplicate training
identities, complete episodes, heldout exclusion, source loss scaling, schedule
restore, and evaluation milestones. Confirm policy/raw-argmax checks.

Before continuing past update 1, require finite gradients, expected head/LoRA
updates, reserved memory below 115 GiB, committed file hashes, exact optimizer/
RNG/cursor restore, collected-batch replay, and independent evaluation-node
input/logit parity and raw-argmax smoke. Retain probability tolerance .005,
max KL 1e-4, mean KL 1e-5 and selected-logp .05. Reuse baselines only after
checkpoint, media, prompt, annotation, generator, scorer and cohort hashes match.

The prior run failed waiting for learners at its last update. Before launching,
inspect saved worker traces and resolve or instrument that failure. Add explicit
per-rank progress/phase timestamps and diagnostic capture on timeout; do not
assume that extending a timeout fixes a stall. Bound all batch sizes as above.
The old failed final batch had 10,284 decisions and 55 episodes; its stack files
were empty, so the root cause remains unknown. The implementation now records
per-rank progress and captures worker stacks on timeout. New batches never
exceed 5,248 decisions; no timeout-only repair is claimed.
An interrupted run resumes only from the last verified committed checkpoint,
restoring sampler/optimizer/RNG state; incomplete batch artifacts stay separate
and never advance the cursor. Do not retry deterministic failures indefinitely.

## Evaluation and reporting

Retain the exact seven main cohorts: WEB train 400 / heldout 100, EGO train 145 /
heldout 181, original train 128 / validation 128, and selected perfect20.
Distinguish membership in the allowed training split from actual exposure:
v2 uses only 46 WEB and 46 EGO identities, so additionally report actually
sampled versus unsampled identities inside each allowed training split.

On the evaluation node, retain original SFT and mixed-v1 full step 5/10/14 results
as references; evaluate mixed-v1 step 16 there to close its missing final result.
This reference job must not indefinitely block the v2 first-update gate.

- First-update gate: independent replay/parity and raw-argmax first-batch smoke.
- Fast probes: at completed update boundaries crossing approximately every 10k
  decisions, fixed 32-episode subsets of the six main cohorts plus perfect20.
  Full evaluations replace coincident fast probes.
- Full evaluations: steps 6, 12, 18 and 23 for the CPU schedule above (first
  updates reaching 25k/50k/75k decisions, then final budget).
- On stop/failure: retain all committed checkpoints and complete pending full
  evaluations, including the latest committed checkpoint where feasible. An
  evaluator must drain pending jobs even after training fails; mark incomplete
  runs honestly and do not discard pending checkpoint jobs.

Each full result reports combined full-WEB/full-EGO PAUC-F as well as split
scores, episode means, useful/wasted/total fires, precision, temporal recall and
paired episode-bootstrap differences against SFT and available prior-run
checkpoints. Full-domain scores include training identities and do not replace
the historical 54.67 full-WEB floor with a heldout-quality claim. Historical
cohort reuse prevents an untouched-test claim.

Add EGO temporal diagnostics to every full evaluation: response frequency and
useful/wasted responses in 0–120 / 120–240 / 240–360 seconds, event-window
coverage, learned Response probabilities, memory presence/age, and a distinction
between repeated and new event descriptions. Report actual input token lengths
and frame counts so late-video behavior can be assessed without changing inputs.

For each training update, log episode and decision exposure by source; all-four
PAUC-F values, mean and best-of-four per episode; reward flatness and nonzero
advantages; temporal response/probability statistics; KL, entropy, parity,
gradient norms, replay loss, throughput and GPU-hours. Plot fixed-cohort results
separately from changing training episodes. Do not equate best-of-four sampling
with raw-argmax deployment quality.

Never drop full evaluations. Supersede stale fast probes; pause at update
boundaries when training is more than 10k decisions ahead of an unresolved full
evaluation (allow one whole-update overshoot). Retain the old coverage stops:
heldout fires below 10% of SFT, or useful responses below 90% of SFT in either
heldout domain at two consecutive full checks. These stop training, not inference.
Report EGO wasted-response growth explicitly even if coverage gates pass.

Judge success primarily by EGO-heldout PAUC-F and useful/wasted responses,
alongside WEB-heldout and original validation retention. Inspect confidence
intervals and report all four milestones; no best-checkpoint-only improvement
claim. The final report includes actual episode identities, budgets, timings,
node provenance, any failure/recovery, and remaining uncertainty.

## 2026-11-13: user-authorized continuation to step 32

The original coverage rule stopped the learner at step 15 / 63,216 decisions
after full steps 6 and 12 had EGO-heldout useful counts 239 and 235 versus SFT
280. The user explicitly overrode that rule: continue through step 32 regardless
of coverage and keep evaluations running. Preserve the original stop request,
freezes, checkpoints and reports as audit evidence.

Continuation output is
`outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/`. Its frozen schedule
preserves all original 23 batches exactly and extends to 32 updates / 138,460
decisions, with 128 original / 64 WEB / 64 EGO distinct episodes. Resume from the
exact step-15 model, optimizer, all learner RNG states and sampler cursor; 17
updates / 75,244 decisions remain. Original learning rates, reward, source loss
weights, supervised replay, prompt, memory and raw-argmax inference are unchanged.

Coverage and near-silence quality diagnostics are warnings only in the
continuation; numerical, state-integrity and replay correctness checks remain.
The run-local trainer/evaluator snapshots and their diff are recorded in the
continuation directory so the original run's frozen source files remain valid.
Full evaluations are scheduled at 16/24/32, subsets at 20/28, with the same
10k-decision bound on evaluation lag. Update-32 evaluation and judging must
finish before reporting workflow completion.

Both pinned-runtime workflows launched at approximately 03:34 UTC in the same
two allocations. The training node first verifies seven-learner restore at step
15. The evaluation supervisor waits for the already-running prior-run step-16
reference evaluation to finish, then uses all eight evaluation GPUs for the
continuation. No quality evaluation is placed on the training node. Two targeted
continuation tests, generated-source compilation, and exact equivalence between
the prepared freeze and the trainer-generated resume contract passed before
launch. See `continuation.json`, `workflow-status.json`, `evaluation-status.json`
and `extension-resume-verification.json` for provenance and live progress.

### Step-31 replay-bank repair

On 2026-11-13 at 12:31 UTC the continuation failed before its step-31 optimizer
commit: supervised replay requested credits 3979–4123 from a 4,032-row bank.
The extended 138,460-decision schedule needs 4,259 replay credits. Step 30 /
129,368 decisions remained the last intact checkpoint; this was not a coverage
stop. Extending the training budget without checking replay-bank capacity was
an implementation omission.

The repair preserves all original 4,032 rows and appends their first 227 rows
in original order, with no new source examples or coefficient changes. It
preserves all replay selections through step 30 and supplies the exact scheduled
145/135 examples for steps 31/32. The original bank descriptor remains in
`anchor-extended/freeze.json`, with explicit `continuation_replay` metadata and
the active expanded-manifest digest. The prior freezes, preparer source and
uncommitted step-31 artifacts are archived under the continuation's
`audit/replay-bank-repair-20261113T142310Z/`; active freezes record the manifest
extension and resume-30 supervisor change. The 32-step schedule is unchanged.

Three tests, including replay partitioning across the bank boundary, and exact
trainer-contract reconstruction passed. At approximately 14:24 UTC the same
two allocation workflows restarted: seven-learner step-30 restore on the
training node and the evaluator waiting for step 32 on the evaluation node.
Step-30 full evaluation completed at 56.560 WEB / 42.826 EGO PAUC-F. See
`replay-bank-repair.json` and `resume-step30-verification.json` for recovery
provenance and actual restore success; launch itself is not a completed update.
