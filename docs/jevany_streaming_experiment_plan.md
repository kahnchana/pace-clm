# PACE × JevAny: streaming decision-model experiment plan

Date: 2026-10-22  
Execution repository: `pace-clm`  
Status: implementation and experiment plan; no new training runs completed.

## Objective and hypothesis

Test whether a JevAny-style, candidate-conditioned decision model improves
proactive streaming response decisions when trained on the existing PACE data.
At each decision time, predict `Silence`, `Hold`, or `Response`; generate an
answer only for `Response`.

The main hypothesis is that adapting the decoder together with a pointer head
outperforms a frozen-feature classifier. Switching repositories or using a
different head alone is not evidence of improvement.

The first study changes the controller, not the response generator, dataset,
streaming evidence, or scoring protocol. Learned memory retrieval is a later
extension.

## Evidence motivating this round

PACE-CLM has already completed frozen CLM/MLP heads, data-mixture studies, and
exposure scaling. CLM did not establish a repeatable advantage over MLP.
The selected frozen MLP improved timing on a held-out PACE cohort but reduced
judged WEB/EGO PAUC-F to 53.50/43.95, versus promoted PACE's 57.47/46.05.
Fewer fires alone did not produce better useful-response coverage.

Read before implementation:

- `docs/design.md`
- `docs/experiment_log.md`
- `agent_notes/experiments/status.md`
- `agent_notes/experiments/pace_clm_10pct_mixture_plan.md`
- `agent_notes/experiments/pace_clm_web_ego_full.md`
- `configs/qwen_mvp.yaml` and `configs/actions_v1.yaml`

Do not overwrite or reinterpret the completed frozen-head results as results
of the new experiment.

## Repository responsibilities

| Repository | Responsibility |
| --- | --- |
| `pace-clm` | Experiment orchestration, JevAny integration, manifests, configurations, controller checkpoints, reports |
| `stream-gemma` | PACE loading, causal windows and history, labels, generation, benchmark adapters and established scorers |
| `JevAny` | Pinned reference implementation for pointer readout and multimodal decoder-LoRA training |

Integrate a pinned JevAny checkout or an attributed implementation of the
required components. Record its exact commit and local patch. Preserve its
license and attribution. Do not copy the entire benchmark stack or replace the
Stream-Gemma runtime with JevAny's generic server.

JevAny supports native media training with frozen vision encoder/projector,
decoder LoRA, and a decision head. This does not automatically provide PACE's
interleaved streaming prompts, existing adapter loading, or cached-frame path.
Its released decision adapters disclose text-only decision training; they are
not the default initialization for this controlled experiment.

References:

- <https://github.com/SimpleJev/JevAny>
- <https://github.com/SimpleJev/JevAny/blob/main/docs/TRAINING.md>
- <https://github.com/SimpleJev/JevAny/blob/main/jevany/model.py>
- <https://github.com/SimpleJev/JevAny/blob/main/jevany/backbones.py>

## Fixed model and streaming contract

Use `configs/qwen_mvp.yaml` and the promoted artifact as the source of truth.
Verify these recorded values against files on the server:

| Field | Recorded value |
| --- | --- |
| Base | `Qwen/Qwen3.5-35B-A3B` |
| Base revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| PACE adapter | `../stream-gemma/outputs/lora-qwen-w1-long-neg` |
| Adapter SHA-256 | `eb11e600de59edf48ef2d71bb632ae5944c0474441f264bb3ce6ceae8bb71da9` |
| Precision/attention | BF16 / SDPA, subject to pinned runtime compatibility |
| Observation cadence | 1 FPS |
| Window | At most 32 frames, ending at the decision time |
| Frame preprocessing | 384 px; 144 vision tokens/frame |
| Response memory | `last-response`, instruction `v2`, `k=1`, `aged-out` |
| Generation prefix | `direct` |
| Label mapping | `Standby -> Hold`; preserve `Silence` and `Response` |

Use the same original PACE generator checkpoint and decoding configuration for
all controllers. The controller's adapted weights must never accidentally be
used for response generation. Use separate model instances or verified adapter
switching; account for the resulting memory, recomputation, and latency costs.
If using adapter merging to initialize the controller, merge only a copy and
verify prediction parity before training. Do not assume a PACE PEFT checkpoint
can be passed directly to JevAny's checkpoint loader.

## Data and causal decision representation

1. Reuse the exact 16,994-video stride-10 training cohort and existing
   video-disjoint development/validation manifests. Resolve paths on the server
   and save hashes before extracting or training.
2. Reconstruct the promoted PACE episode sampling and `learn` masks, including
   trailing and post-response negatives. The prior documented PACE exposure was
   26,080 episodes and 263,760 supervised decision positions; verify actual
   counts. The earlier balanced 24,300-decision head sample is a different
   distribution and must not be silently substituted.
3. Freeze one common decision manifest for the new matched experiments.
   Include video/query ID, time, causal frame IDs, episode/history identity,
   memory provenance, target, sampling weight, and split. Distinguish repeated
   presentations from unique decision positions.
4. Each scored prefix contains the query, causal visual window, allowed prior
   assistant history, response memory, action instructions/options, and a final
   decision marker. No future frames or current/future gold answers enter it.
5. Score explicit candidate options with a JevAny-style pointer: compare the
   final decision-marker representation with candidate representations.
   Candidates are contextualized by the current evidence, unlike a bank of
   separately cached action embeddings. Freeze wording/order for this study.
6. Keep the existing causal frame/timestamp topology. A custom native-media
   bridge may be required; do not collapse timestamped frame/history turns into
   a generic video prompt without treating that as a separate experiment.

Training may begin with existing teacher-forced histories, with that limitation
recorded. Validation selection and benchmarks must use self-generated rollout
histories. Generated-history training is a later, separate intervention.

**Context preflight:** JevAny's inspected default training budget is 2,048
tokens. PACE's visual allocation alone is 4,608 tokens for 32 frames. Measure
expanded token counts including history/options, set adequate limits, and
report admission/rejection counts. Do not silently shorten windows, resample
media, discard long examples, or truncate evidence to make a run fit.

## Controlled experiment matrix

| ID | Controller | Trainable parameters | Question |
| --- | --- | --- | --- |
| B0 | Promoted PACE token decision | None | Existing reference |
| B1 | Existing selected frozen MLP | None for archived checkpoint | Previous approach |
| J0 | JevAny-style pointer over PACE decoder | Pointer head only | Does contextual option scoring help without decoder adaptation? |
| J1 | Same pointer and inputs as J0 | Decoder LoRA plus pointer; decision embeddings only if required | Does decoder adaptation improve decisions? |

Re-evaluate B0/B1 on the current frozen validation protocol. Their historical
scores are references, not substitutes for matched predictions. For strict
data-matched head attribution, also refit the MLP on the new common decision
manifest; label this separately from archived B1. J0 and J1 must share the
same prompts, candidates, labels, decision sample, and evaluation rules.

Initialize J1 from the promoted PACE model. Reuse the PACE LoRA architecture
and target modules where possible; do not change rank, initialization source,
and readout all at once. Record which adapter is trainable and its targets.
Use decision cross-entropy for J0/J1 initially, without new RL or answer loss.
Keep vision frozen. Avoid generic option/label augmentations that change the
meaning of these three streaming actions.

After the main comparison, optional ablations are binary `wait/respond`,
direct-token readout, and raw-base versus PACE initialization. Do not launch
all of them before establishing the primary result.

## Execution stages and acceptance criteria

### Stage 0 — source and environment audit

- Record commits, dirty patches, model/adapter hashes, dataset/split hashes,
  hardware, and dependency lock/freeze.
- Check server paths and dataset availability. Use a dedicated environment if
  JevAny dependency requirements conflict with the working PACE environment.
- Reproduce a small B0 rollout and scorer invocation before integration.
- Audit initialization, processor, prompt, labels, and full-frame admission.
- Estimate full resident MoE weights, activations, optimizer state, and generator
  residency. The active parameter count is not the total weight memory.

### Stage 1 — integration smoke and tiny overfit

Implement a differentiable controller path rather than reusing
`FrozenFeatureExtractor.extract()` for J1: it freezes the model and executes
under `torch.inference_mode()`.

Verify causality, exact frame/timestamp alignment, memory formatting, label
mapping, and complete media-token expansion. Verify only intended LoRA/head
parameters get gradients; vision remains frozen. Verify checkpoint reload and
generator isolation. Test J0/J1 on identical causal inputs.

Run a short real-model training smoke, then overfit a fixed tiny set containing
all three states, nonempty memory, and long windows. Require finite losses,
updated intended parameters, clear fit improvement, successful save/reload,
and no OOM or visual-slot mismatch. This demonstrates plumbing, not quality.

### Stage 2 — matched pilot and learning curves

- Start with seed 42 and a deterministic subset of the common training manifest
  that includes rare Hold cases, response boundaries, and sustained silence.
- Train J0/J1 with separate head/LoRA learning-rate fields and a small,
  prespecified validation-only search. Record the chosen values before the
  complete 10% run; do not copy the old frozen-head LR into LoRA blindly.
- Save fixed exposure checkpoints. Compare equal decision presentations and
  report updates, unique examples, wall time, peak GPU memory, and throughput.
- Do not repeat a small dataset merely to reach 3,260 steps: previous head
  repetition did not help. Use the actual common decision exposure.
- Advance technically sound J1 to the full 10% cohort, then confirm the selected
  configuration with seeds 42/43/44. Run matched J0/control seeds as feasible
  and report missing runs rather than implying seed robustness.

### Stage 3 — validation selection and freeze

Reuse the existing validation sets. Clearly label previously inspected
selection/final cohorts as reused validation, not untouched confirmation.
Never choose the model or threshold from WEB/EGO/TV/VAD or other benchmarks.

Report per-state precision/recall/F1, NLL/calibration, timing PAUC-F, fires and
repeats/video, missed events, and latency. Inspect long silence, post-response
negatives, and Hold regions. Timing-only PAUC-F is a diagnostic, not a semantic
quality result.

Before comparing candidates, freeze a manageable subset of the same validation
pool for full generated-response judging. Use the established GPT-4.1 judge,
`pauc-f-v1`, and `omega=0.5`, checking scorer parity. Rank finalists by judged
validation PAUC-F; inspect recall, repetitions, and seed consistency. Prefer a
simpler/cheaper model when differences are inconclusive. Choose any response
threshold/margin on validation only and store it with the checkpoint.

Freeze selected configuration, checkpoint-selection rule, seed policy, and
threshold before benchmark generation. Default benchmark seed is 42, not the
best benchmark-performing seed. If J1 does not improve judged validation,
report the outcome and diagnose it before increasing scale.

### Stage 4 — matched benchmark reporting

Evaluate the frozen candidate and B0 with the same generator, frame budgets,
decoding, and unchanged official/Stream-Gemma scorers. Keep archived B1 results
clearly identified. Preserve resumable per-rank predictions and judge caches.

| Evaluation | Required reporting | Priority |
| --- | --- | --- |
| ProactiveVideoQA WEB/EGO/TV/VAD | Judged PAUC-F per split, components, fires/repeats, coverage | Primary |
| Full ESTP-Bench | Official ESTP-F1 with Explicit/Implicit/Contextual breakdown; existing PAUC-F breakdown | Primary |
| OVO CRR | Paper-compatible official accuracy/timing protocol, with exact scorer definition | Primary |
| Previously reported OVO forward tasks | REC/SSR/CRR breakdown under their existing contracts | Continuity |
| StreamingBench PO | Official time accuracy | Primary |
| StreamingBench PO-I | Existing injected-query protocol, if retained in the paper | Continuity |
| Full OVO | Backward, realtime, forward, per-task scores and aggregate only with complete coverage | Secondary |
| Broader StreamingBench | Supported task scores, modality coverage, official aggregate only when valid | Secondary |

Audit OVO's exact existing timing/forced-answer semantics before reusing a
score name; do not call every OVO forward result a true continuous-streaming
timing measure. Record benchmark version and evaluation mode.

Full OVO is useful for diagnosing broader understanding and later memory work.
Broader StreamingBench is meaningful for the combined controller/generator,
but audio-dependent tasks require audio inputs and a supported model path.
Report a visual-only subset as such; do not present it as a full multimodal
score. For tasks requiring an answer at a specified query time, follow their
official answer protocol rather than introducing a new wait penalty.

Keep benchmark data evaluation-only. Broader QA scores may stay unchanged
with a fixed generator; they are not automatically evidence about firing-head
quality. If budget limits broader evaluation, report coverage and prioritize
the paper's proactive suite rather than fabricating a full aggregate.

Use paired video-level bootstrap intervals when matched raw predictions are
available. Historical promoted WEB/EGO raw outputs were absent locally in the
previous report; regenerate B0 or locate verified outputs before claiming
paired significance.

## Server launch handoff

The following is an implementation handoff, not an existing JevAny launcher.
The current `sbatch_train_eval_10pct.sh` trains cached-feature heads and must
not be used to launch J1 unchanged. No GPU jobs are launched by this document.

Existing server scripts record these locations; verify and override them:

```bash
export PACE_ROOT=/fsx/home/kranasinghe/repo/pace-clm
export STREAM_GEMMA_ROOT=/fsx/home/kranasinghe/repo/stream-gemma
export PACE_JEV_OUT=/fsx/sfr/data/kanchana/stream-gemma/pace-jevany
cd "$PACE_ROOT"
git status --short
git rev-parse HEAD
```

Required new implementation deliverables (proposed names, not existing files):

- `configs/jevany_streaming.yaml`: pinned sources, initialization, labels,
  window/token budgets, data manifests, optimizer, seeds, validation rule.
- `scripts/prepare_jevany_decisions.py`: deterministic causal train/val export
  and manifest validation, preserving promoted episode masks.
- `scripts/smoke_jevany_streaming.py`: processor, gradients, save/reload,
  tiny-overfit and generator-handoff checks.
- `scripts/train_jevany_streaming.py`: J0/J1 training, resumable optimizer/RNG
  state, exposure checkpoints, validation metrics and provenance.
- `scripts/eval_jevany_streaming.py`: controller hook into the existing
  Stream-Gemma runtime and benchmark adapters; no scorer reimplementation.
- `scripts/sbatch_jevany_streaming.sh`: stage-specific launch for smoke,
  training, validation rollout, and benchmark rollout; explicit paths/resources.

Model the scheduler integration on `scripts/sbatch_extract_10pct.sh`, which
records `xgen-mm`, `ml.p5en.48xlarge`, and eight GPUs, but verify current account,
partition, and allocation. Profile smoke memory before fixing production
resources. Standard DDP replicates the full model on every GPU; eight GPUs do
not automatically pool their memory. Create log directories before submission
and keep paid judge execution separate from model training jobs.

The server implementation agent should:

1. Read this plan and current repo instructions; pin JevAny and audit deltas.
2. Implement the bridge, config, manifests, focused tests, and stage launchers.
3. Run CPU checks and the real-model smoke/tiny overfit.
4. Save a preflight report with exact commands, resource estimates, counts, and
   resolved paths. Repair implementation failures before pilot training.
5. Launch the staged pilot, full 10% study, validation freeze, and selected
   benchmark suite in the order above within the available allocation/budget.
6. Update `docs/design.md` with implemented contracts and
   `docs/experiment_log.md` only with completed checks/results. This plan
   supersedes frozen-head-only scope for the new named experiment; retain the
   historical experiment configurations and conclusions.

## Artifact and reporting contract

Use a new output namespace, for example:

```text
pace-jevany/
  source/                  # revisions, dependency lock, patches
  manifests/               # data identities, sample counts, validation subsets
  smoke/                   # parity, gradient, reload, tiny-overfit reports
  runs/{experiment}/seed-{seed}/
  validation/              # logits, rollout predictions, judged scores
  selection.json           # checkpoint, threshold, rule and validation evidence
  benchmarks/{benchmark}/{split}/
  reports/RESULTS.md
```

Every result identifies checkpoint/hash, seed, split/version, controller,
generator/hash, history source, prompt version, sampling/context, threshold,
judge/scorer versions, missing/rejected examples, wall time, and GPU resources.
Report all planned completed conditions, including regressions.

## Later phase: selective memory retrieval

Only after the controller comparison, add `retrieve / skip`, candidate memory
selection, then a post-retrieval response decision. Begin with fixed memory
storage and candidate retrieval; defer learned writing/eviction. Existing PACE
timing labels do not directly label retrieval necessity. Derive and audit weak
targets using past-only evidence outside the visual window, irrelevant-memory
negatives, and answer-quality changes with/without retrieval.

Evaluate quality versus memory tokens, retrieval frequency, and end-to-end
latency on PACE plus appropriate OVO backward/contextual tasks. Keep this data
and architecture change separate from the same-data J0/J1 comparison.

## Completion checklist

- [ ] Sources, dependencies, artifacts, data, and validation identities frozen.
- [ ] Native streaming bridge preserves full causal evidence and history.
- [ ] LoRA gradients, generator isolation, save/reload, smoke and overfit pass.
- [ ] B0/B1/J0/J1 identities and matched training comparisons documented.
- [ ] 10% study and selected seed confirmations complete.
- [ ] Judged validation selection and threshold frozen before benchmarks.
- [ ] WEB/EGO/TV/VAD, full ESTP, OVO CRR/forward and StreamingBench PO reported.
- [ ] Full OVO / broader StreamingBench reported with accurate scope, or
      explicitly listed as pending with the reason.
- [ ] Results, failures, compute costs, provenance, and docs updated.
