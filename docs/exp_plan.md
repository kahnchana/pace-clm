# Ten-run PACE-CLM experiment plan

Status: planned, not launched. This plan uses one eight-H200 node and at most ten
head-training runs. It follows the judged WEB/EGO result in
[`../agent_notes/experiments/pace_clm_web_ego_full.md`](../agent_notes/experiments/pace_clm_web_ego_full.md).

## Objective and reason for the design

The frozen run 5 MLP scored 53.50 judged PAUC-F on WEB and 43.95 on EGO,
versus 57.47 and 46.05 for promoted PACE Qwen token generation. Its judged
fire precision was close to PACE's, but it found fewer value-adding responses
(WEB 654 versus 800; EGO 610 versus 754). The primary target is therefore
better response coverage without sacrificing answer quality or adding many
wasted fires. The earlier CLM variants also failed to beat a matched MLP
reliably, so every idea gets one CLM run and one MLP control on identical data.
That is ten training runs total; feature extraction, teacher labeling, judging,
and baseline evaluation are shared preparation or evaluation, not additional
head-training runs. One seed per run makes this a screening campaign; a
repeatability claim would require additional seeds after this ten-run cap.

Success means a frozen candidate reaches at least the promoted PACE judged
PAUC-F on **both** full splits (WEB 57.475, EGO 46.051). A practical parity
result is within 1.0 point on both splits with a measured serving-latency gain.
Neither result may be inferred from timing-only PAUC-F. Report the CLM–MLP
difference within each matched pair even if neither reaches PACE.

## Common protocol

- Freeze the promoted `Qwen/Qwen3.5-35B-A3B` revision and
  `lora-qwen-w1-long-neg` adapter, the 1-FPS causal 32-frame window, direct
  prefix, and last-response memory v2. Use the original Qwen response
  generator on every predicted Response. Only heads and any small causal
  temporal or utility modules in this table train. Do not update the Qwen
  weights or change the benchmark scorer.
- Use the existing 10% PACE training-video pool, with strict video separation
  from all development and test cohorts. Start from the frozen run 5
  24,300-position mixture and 2,700-position internal development split. Each
  run sees 24,300 labeled decision positions and uses seed 42; paired runs use
  the same positions, histories, labels, and sampling order. New history views
  or auxiliary labels attach to those positions. Record any repeated positions.
- Keep the existing optimizer and budget unless an experiment explicitly
  requires otherwise: AdamW `3e-4`, weight decay `1e-3`, batch 128, at most 60
  epochs, six-epoch patience on internal-dev cross-entropy. Use the same
  internal-dev Response-margin search for both members of a pair. Fix each
  pair's auxiliary-loss weights before the fresh validation results are read;
  store them in a versioned config.
- Use the existing short-label 256-dimensional CLM action head and
  capacity-matched MLP as the two base heads. In a pair, give both methods the
  same new data, temporal encoder, auxiliary targets, and loss terms wherever
  they apply. The CLM still scores frozen action-text embeddings; the MLP
  directly emits three action logits. Report parameter count and inference
  latency for both.
- Do not use WEB/EGO examples, judge verdicts, or the already inspected WEB/EGO
  failures for training, calibration, or checkpoint selection. Existing
  WEB/EGO numbers are comparison references only.

## The ten runs

Each numbered entry is one trained checkpoint followed by evaluation on the
fresh selection cohort below. Within each pair, the intervention is identical
apart from the CLM versus MLP decision head.

| Runs | Idea | Fixed intervention and specific hypothesis |
|---|---|---|
| **E01 CLM; E02 MLP** | Missed-window contrastive mining | On training videos only, use frozen run 5 to identify missed gold Response windows. Replace 20% of the run 5 Response positions with distinct missed-window positions, preserving the total class counts and per-video cap. Match each positive to a same-video state immediately before its valid window and one after the response. Add `0.25 ×` pairwise Response-score ranking loss with a 0.2-logit margin to cross-entropy. Test whether boundary-specific examples recover useful fires. |
| **E03 CLM; E04 MLP** | PACE decision distillation | Cache the promoted PACE token decision's three action probabilities on the unchanged run 5 training positions. Train with `0.5 × gold cross-entropy + 0.5 × T² KL(teacher || head)` at temperature `T=2`. Test whether the head retains PACE's response coverage while reducing token-decision latency. |
| **E05 CLM; E06 MLP** | Generated-history replay | On training videos, roll out the frozen run 5 policy with its generated Qwen responses and extract the same decision timestamps under those induced memories. Replace half the gold-history feature views with these generated-history views; labels still come from the frozen gold timing windows. Exclude ambiguous points and refill from the same training pool to retain 24,300 positions and the run 5 class counts. Use the common cross-entropy objective. Test the train–inference history mismatch. Both runs use the same replay cache. |
| **E07 CLM; E08 MLP** | Causal temporal head | Cache the current and preceding three consecutive 1-FPS final-token states for each run 5 position, preserving causal history. Feed them through the same small causal GRU (hidden width 256) before either head. Train with the common cross-entropy objective. Test whether short-term state change identifies answer readiness better than one pooled state. |
| **E09 CLM; E10 MLP** | Answer-aware utility | On training videos, generate Qwen answers at 4,000 frozen candidate times: 2,000 missed valid-window times, 1,000 current-head fires, and 1,000 near-window negatives. Use the same GPT-4.1 rubric; the utility target is judge score divided by 3 inside a valid window and zero outside. Add `0.25 ×` auxiliary mean-squared-error loss to cross-entropy and add `α × predicted utility` to the Response score, with `α` chosen only on internal dev from `{0, 0.25, 0.5, 1}`. Test whether expected answer utility improves *judged* timing decisions. Both runs use exactly the same candidates. |

E01/E02 keep the overall Response count fixed to isolate the effect of better
positive placement. E05/E06 use a shared initial rollout rather than
head-specific iterative DAgger, making the pair comparable within this budget.
E07/E08 require a new contiguous-state cache; the old sparse feature cache is
insufficient. E09/E10 require external judge calls only for **training-video**
examples and must log the request count, payload hash, and rubric version.

## Validation and promotion sequence

1. **Freeze manifests before training.** The Markii valid population has 710
   videos unused by the earlier PACE-200, mixture, and gate cohorts. Allocate
   200 by a new deterministic hash to `exp10-selection`, the next 200 to
   `exp10-confirmation`, and leave 310 untouched. Freeze IDs, gold windows,
   source hashes, and train-video nonoverlap. The existing 2,700-position
   internal dev remains the only early-stopping and margin-calibration set.
2. **Evaluate all ten on selection.** Run the full streaming policy with Qwen
   response generation on the same 200 selection videos. Judge generated
   responses and compute judged PAUC-F with the unchanged scorer. Record its
   recall component, judged fire precision, fires/video, wasted fires,
   duplicate responses, per-state F1, and latency. Run the frozen PACE token
   baseline and the existing run 5 MLP on this cohort once with the same
   generator and judge. These reference evaluations add no training runs.
   Rank by judged PAUC-F, with no per-candidate WEB/EGO inspection.
3. **Confirm at most two finalists.** Choose the highest-ranked CLM and the
   highest-ranked MLP that are no worse than the current run 5 MLP on selection
   judged PAUC-F and do not lose more than 1 point of judged fire precision
   relative to that MLP. Freeze their checkpoints and margins. Evaluate each once on the separate
   200-video confirmation cohort against the same PACE token baseline and run
   5 MLP reference. Advance only a candidate that improves over the current
   run 5 MLP on judged PAUC-F there; report all failures rather than relaxing
   this gate.
4. **Use WEB/EGO only for frozen finalists.** Run at most two confirmed
   candidates on the complete 500-video WEB and 326-video EGO splits, with
   GPT-4.1, `pauc-f-v1`, and `omega=0.5`. Compare with the recorded PACE
   57.475/46.051 and current MLP 53.498/43.953. Recover or rerun the promoted
   PACE raw outputs once if needed for paired video bootstrap and exact scorer
   parity; their old raw outputs are absent locally. Report full-split point
   estimates and paired 95% intervals where raw predictions permit them.

The WEB/EGO splits and past Markii cohorts have already been inspected. This
is a new development cycle, so only the newly frozen confirmation cohort is an
untouched within-project test. Any WEB/EGO comparison after this search must
disclose that the benchmark has been used in earlier model selection.

## One-node execution and artifacts

- Before allocating the node, implement and dry-run the shared teacher-logit
  cache, contiguous-state cache, induced-history cache, configurable paired
  trainer, and full Markii rollout/judge path on a few **training** videos.
  Existing scripts cover pieces of these workflows but do not yet implement
  the five interventions or full judged Markii evaluation. Check one-to-one
  manifest alignment, causal timestamps, and exact scorer parity before any
  of the ten full training runs.
- Allocate one eight-H200 node for the whole campaign. Perform common Qwen
  extraction, teacher inference, generated-history replay, and E09/E10 answer
  generation once, in that order. Train the small heads on one GPU each, with
  no more than four concurrent jobs and serialized writes. Run full streaming
  rollouts across eight GPUs one candidate at a time. Judge and aggregate on
  CPU while the next GPU training job runs if gateway throughput permits.
- Make each stage resumable and hash its inputs. Store the frozen config, hash
  records, and locally generated manifests under `manifests/exp10/`. Commit the
  small config and hash records; Git ignores the generated JSONL and gold JSON
  payloads. Store caches, checkpoints, per-video predictions, judge outputs,
  scores, logs, and a completion marker per run
  under `outputs/pace-clm-exp10/`. Never overwrite earlier gate or WEB/EGO
  artifacts. Run order: E01–E04 first, E05–E08 after shared feature caches,
  then E09–E10 after training-only quality labels are complete.
- Stop a run on malformed or empty Response text, noncausal feature access,
  manifest overlap, missing judge responses, or scorer mismatch. Report the
  partial artifact and repair the issue before proceeding. Do not count a
  failed preflight as an experiment; count each completed trained checkpoint
  exactly once toward the ten-run cap.

## Research basis

- Supervised contrastive learning: <https://arxiv.org/abs/2004.11362>.
- Knowledge distillation: <https://arxiv.org/abs/1503.02531>.
- Training under induced sequential states: <https://proceedings.mlr.press/v15/ross11a.html>.
- Temporal contrastive video representations: <https://arxiv.org/abs/2101.07974>.
