# PACE-CLM 10% scaling plan

## Purpose

Run one controlled scaling round to determine whether CLM provides a repeatable
advantage over a capacity-matched classifier. The pilot is data-limited: the
large heads nearly memorize 768 examples, CLM does not improve macro-F1 over the
MLP, and it fires roughly twice as often as the PACE token baseline.

## Execution plan

1. **Calibrate the existing runs first.** Sweep the same Response logit-margin
   rule for linear, MLP, and CLM on PACE-200. Record macro-F1, no-judge PAUC-F,
   and fires/video. This tests whether over-firing is primarily a calibration
   problem before paying for more feature extraction.
2. **Freeze the Stream-Gemma 10% training pool.** Use its exact stride-10
   selection after valid-record filtering from the 176,021-record PACE training
   set. Exclude all PACE-200 video IDs and save the exact manifest,
   distributions, and hashes.
3. **Select 24k–30k diverse training decisions.** Target 8k–10k examples per
   state. Cap examples per video/event; include hard Silence decisions near
   event boundaries and Hold decisions at varied event offsets. Create a
   video-disjoint internal train/dev split for early stopping.
4. **Extract frozen features once.** Reuse the pinned promoted PACE-Qwen model,
   causal 32-frame inputs, final valid language-token pooling, and common gold
   history. Save sharded and merged artifacts with provenance.
5. **Build a nested learning curve.** Train on deterministic subsets of 768,
   3k, 12k, and the full 24k–30k examples from the same feature store.
6. **Run the focused comparison.** Evaluate linear, capacity-matched MLP,
   separate-head CLM 128/256-d, and short-label CLM with seeds 42/43/44. Use
   early stopping, regularization, and the same calibration budget for every
   learned method.
7. **Freeze choices, then evaluate PACE-200 once.** Report accuracy, macro- and
   per-state F1, NLL/ECE, confusion matrices, Timing F1@4s, no-judge PAUC-F and
   its components, fires/video, repeats, and latency. Do not run WEB/EGO during
   method development.
8. **Measure serving value.** Compare decision latency and compute with the PACE
   token-generation baseline; CLM is useful only if its quality/efficiency
   tradeoff is materially better.

## Decision rule

Continue CLM only if, across seeds, it beats the capacity-matched MLP by at
least **2 absolute no-judge PAUC-F points** and **1 macro-F1 point**, does not
increase fires/video, and its advantage grows with training-set size. For a
deployment case, it should also approach the PACE baseline within roughly 2–3
PAUC-F points while providing a measurable latency or compute benefit.

If CLM and MLP scale together, or the advantage disappears after equal
calibration, stop the CLM branch and pursue a calibrated classifier or temporal
gate distilled from the PACE token decision. Semantic judging and WEB/EGO begin
only after the scaled method passes this gate and is frozen.

## Execution record (2026-09-27)

- Stream-Gemma's pinned Qwen run used `video_sample_stride=10` after filtering
  to 169,939 valid records. The frozen pool has 16,994 videos, exactly matching
  its reported `records_after_stride`. There is no PACE-200 video overlap.
  See `manifests/pace_train_stream_gemma_10pct.jsonl` and
  `manifests/pace_train_10pct_summary.json`.
- The decision manifest has 27,000 examples: 8,100 per state for training and
  900 per state for internal dev, split by video. Hold is concentrated in 1,539
  training videos, so its per-video cap is 10; the manifest records the early,
  middle, and late Hold mix. See `manifests/pace_train_10pct_decisions.jsonl`.
- Existing PACE-200 pilot logits received an equal 42-threshold diagnostic sweep.
  Best no-judge PAUC-F was 75.76 for MLP, 74.75 for 256-d CLM, and 74.90 for
  short-label CLM; these thresholds were selected on PACE-200 and are **not**
  used for the scaled evaluation. Scaled thresholds will be selected on the
  internal dev split, weighted by the full cohort's state/stratum frequencies.
- Large artifacts and logs are under
  `/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/`. Slurm extraction job
  `36566` covered the first 450 assigned videos per rank. The always-on
  eight-H200 node `ip-10-3-15-107` extracted the disjoint tail and ran the
  downstream comparison. The pending Slurm training job was cancelled in favor
  of this node; extraction job `36566` was cancelled after its required shards
  were complete. The verified merge contains exactly 27,000 decisions from
  10,111 videos, with 9,000 examples per state.

## Outcome (2026-09-27)

- All 60 prespecified learning-curve runs finished on the always-on eight-H200
  node. The frozen PACE-200 evaluation used the internal dev-selected margins
  for all five methods and three seeds. The full per-seed metrics, learning
  curve, checkpoints, and provenance are in the external `results/` directory.
- Mean no-judge PAUC-F / macro-F1 / fires per video were: linear
  78.06 / 70.53% / 4.03; MLP 81.31 / 71.69% / 2.99; CLM-128
  80.78 / 72.20% / 3.49; CLM-256 77.91 / 70.90% / 3.89; short-label
  CLM-256 81.46 / 70.81% / 3.05. The existing PACE token baseline has
  80.10 no-judge PAUC-F.
- No CLM variant passes the across-seed gate. The short-label variant's mean
  PAUC-F is only 0.15 points above the MLP, while its macro-F1 is 0.88 points
  lower and it fires more often. CLM-256 has a large seed-44 regression.
- In 32 paired serving decisions, the head-based methods had roughly 1,121 ms
  median latency versus 1,496 ms for four-token PACE generation on this node.
  This latency benefit does not overcome the failed quality gate. Per the
  decision rule, the CLM branch stops here; semantic judging and WEB/EGO are
  not triggered. The next candidate is a calibrated classifier or temporal
  gate distilled from the PACE token decision.
- Final report: `/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/results/RESULTS.md`.
  Structured decision and artifact hashes are in `decision_summary.json` and
  `run_provenance.json` beside it; `complete.done` marks the finished run.
