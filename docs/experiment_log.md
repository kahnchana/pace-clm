# Experiment log

## 2026-09-27 — repository foundation

- Created standalone Apache-2.0 repository.
- Pinned and validated promoted PACE-Qwen contract.
- Added versioned MVP/action configuration.
- Added causal input, label, pooling, freezing, LM-head bypass, and projection
  gradient tests.
- No GPU experiment was launched; real-model smoke extraction and tiny-set
  overfitting remain later execution-order milestones.

## 2026-09-27 — PACE-200 CLM pilot

- Verified the pinned promoted Qwen/PACE model on GPU: hidden width 2048,
  final valid language-token pooling, no LM-head call, frozen backbone and
  adapter, and exact cached/uncached action-feature parity.
- Froze a balanced 768-example PACE-train manifest (256 examples per state,
  353 videos) and the existing 200-video validation manifest (10,658 decision
  points); train and validation video IDs are disjoint.
- Extracted reusable train/validation features and passed the balanced
  24-example overfit gate at 100%.
- Trained the required raw-cosine, linear, MLP, and separate-head CLM methods,
  plus projection-width, shared-head, and short-label variants. Trained methods
  use seeds 42, 43, and 44.
- Evaluated the existing PACE token-generation decision on the same inputs.
  It leads with 83.70 macro-F1 and 80.10 timing PAUC-F. The best CLM timing
  result is 65.14 PAUC-F for 256-d; no CLM variant beats the token baseline.
- Saved checkpoints, exact manifests, reusable features, per-decision logits
  and predictions, confusion matrices, unchanged-scorer outputs, and run
  provenance under `outputs/`.
- Verified the response-generation handoff on eight predicted CLM Response
  events from eight deterministic short videos. All eight produced non-empty,
  query-relevant text through the original PACE generation path; this is a
  qualitative smoke sample, not a judged generation metric.
- Semantic judged PAUC-F was intentionally not run; this pilot is timing-only.
  WEB/EGO remains untouched.
