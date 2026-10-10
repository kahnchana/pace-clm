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

## 2026-10-02 — fresh gate and scale-up follow-through

- Froze two new disjoint 200-video Markii cohorts, with 11,563 development and
  12,658 final decisions, excluding the prior PACE-200 and mixture cohorts.
- Replayed the 3,260-update run to exact internal-dev-NLL minima at steps
  200–900, and trained a doubled-Hold-loss variant on the run 5 mixture.
- Selected a +0.5 Response margin on the balanced MLP using only the new
  development cohort and firing/Hold constraints. It reduced final fires but
  reduced timing PAUC-F from 81.92 to 81.42.
- On the new final cohort, the frozen run 5 MLP scored 83.38 mean timing PAUC-F
  across three seeds versus 82.08 for PACE token generation, with fewer fires
  and repeats. Its paired video bootstrap PAUC-F difference was +1.31 points,
  95% interval [+0.17, +2.50].
- Early replay, doubled Hold loss, and full PACE-exposure training all scored
  below the frozen 24.3k run 5 MLP on final timing PAUC-F. No CLM head showed
  a repeatable advantage over its MLP comparator.
- Saved exact manifests, features, checkpoints, per-video and token predictions,
  scorer outputs, bootstrap, and completion marker under
  `outputs/pace-clm-gate/`. Judged response quality and WEB/EGO remain unrun.

## 2026-10-10 — full WEB/EGO judged transfer

- Evaluated the frozen run 5 MLP seed-42 decision head with the promoted Qwen
  backbone and original response generator on all 500 WEB and 326 EGO videos.
- Completed 839/839 WEB and 716/716 EGO GPT-4.1 judge requests after explicit
  authorization; no reported request errors or missing scored videos.
- With `pauc-f-v1` and `omega=0.5`, judged PAUC-F was 53.50 WEB and 43.95 EGO,
  below promoted PACE Qwen token generation at 57.47 and 46.05. The MLP fired
  925 versus 1,132 times on WEB and 1,310 versus 1,634 times on EGO, but lost
  judged recall on both splits.
- Saved full predictions, judge outputs, scored records, and summaries under
  `outputs/pace-clm-web-ego-full/`. Promoted PACE WEB/EGO raw outputs are absent
  locally, preventing paired video bootstrap and baseline rescoring.
