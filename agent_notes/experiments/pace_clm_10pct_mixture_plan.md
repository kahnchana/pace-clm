# PACE-CLM data-mixture experiments at the current 24.3k scale

## Why this round

The first CLM scaling round used the exact Stream-Gemma stride-10 pool, but its
head saw 24,300 balanced decisions from 9,107 training videos. The promoted
PACE-Qwen adapter was trained on 26,080 capped episodes from the same 10% pool,
with a much larger set of masked-in frame turns. The previous comparison
therefore tested head architecture at a smaller, different supervision mix. Its
best CLM did not pass the across-seed quality gate; see
`pace_clm_10pct_scaling_plan.md` and the external `results/RESULTS.md`.

| | PACE 10% pilot | Earlier 24.3k-decision CLM run |
|---|---:|---:|
| Training unit | 26,080 sampled episodes, up to 32 frames each | 24,300 selected frame decisions |
| Supervised decision positions | 263,760 masked-in frame turns | 24,300 per model/seed |
| Model updated | Qwen LoRA adapter | CLM or MLP head; Qwen frozen |
| Optimizer schedule | 3,260 steps, one epoch | Dev-NLL early stopping, at most 60 epochs |

The 263,760 PACE positions comprise 215,502 Silence (81.70%), 7,038 Standby
(2.67%), and 41,220 Response (15.63%). By contrast, the earlier CLM head used
8,100 of each state (33.33% each). The full-frame candidate pool in that exact
10% cohort is about 91.7% Silence, 1.6% Hold, and 6.7% Response. Standby maps to
the head's Hold label. These are different sampling and loss distributions;
matching the video pool alone does not match training exposure.

The PACE pilot's episode construction is material: response-bearing chunks;
all Response turns learned; first and limited continuation turns learned from
Silence/Standby runs; other context turns masked at `-100`; trailing Silence
retained; and extra learned Silence windows 4–16 frames after Responses. The
26,080-episode cap uniformly samples built episodes. The earlier head selected
frames by class and simple boundary/offset strata, without replaying this mask.
The frozen backbone retains the PACE training, but the head loss does not.

## Shared protocol

1. Retain the exact 16,994-video stride-10 cohort and the existing
   video-disjoint split. **Each condition uses 24,300 training decisions**;
   the same 2,700-decision dev set is held fixed. Runs 1–2 reuse the existing
   frozen features. For runs 3–5, sample from the original training-video
   partition only, with the existing 14-decision-per-video cap and no overlap
   with dev or PACE-200. Freeze and hash each new decision manifest before
   extraction. Report achieved state and stratum counts, unique videos, and
   any repeated frame positions.
2. Reconstruct the PACE base and W1 `long-neg` `learn` masks on the same 10%
   videos for runs 3–5. Keep only mask-selected decisions for each condition;
   sample without replacement to the 24,300-decision budget. Use the same
   frozen PACE-Qwen checkpoint, causal 32-frame context, last-response memory,
   and final valid language-token pooling as the completed CLM run. Check that
   episode-specific prompts and history are represented consistently. Store
   large manifests, feature caches, checkpoints, and reports under
   `outputs/pace-clm-mixtures/` in the pace-clm repo. Prefer the
   assigned always-on eight-H200 node for extraction and training.
3. Train the capacity-matched MLP and short-label CLM-256 for each condition,
   with seeds 42/43/44, batch 128, AdamW at `3e-4`, weight decay `1e-3`, and
   the original 60-epoch maximum and six-epoch dev-NLL patience. Keep the
   original unweighted dev-NLL early stopping and identical 41-candidate
   Response-margin calibration for all conditions. For runs 1–2, normalize
   class weights to mean one. The completed balanced run is the control and
   must be evaluated alongside the new conditions.
4. Freeze two fresh, video-disjoint sets from Markii validation videos excluding
   PACE-200: a mixture-selection set for comparing the five conditions and an
   untouched final set for the later scale-up. Keep their natural state mixes.
   PACE-200 has already been inspected and must not select the winner. Report
   macro/per-state F1, NLL/ECE, no-judge PAUC-F and components, Timing F1@4s,
   fires/repeats per video, and seed variability. Inspect results by video
   length, distance to Response, post-response offset, and Hold-run offset.

## Five controlled runs

Runs 1–2 change only the effective loss prior on the existing 24,300 decisions.
Runs 3–5 change which 24,300 decisions are selected. The completed balanced
run remains the common control and is not counted as a sixth proposed run.

| Run | Training mixture | Controlled question |
|---|---|---|
| **1. PACE-prior weighting** | Reuse the existing balanced 24,300 decisions; weight their loss toward 81.7% Silence / 2.7% Hold / 15.6% Response. | Does the checkpoint's supervised firing prior help without changing selected frames? |
| **2. Full-frame-prior weighting** | Reuse those same decisions; weight toward the cohort's approximate 91.7% / 1.6% / 6.7% candidate-frame mix. | Is the natural frame prior preferable to the PACE mask prior? Compare with run 1. |
| **3. Base episode mask** | Sample 24,300 distinct decisions from the response-bearing `v25_last_response` learned turns before W1 trailing-Silence and post-response additions. Preserve its observed class proportions as closely as feasible (about 75.5% / 3.2% / 21.3%). | Does selecting PACE-masked frames help beyond class weighting? |
| **4. W1 long-negative mask** | Sample 24,300 decisions from the W1 learned turns, including trailing Silence and 4–16-frame post-response negatives; target the W1 proportions of 81.7% / 2.7% / 15.6%. | Do those targeted negatives improve firing behavior relative to run 3? |
| **5. W1 plus response-free Silence** | Start from run 4's 24,300-decision manifest. Replace 10% of its Silence decisions with Silence from response-free stretches of at least 32 frames, preferably from videos with no Response; retain its total and class counts. | Does sustained no-fire coverage help beyond W1's post-response negatives? |

Runs 3–4 need a preflight count of eligible unique decisions per state within
the original train-video partition. Run 5 additionally needs a count of long
response-free stretches. If a target mixture is infeasible at this budget,
record the maximum feasible mix before training rather than duplicate scarce
Hold decisions. Report effective sample size under runs 1–2's class weights.

## Select one mixture, then scale it

Treat the five as a data-mixture ablation, not five chances to tune on PACE-200.
On the mixture-selection set, rank conditions by mean no-judge PAUC-F across
three seeds. Prefer a gain over the completed balanced control that is
consistent across seeds, with no increase in fires/repeats and no material loss
of per-state recall. Use lower fires/video, then lower seed variance, to break
close PAUC-F results. Select the best mixture by this rule even if none clears
the control; report that outcome plainly. Freeze the selected mixture before
the next stage.

The second stage runs **that one selected mixture at PACE-like exposure**:
target 263,760 supervised decision presentations and 3,260 optimizer updates,
using the same 10% video pool and the selected sampling/weighting recipe. Train
both MLP and short-label CLM-256 with seeds 42/43/44, plus a balanced-loss
control on the same selected positions and budget. Preserve the mixture's state
shares; if scarce Hold positions must repeat, record unique positions, videos,
repetition, and effective sample size. Match the head-training budget across
these comparisons; this still does not equate frozen-head compute with PACE LoRA
training. Evaluate the scale-up once on the untouched final set, with identical
calibration rules. This phase tests whether the selected mixture's benefit
persists with more exposure and is separate from the five 24,300-decision runs.

Do not resume semantic judging or WEB/EGO until a method and mixture have been
selected without using those evaluations.

## Source records

- PACE W1 recipe and measured mask distribution:
  `../stream-gemma/agent_notes/docs/experiments/qwen_w1_long_neg.md`.
- Episode mask and negative-window implementation:
  `../stream-gemma/src/stream_gemma/episode.py`.
- Prior CLM decision selection and distribution:
  `scripts/build_10pct_manifests.py`,
  `manifests/pace_train_10pct_summary.json`.
- Prior head training and outcome:
  `scripts/train_10pct_curve.py`,
  `/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/results/RESULTS.md`.
