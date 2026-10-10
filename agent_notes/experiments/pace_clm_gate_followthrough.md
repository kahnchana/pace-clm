# PACE-CLM gate and scale-up follow-through (2026-10-02)

## Protocol

Two new 200-video Markii validation cohorts were frozen before this round by
skipping the first 400 hash-ordered eligible videos used by the earlier mixture
selection and final sets. The new development and final cohorts contain 11,563
and 12,658 one-FPS decisions, respectively. They are video-disjoint from each
other, the earlier PACE-200 and mixture cohorts, and the 10% training pool.
The final manifest SHA-256 is
`61de60842a3e889752830d8293cfa30c3cfca5944720f1b6295b8b4a2f0f4f9a`.
All methods use the promoted frozen PACE-Qwen adapter, the existing causal
streaming feature contract, gold prior-action history, and the unchanged
Stream-Gemma timing-only PAUC-F scorer.

The development sweep considered the balanced MLP, the frozen 24.3k run 5 MLP,
an early checkpoint replay of the long run, and run 5 with doubled Hold loss.
For each, it searched seven Response-margin offsets and cooldowns of 0, 2, 4,
or 8 frames across seeds 42/43/44. Eligibility required no more than 0.05
additional fires/video, two additional repeat fires per 200 videos, or two
percentage points less Hold recall than the balanced MLP on the new development
set. The development winner was the balanced MLP with a +0.5 Response margin
and no cooldown: 77.88 PAUC-F and 2.66 fires/video, versus 77.71 and 3.01 for
its original margin. This choice was frozen before final scoring.

## Final timing-only result

Head values are means across three seeds. PAUC-F, macro F1, and Hold recall are
percentages; repeats are mean counts over 200 videos. PACE token generation is
one deterministic run. Results from the prior 200-video final set are not mixed
into this table.

| Method | PAUC-F | Macro F1 | Hold recall | Fires/video | Repeats/200 videos |
|---|---:|---:|---:|---:|---:|
| PACE token generation | 82.08 | 86.94 | 81.8 | 4.43 | 15.0 |
| Balanced MLP, original margin | 81.92 ± 0.43 | 79.65 | 99.6 | 3.44 | 0.0 |
| **Selected balanced MLP, +0.5 margin** | **81.42** | **78.90** | **99.6** | **3.11** | **0.0** |
| Frozen run 5 MLP | 83.38 ± 0.20 | 87.26 | 90.4 | 3.77 | 4.7 |
| Frozen run 5 short-label CLM | 82.59 | 85.99 | 92.2 | 3.89 | 5.7 |
| Early replay run 5 MLP | 82.48 | 85.80 | 92.9 | 3.96 | 5.3 |
| Doubled-Hold-loss run 5 MLP | 82.61 | 86.50 | 93.2 | 3.92 | 1.7 |
| PACE-exposure run 5 MLP | 81.01 | 85.74 | — | 4.05 | 11.0 |

The development-selected margin reduced final fires from 3.44 to 3.11/video
but lowered PAUC-F by 0.50 points. The run 5 MLP exceeded token generation
by 1.31 PAUC-F points and fired 0.66 fewer times/video. Its timing fire
precision was 84.20% versus 77.51% for token generation, while its PAUC recall
component was 82.63% versus 87.21%. Run 5 traded away 9.2 Hold-recall points
and added 4.7 repeat fires per 200 videos relative to the balanced MLP, yet
both measures remained better than token generation on this final cohort.

A 10,000-replicate paired video bootstrap of the three-seed mean run 5 MLP
minus token generation yielded a +1.31-point PAUC-F difference with a 95%
interval of [+0.17, +2.50]. The analogous run 5-minus-balanced-MLP interval
was [+0.08, +2.93]. These intervals describe this 200-video timing-only cohort;
they do not cover semantic response quality or method-selection uncertainty.

## Scale-up diagnosis and mixture check

The previous PACE-exposure run made 263,760 presentations and 3,260 updates
per head/seed by repeating 24,300 unique positions 10–11 times. Across the six
MLP/CLM runs, the best logged internal-dev NLL occurred at step 483 on average,
and the final-step NLL was 0.205 higher. Exact early replay checkpoints were
retained at each run's minimum logged dev NLL (steps 200–900). On this new final
set, early replay MLP scored 82.48 PAUC-F and the full-exposure MLP 81.01,
versus 83.38 for the original early-stopped 24.3k run 5 MLP. More repetition
did not improve generalization; the internal-dev NLL minimum alone did not
recover the best final result.

Doubling Hold loss on run 5's existing hard-negative and response-free-Silence
positions raised final MLP Hold recall from 90.4% to 93.2% and reduced mean
repeat fires from 4.7 to 1.7 per 200 videos. It lowered PAUC-F from 83.38 to
82.61 and raised fires/video from 3.77 to 3.92. The frozen short-label CLM
remained below the corresponding MLP on PAUC-F, and no CLM variant cleared the
prespecified advantage over a capacity-matched classifier.

## Decision and limits

The new calibrated gate did not generalize as a PAUC-F improvement. The frozen
run 5 **MLP** is the strongest practical candidate from this round: it beat
PACE token generation on timing PAUC-F, macro F1, Hold recall, firing frequency,
and repeat fires on a new held-out cohort. It remains an exploratory candidate
because this round compared several frozen variants and timing-only scoring
does not judge generated response content. No new Qwen LoRA or end-to-end VLM
training was launched. The earlier 32-decision paired serving benchmark for
the same balanced MLP path measured about 1.12 s median versus 1.50 s for
four-token PACE generation; new run 5 serving latency was not measured here.

Next, freeze run 5 MLP and its existing dev-selected margin, run judged
response-quality evaluation with the original PACE generation handoff, and
confirm the quality and latency tradeoff on a separate held-out cohort before
any deployment claim. Continue to defer WEB/EGO until that confirmation.

Artifacts: `outputs/pace-clm-gate/` contains exact manifests, features,
checkpoints, per-seed and per-video scores, token predictions, bootstrap
results, logs, and `results/complete.done`. The execution script is
`scripts/finish_gate_followthrough.sh`.
