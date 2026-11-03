# J1-S selector with Gemini 3.8 Flash answers

Date: 2026-11-02; EGO result added 2026-11-03. WEB and EGO complete.

## Question and setup

This is a closed-loop generator swap for the frozen J1-S selector at
`outputs/jevany-j1s/full-seed42/step-008243`. Qwen and its J1-S pointer head
still choose `Silence`, `Hold`, or `Response` at 1 FPS with a 32-frame causal
window and zero response margin. At each `Response`, Vertex Gemini 3.8 Flash
(`gemini-3.8-flash`, `global`) receives the standing request, observed frames
through the current timestamp, and prior emitted answers. Gemini writes one
concise answer; that text enters J1-S's later controller history. The
promoted PACE Qwen generator adapter is not loaded in this cell.

The Gemini request uses temperature zero, 384-pixel JPEG frames, a 128-token
thinking budget, and a 96-visible-token initial answer budget. An HTTP-200
empty answer is retried once with a 256-visible-token budget. If the retry is
also empty, the emitted answer is `I cannot determine the answer from the
visible frames.` and the fallback is marked in the per-video `gemini_calls`
trace. Non-200 failures remain fatal. The [rollout manifest](../../outputs/jevany-j1s/evaluation/benchmarks/web-gemini38/full/manifest.json)
pins the selector hashes and generation protocol.

The first WEB rollout stopped after 342 completed videos when Gemini returned
an empty answer. It resumed from those saved rows with the empty-answer policy
above; no partial score was run. The [resume note](../../outputs/jevany-j1s/evaluation/benchmarks/web-gemini38/resume_note.json)
records the change. The completed 500-video rollout had eight empty-answer
retries, all of which used the explicit unavailable fallback, and no non-200
Gemini calls.

## Full WEB result

GPT-4.1 judged `pauc-f-v1`, `omega=0.5`, on all 500 WEB videos and 1,328 gold
turns. `R` is the scorer's PAUC recall component.

| Controller and answer generator | P | R | PAUC-F | Fires | Useful | Wasted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1-S + promoted PACE Qwen | 65.36% | 49.77 | 56.51 | 1,357 | 887 | 470 |
| J1-S + Gemini 3.8 Flash | 66.08% | 52.22 | **58.34** | 1,365 | 902 | 463 |
| Promoted PACE `long-neg` (historical) | 70.67% | 48.43 | 57.47 | 1,132 | 800 | 332 |

Gemini gained 1.83 PAUC-F points over J1-S with promoted answers, with eight
more physical fires, 15 more useful fires, and seven fewer wasted fires. It
was 0.87 points above the historical promoted PACE WEB score. The result
averaged 2.73 responses per video.

Sources: [Gemini predictions](../../outputs/jevany-j1s/evaluation/benchmarks/web-gemini38/full/predictions.jsonl),
[Gemini GPT-4.1 summary](../../outputs/jevany-j1s/evaluation/benchmarks/web-gemini38/full/pauc_f_summary-gpt-4.1-v1.json),
and [J1-S GPT-4.1 summary](../../outputs/jevany-j1s/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json).
The promoted PACE figures are historical point estimates documented in the
[J1-S experiment note](jevany_response_blind_next_three.md).

This is a controller-generator pair result: Gemini's answer wording can
change later J1-S decisions, as shown by the earlier
[generator-feedback analysis](controller_generator_feedback.md). It does not
isolate answer quality at fixed fire times. No paired significance interval
has been computed for this swap. WEB has informed prior experiment choices.
The eight explicit unavailable fallbacks are included in the WEB score.

## Full EGO result

The same frozen selector and Gemini protocol completed all 326 EGO videos.
GPT-4.1 judged `pauc-f-v1`, `omega=0.5`, on 1,575 gold turns.

| Controller and answer generator | P | R | PAUC-F | Fires | Useful | Wasted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1-S + promoted PACE Qwen | 35.77% | 50.55 | 41.90 | 2,555 | 914 | 1,641 |
| J1-S + Gemini 3.8 Flash | 21.57% | 51.23 | **30.36** | 4,089 | 882 | 3,207 |
| Promoted PACE `long-neg` (historical) | 46.14% | 45.96 | 46.05 | 1,634 | 754 | 880 |

Gemini lowered EGO PAUC-F by 11.54 points relative to J1-S with promoted
answers. It made 1,534 more responses, including 1,566 more wasted responses
and 32 fewer useful responses. The response rate rose from 7.84 to 12.54 per
video. The PAUC recall component increased only 0.67 points while fire
precision fell by 14.20 percentage points. No EGO empty-answer retry or
fallback occurred; all recorded Gemini calls returned HTTP 200.

The initial EGO rollout stopped at 186 saved videos after Triton compilation
filled the node's root `/tmp`. A restart first hit a manifest read/write race;
the evaluator now writes the manifest by atomic replacement. The successful
continuation used `/dev/shm` for Triton compilation and resumed the saved
per-rank predictions. These were execution repairs, not selector or generator
recipe changes.

Sources: [EGO predictions](../../outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/full/predictions.jsonl),
[Gemini GPT-4.1 summary](../../outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/full/pauc_f_summary-gpt-4.1-v1.json),
and [J1-S GPT-4.1 summary](../../outputs/jevany-j1s/evaluation/benchmarks/ego/pauc_f_summary-gpt-4.1-v1.json).

**Decision:** Do not promote this pair across WEB/EGO. Its WEB gain did not
generalize to EGO, where closed-loop firing rose sharply and judged PAUC-F
fell below both comparison controllers. The result does not establish whether
the cause is Gemini answer wording, answer accuracy, or their interaction
with J1-S history; a fixed-schedule answer replay would separate answer
quality from controller feedback.
