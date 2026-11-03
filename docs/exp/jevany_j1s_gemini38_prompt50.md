# Gemini 3.8 Flash answer-prompt check on 50 EGO videos

Date: 2026-11-03. Status: four two-GPU rollouts restarted after Vertex
authentication was refreshed; GPT-4.1 judging is waiting for generation.
The frozen cohort and baseline scores are complete. The first attempt stopped
at the first Gemini call without saving a video.

## Purpose and cohort

The [full EGO generator swap](jevany_j1s_gemini38_generator.md) scored 30.36
GPT-4.1 judged PAUC-F with Gemini answers, versus 41.90 with promoted Qwen
answers. J1-S first fired at the same time with both generators in all 319
videos where it fired. Subsequent Gemini-path responses often arrived one
second apart. This cell asks whether four answer-prompt changes can reduce
that closed-loop overfiring without changing J1-S weights, response margin,
frames, answer API model, or scorer.

The 50 EGO videos are a fixed-seed (`42`) random sample from the completed
326-video split. The [selection audit](../../outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50/selection_audit.json)
records 18 videos where the original Gemini path fired more than the promoted
Qwen path, 14 with equal fire counts, and 18 where it fired fewer. This is a
diagnostic subset of a benchmark already used to form the hypotheses. Its
scores should not be treated as untouched full-EGO estimates.

Existing GPT-4.1 judgments were filtered to the identical 50 video IDs for
the baseline rows. The baseline scores are:

| Selector and generator | P | R | PAUC-F | Fires | Useful | Wasted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1-S + promoted Qwen | 40.62% | 53.66 | 46.24 | 389 | 158 | 231 |
| J1-S + Gemini baseline prompt | 31.59% | 53.61 | 39.75 | 478 | 151 | 327 |

The [frozen IDs](../../outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50/selected_ids.json)
and baseline scored artifacts are saved under
`outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50/`.

## Four independent prompt additions

Each variant retains the original Gemini causal-frame prompt and changes only
its system instruction with one of the following additions. The exact text
is versioned in `scripts/eval_jevany_streaming.py` and saved in each rollout
manifest.

1. `event_boundary`: treat uninterrupted action as one event; reuse the
   earlier sentence for continuation; describe a new action only at onset or
   resumption.
2. `no_premature_absence`: report positively visible actions only; avoid
   anticipation and variable “not yet” answers; use one fixed sentence when
   the requested action is absent.
3. `stable_action_words`: keep actor, verb, and object wording stable for the
   same action; avoid minor tool, location, and progress variations.
4. `current_event_only`: answer with the requested action visible at the
   current time; avoid cumulative recaps and irrelevant background activity.

The four rollouts run concurrently on GPU pairs 0–1, 2–3, 4–5, and 6–7.
Each uses the frozen J1-S checkpoint, zero response margin, Vertex
`gemini-3.8-flash` at the global endpoint, and the WEB/EGO answer handling
from the full run. GPT-4.1 `pauc-f-v1`, `omega=0.5`, judges all completed
responses. The final matched comparison will be saved as
`outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50/comparison.md`.

## Results

Judging is pending.
