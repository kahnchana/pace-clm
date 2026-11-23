# Step-160 ESTP-Bench explicit proactive PAUC-F

Completed 2026-11-16 on ip-10-3-120-167 (allocation 38177), using eight H200 GPUs for inference and the authorized GPT-4.1 gateway for judging.
Only the 851-case explicit proactive split was evaluated. No official ESTP-F1 or other benchmark evaluations were launched.
Unchanged stream-gemma estp-pauc-f-v1, omega 0.5, strict native positive-width windows and unweighted eight-task macro-average. The 71 zero-duration gold turns are excluded by the PAUC protocol.
Controller: committed mixed-web-ego-estp-balanced-v4-to160/step-000160. Raw learned argmax verified on all saved action traces; response margin zero. Same promoted answer generator and 96-token limit as prior endpoints.

| Task | Step 160 PAUC-F |
|---|---:|
| Object Recognition | 27.402 |
| Attribute Perception | 30.697 |
| Text-Rich Understanding | 26.177 |
| Object Localization | 32.707 |
| Object State Change Recognition | 20.985 |
| Ego Object Localization | 28.399 |
| Ego Object State Change Recognition | 15.782 |
| Action Recognition | 24.313 |
| **Explicit All (task macro)** | **25.808** |

Change versus the existing step-128 result (25.710): +0.097 PAUC-F points. These are reused benchmark comparisons, not untouched tests.
Scorer, official scorer, annotation and judge prompt hashes match the step-128 explicit evaluation.
All 625 unique GPT requests completed (827 requests before deduplication); raw requests, responses, predictions, traces and source hashes are retained.

Artifacts: [evaluation result](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/estp-explicit-step160/result.json), [PAUC-F summary](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/estp-explicit-step160/estp_pauc_explicit/pauc_f_summary-gpt-4.1.json).
