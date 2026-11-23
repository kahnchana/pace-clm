# RL checkpoint ESTP-Bench explicit PAUC-F

Saved predictions only; no new model inference. User authorized additional GPT-4.1 rescoring on 2026-11-16.
Unchanged stream-gemma estp-pauc-f-v1: 851 cases, strict native positive-width windows, omega 0.5, eight-task macro-average.
The 71 zero-duration gold turns are excluded by this PAUC protocol. Official ESTP-F1 outputs remain separate and unchanged.

| Task | Step 32 | Step 128 | Change |
|---|---:|---:|---:|
| Object Recognition | 27.854 | 27.250 | -0.604 |
| Attribute Perception | 30.425 | 30.815 | +0.391 |
| Text-Rich Understanding | 27.315 | 28.704 | +1.389 |
| Object Localization | 32.111 | 32.681 | +0.571 |
| Object State Change Recognition | 19.696 | 19.513 | -0.184 |
| Ego Object Localization | 27.284 | 27.222 | -0.062 |
| Ego Object State Change Recognition | 14.030 | 15.750 | +1.720 |
| Action Recognition | 22.527 | 23.747 | +1.220 |
| **Explicit All (task macro)** | **25.155** | **25.710** | **+0.555** |

Task-level precision, temporal recall, useful/wasted fires, source hashes, requests and raw judge responses are retained in each endpoint directory under estp_pauc_explicit/.
These are reused benchmark comparisons, not untouched tests or a step-160 result. No further model evaluations are queued.
