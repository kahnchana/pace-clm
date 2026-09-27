# PACE-200 CLM pilot

| What we tried | Configuration |
|---|---|
| Frozen representation | Promoted PACE-Qwen (`Qwen/Qwen3.5-35B-A3B` + `lora-qwen-w1-long-neg`), final valid language token, 32-frame strictly causal window, common gold prior-action history |
| Training data | Deterministic balanced PACE-train subset: 768 examples, 256 each of Response, Silence, and Hold |
| Validation data | Frozen PACE-200: 10,658 decisions across 200 videos; 670 Response, 9,937 Silence, and 51 Hold |
| Required comparisons | Existing PACE token generation, raw state/action cosine, linear classifier, capacity-matched MLP, and separate-head CLM |
| CLM variants | Projection dimensions 128/256/512, shared versus separate heads, descriptive actions versus short labels |
| Repeats | Three seeds for trained heads; deterministic single runs for raw cosine and PACE token generation |
| Correctness gate | CLM tiny-set overfit reached 100% accuracy on 24 balanced examples |
| Conclusion | Existing PACE token generation remains clearly strongest. CLM does not beat the token baseline or ordinary classifiers overall; among CLM variants, short labels give the best mean state metrics, while 256-d gives the best seed-42 no-judge PAUC-F. Do not advance this CLM design unchanged to WEB/EGO. |

| Method | Accuracy ↑ | Macro-F1 ↑ | Response F1 ↑ | Hold F1 ↑ | Timing F1@4s ↑ | PAUC-F (no judge) ↑ | Fires/video ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw cosine | 8.92 | 5.92 | 0.30 | 1.05 | 0.30 | 33.43 | 0.01 |
| Linear | 90.71 ± 0.52 | 66.69 ± 0.48 | 57.73 ± 1.52 | 47.41 ± 1.55 | 58.87 ± 1.63 | 59.01 | 7.49 |
| MLP | 90.97 ± 0.37 | **67.12 ± 0.60** | 58.63 ± 1.02 | **47.69 ± 0.56** | 59.87 ± 1.15 | 60.96 | 7.37 |
| CLM 128-d | 91.55 ± 0.12 | 66.87 ± 1.11 | 60.17 ± 0.99 | 45.01 ± 4.34 | 61.66 ± 1.05 | 61.11 | **6.91** |
| CLM 256-d | 91.12 ± 1.05 | 66.64 ± 0.49 | 59.04 ± 3.70 | 45.72 ± 4.81 | 60.41 ± 3.87 | **65.14** | 7.20 |
| CLM 512-d | 90.94 ± 0.55 | 66.08 ± 0.93 | 58.54 ± 1.54 | 44.65 ± 1.85 | 59.95 ± 1.67 | 59.36 | 7.31 |
| CLM shared head | 90.70 ± 0.44 | 63.85 ± 1.96 | 58.84 ± 1.45 | 37.79 ± 6.34 | 60.39 ± 1.42 | 59.14 | 7.18 |
| CLM short labels | **91.62 ± 0.60** | 67.06 ± 0.89 | **60.49 ± 1.75** | 45.23 ± 1.26 | **61.89 ± 1.94** | 64.46 | 6.93 |
| Existing PACE token generation | **96.80** | **83.70** | **76.77** | **76.00** | **78.19** | **80.10** | 3.71 |
