# Qwen3.5-2B J1-S controller: full WEB result

Date: 2026-11-04. The [matched-run plan](../plans/qwen35_2b_j1s_web_plan.md)
was completed with seed 42 and the final one-epoch checkpoint at step 8,243.
The controller started from official post-trained `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, with a fresh LoRA and
pointer head. The frozen promoted 35B MoE generated answers. The training
manifest SHA-256 was
`89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222`
(263,760 presentations). Training took 8.07 hours, including resumes.

All 500 unique WEB videos were evaluated at response margin 0 with actual
controller history, 32 causal frames, and greedy promoted-MoE answers capped
at 96 tokens. GPT-4.1 judging used `pauc-f-v1`, omega 0.5.

| WEB metric | 2B controller | Historical 35B MoE J1-S |
| --- | ---: | ---: |
| Judged PAUC-F | 54.67 | 56.51 |
| Fire precision | 70.74% | 65.36% |
| PAUC recall component | 44.54 | 49.77 |
| Fires | 991 | 1,357 |
| Useful / wasted fires | 701 / 290 | 887 / 470 |

The 2B controller fired 1.98 times per video. Its PAUC-F was 1.85 points below
historical J1-S and 2.80 below the promoted PACE token policy (57.47). A
paired, 1,000-replicate WEB bootstrap against stored J1-S predictions gave a
95% interval of **[-3.31, -0.33] points** for the 2B minus J1-S difference.
The 2B controller reduced wasted fires but lost recall. Historical J1-S
started from a PACE-trained adapter, while 2B started from official weights;
the difference cannot be attributed to parameter count alone.

Measured controller decision time was 0.496 seconds per decision; promoted-MoE
generation took 3.409 seconds per fire. Peak allocated GPU memory was 7.36 GiB
during 2B training and 71.72 GiB during two-model evaluation.

The ignored `outputs/` tree contains the [full report](../../outputs/jevany-qwen35-2b-j1s/reports/web_result.md),
[machine-readable summary](../../outputs/jevany-qwen35-2b-j1s/reports/web_result.json),
[final checkpoint](../../outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243),
[merged predictions](../../outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web/predictions.jsonl),
and [judged score](../../outputs/jevany-qwen35-2b-j1s/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json).
