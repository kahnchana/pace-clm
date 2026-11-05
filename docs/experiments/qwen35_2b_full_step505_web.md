# Qwen3.5-2B full-data step-505 WEB evaluation

Date: 2026-11-05. Training was stopped at the user's request on
`ip-10-3-120-167`; all eight local GPUs were then used for WEB evaluation.
The latest durable controller checkpoint was
[step 505](../../outputs/jevany-qwen35-2b-full/train/step-000505), covering
426,464 of 8,343,088 planned training presentations (5.11%). This is an
intermediate checkpoint, not a completed full-data epoch.

The evaluator used the frozen promoted generator, response margin 0,
32 causal frames, up to 96 generated tokens, and all 500 WEB videos. Its
[protocol manifest](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/manifest.json)
matches the earlier [2B J1-S WEB protocol](qwen35_2b_j1s_web.md) in every
field except the controller adapter and pointer-head hashes. The unchanged
GPT-4.1 `pauc-f-v1` judge scored all 500 unique merged predictions.

| WEB metric | Full-data step 505 | Earlier 2B J1-S final | Difference |
|---|---:|---:|---:|
| Judged PAUC-F | **47.12** | 54.67 | **-7.55** |
| Fire precision | 59.56% | 70.74% | -11.18 points |
| PAUC recall component | 38.97 | 44.54 | -5.57 |
| Fires | 863 | 991 | -128 |
| Value-adding fires | 514 | 701 | -187 |
| Wasted fires | 349 | 290 | +59 |

The step-505 controller is worse on this matched WEB protocol. It fires less
often, but also produces fewer useful fires and more wasted fires. The
training recipes differ in data selection, global batch size, and completed
optimizer updates (505 versus 8,243), so this result does not isolate the
effect of any one training change or predict the final full-data score.

Artifacts: [merged predictions](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/predictions.jsonl),
[judged score](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/pauc_f_summary-gpt-4.1-v1.json),
[evaluation log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_eval.log),
and [scoring log](../../outputs/jevany-qwen35-2b-full/logs/web_step505_score.log).
Training remains stopped after this evaluation.
The [early-checkpoint comparison](qwen35_2b_full_checkpoint_web_comparison.md)
also includes complete WEB results for steps 10 and 25.
