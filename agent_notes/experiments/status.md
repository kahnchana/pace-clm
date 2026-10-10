# Experiment status

PAUC-F and F1 are percentages. Scores from different evaluation sets should not be compared directly.

| Completed work | Evidence and result | Decision |
|---|---|---|
| [PACE-200 pilot](pace200_clm_pilot.md) | Trained heads on 768 balanced decisions. Best CLM no-judge PAUC-F was 65.14, versus 80.10 for PACE token generation. | Do not advance the pilot CLM unchanged. |
| [10% learning curve](pace_clm_10pct_scaling_plan.md) | Froze 16,994 training videos and 27,000 decisions; completed 60 runs across four data sizes, five heads, and three seeds. On PACE-200, short-label CLM scored 81.46 PAUC-F / 70.81 macro-F1 versus MLP's 81.31 / 71.69. | CLM failed the prespecified across-seed quality gate. |
| [Six-condition mixture comparison](pace_clm_10pct_mixture_results.md) | Compared control and runs 1–5 at 24,300 decisions, with MLP and short-label CLM across three seeds, on a fresh 200-video selection set. Run 5 led on mean PAUC-F (82.98 across heads versus 80.17 for control), but increased fires and repeat fires and lost Hold recall. | Selected run 5 for the prespecified scale-up; its behavioral tradeoffs remain unresolved. |
| [PACE-like exposure scale-up](pace_clm_10pct_mixture_results.md) | Trained the selected mixture for 263,760 decision presentations and 3,260 updates. On the untouched final set, mean PAUC-F across heads and seeds was 77.45, versus 79.69 for frozen 24.3k run 5 and 79.24 for the original control. | More exposure reduced PAUC-F by 2.24 points relative to 24.3k run 5; no scaled head is ready to advance. |
| [Fresh gate and scale-up follow-through](pace_clm_gate_followthrough.md) | On a new 200-video final set, frozen run 5 MLP scored 83.38 mean timing PAUC-F across three seeds versus 82.08 for PACE token generation; it fired 3.77 versus 4.43 times/video. The new dev-selected balanced-MLP gate scored 81.42, below its unmodified 81.92. Early replay and doubled Hold loss did not beat run 5. | Freeze run 5 MLP as an exploratory candidate for judged response quality and an independent confirmation; stop expanding the CLM head or exposure budget. |
| [Full WEB/EGO transfer](pace_clm_web_ego_full.md) | Frozen run 5 MLP seed 42 completed WEB 500/500 and EGO 326/326. GPT-4.1 judged PAUC-F was 53.50 WEB and 43.95 EGO, versus promoted PACE Qwen's 57.47 and 46.05. Candidate timing-only upper bounds were 63.58 and 48.79. | The MLP reduced fires but lost judged recall and headline PAUC-F on both splits. Do not claim WEB/EGO improvement over PACE. |

| Next direction | Why it matters | First step |
|---|---|---|
| Reassess the frozen run 5 MLP | Its +1.31-point timing-only gain on a separate Markii cohort did not transfer to judged WEB/EGO PAUC-F. | Investigate missed gold turns and response quality before another model iteration; retain the original PACE decision baseline. |
| Reconcile firing preferences | Run 5 has more fires and lower Hold recall than the balanced MLP, while both are better than token generation on the new final cohort. | State the comparator and acceptable firing/recall thresholds before the next evaluation. |
| CLM and long-exposure training | CLM has no repeatable advantage over MLP; 3,260-update repetition and exact early replay both underperformed the 24.3k run 5 MLP. | Do not spend another large training budget on these head recipes without a new hypothesis. |

Latest gate artifacts, per-seed metrics, token predictions, bootstrap, and completion marker: [`outputs/pace-clm-gate`](../../outputs/pace-clm-gate/). Full WEB/EGO predictions, judge responses, judged scores, and timing-only diagnostics: [`outputs/pace-clm-web-ego-full`](../../outputs/pace-clm-web-ego-full/).
