# Experiment status

PAUC-F and F1 are percentages. Scores from different evaluation sets should not be compared directly.

| Completed work | Evidence and result | Decision |
|---|---|---|
| [PACE-200 pilot](pace200_clm_pilot.md) | Trained heads on 768 balanced decisions. Best CLM no-judge PAUC-F was 65.14, versus 80.10 for PACE token generation. | Do not advance the pilot CLM unchanged. |
| [10% learning curve](pace_clm_10pct_scaling_plan.md) | Froze 16,994 training videos and 27,000 decisions; completed 60 runs across four data sizes, five heads, and three seeds. On PACE-200, short-label CLM scored 81.46 PAUC-F / 70.81 macro-F1 versus MLP's 81.31 / 71.69. | CLM failed the prespecified across-seed quality gate. |
| [Six-condition mixture comparison](pace_clm_10pct_mixture_results.md) | Compared control and runs 1–5 at 24,300 decisions, with MLP and short-label CLM across three seeds, on a fresh 200-video selection set. Run 5 led on mean PAUC-F (82.98 across heads versus 80.17 for control), but increased fires and repeat fires and lost Hold recall. | Selected run 5 for the prespecified scale-up; its behavioral tradeoffs remain unresolved. |
| [PACE-like exposure scale-up](pace_clm_10pct_mixture_results.md) | Trained the selected mixture for 263,760 decision presentations and 3,260 updates. On the untouched final set, mean PAUC-F across heads and seeds was 77.45, versus 79.69 for frozen 24.3k run 5 and 79.24 for the original control. | More exposure reduced PAUC-F by 2.24 points relative to 24.3k run 5; no scaled head is ready to advance. |

| Next direction | Why it matters | First step |
|---|---|---|
| Calibrated MLP or temporal gate distilled from PACE token decisions | CLM has not shown a repeatable advantage over MLP, and mixture gains came with extra fires and repeats. | Define a firing or gating rule using training/dev data, then compare quality and latency against the frozen MLP and PACE token baselines on a new held-out set. |
| Diagnose the scale-up regression | The same run 5 mixture lost 2.24 PAUC-F points with PACE-like exposure on the final set. | Compare checkpoint trajectories, calibration, and per-state/timing errors at intermediate exposure budgets before another full scale-up. |
| Refine the mixture for firing behavior | Run 5 improved selection PAUC-F but missed the planned fires/repeats and Hold-recall preferences. | Use a new development split to test targeted hard negatives and Hold coverage; reserve a fresh final set for one frozen comparison. |

Mixture artifacts, per-seed metrics, and completion marker: [`outputs/pace-clm-mixtures`](../../outputs/pace-clm-mixtures/). Semantic judging and WEB/EGO remain deferred pending a frozen method that clears the quality gate.
