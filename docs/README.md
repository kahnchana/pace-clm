# Documentation

Start with the [project README](../README.md) for current status and the pinned
model contract. Documents here are grouped by purpose:

| Directory | Contents |
| --- | --- |
| [Reference](reference/) | Audited design and source contracts |
| [Plans](plans/) | Proposed experiments and follow-up work |
| [Experiments](experiments/) | Completed results, analyses, resource measurements, and the chronological [experiment log](experiments/experiment_log.md) |

The [GPU optimizations](reference/gpu_optimizations.md) page summarizes the
Qwen3.5-2B training changes and links to their experiment reports.
The [consolidated 2B full-data findings](experiments/qwen35_2b_full_data_findings.md)
cover the GPU measurements, stopped training run, and WEB checkpoint results.
The proposed [controlled 2B scaling plan](plans/qwen35_2b_controlled_scaling_plan.md)
tests representative ordering, fixed global batches, and fresh data against
matched replay before another full-manifest run.
The [completed two-node mixture experiment](experiments/qwen35_2b_mixture_12h_results.md)
records both arms' configuration, all eight judged WEB evaluations, exposure,
runtime measurements, checkpoint fluctuations, uncertainty and next-step
recommendations. The final PAUC-F scores are 48.70 for the existing-order
control and 47.61 for representative ordering; unequal actual budgets and
runtime differences prevent a clean causal A/B comparison.
The [original execution plan](plans/qwen35_2b_12h_mixture_experiments.md)
proposed 12 training hours per arm. The user amended execution to frozen,
preflight-estimated step counts: actual training took 13.533 hours for the
[control](experiments/qwen35_2b_control_12h.md) and 8.914 hours for the
[representative arm](experiments/qwen35_2b_representative_12h.md).
The [earlier mixture proposal](plans/qwen35_2b_data_mixture_plan.md) and
controlled scaling plan remain design history and proposed follow-ups, not
evidence that larger-scale training has passed a quality gate.

The [2B J1-S WEB result](experiments/qwen35_2b_j1s_web.md) and
[Qwen3.5 size sweep](experiments/qwen35_dense_size_sweep_j1s_web.md) cover the
recent controller work. The older frozen-head study is indexed in
[agent notes](../agent_notes/overview.md).

The [completed first 2B RL pilot](experiments/jevany_2b_rl_e1_results.md)
records the 32-update RLOO-style run, its 1,024-span / 131,072-decision exposure,
hardware measurements, and full GPT-scored WEB results at steps 16 and 32.
PAUC-F was 53.31 and 53.09 versus SFT at 54.67: fewer wasted responses came
with lower useful coverage. The report distinguishes the implemented proxy
reward from PAUC-F and RLOO from standard GRPO. Subsequent reward-aligned
RLOO and fixed GRPO pilots both stopped at round 16 for lost useful coverage;
their full-WEB scores were 54.5561 and 53.7149. The
[completed supervised-anchor RLOO pilot](experiments/jevany_2b_rl_e3_anchor.md)
used a small training-only CE term and raw learned inference. All four full-WEB
evaluations finished: round 24 scored 54.8260 with 701 useful / 280 wasted
responses; round 32 regressed to 54.3782. The best point gain is not conclusive
because its paired confidence interval includes zero. The
[RL implementation plan](plans/jevany_2b_rl_eight_h200_plan.md) retains the
profiling and execution history.

The proposed [episode-level RLOO plan](plans/jevany_2b_episode_level_rloo.md)
tests continuous self-generated history and episode PAUC-F rewards on the same
eight GPUs. It is a plan, not a launched experiment.

Put stable implementation contracts in `reference/`, proposed work in
`plans/`, and observed results in `experiments/`. Record a document's date and
status within the document.
