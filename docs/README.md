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

The [2B J1-S WEB result](experiments/qwen35_2b_j1s_web.md) and
[Qwen3.5 size sweep](experiments/qwen35_dense_size_sweep_j1s_web.md) cover the
recent controller work. The older frozen-head study is indexed in
[agent notes](../agent_notes/overview.md).

Put stable implementation contracts in `reference/`, proposed work in
`plans/`, and observed results in `experiments/`. Record a document's date and
status within the document.
