# Documentation

Start with the [project README](../README.md) for current status and the pinned
model contract. Documents here are grouped by purpose:

| Directory | Contents |
| --- | --- |
| [Reference](reference/) | Audited design and source contracts |
| [Plans](plans/) | Proposed experiments and follow-up work |
| [Experiments](experiments/) | Completed results, analyses, resource measurements, and the chronological [experiment log](experiments/experiment_log.md) |

The [provisional Qwen3.5-2B full-data training procedure](reference/qwen35_2b_full_data_training.md)
records the tested node setup and checks for future runs.

The [2B J1-S WEB result](experiments/qwen35_2b_j1s_web.md) and
[Qwen3.5 size sweep](experiments/qwen35_dense_size_sweep_j1s_web.md) cover the
recent controller work. The older frozen-head study is indexed in
[agent notes](../agent_notes/overview.md).

Put stable implementation contracts in `reference/`, proposed work in
`plans/`, and observed results in `experiments/`. Record a document's date and
status within the document.
