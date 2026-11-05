# Qwen3.5-2B GPU utilization: top three TODOs

Status: short probes for all three items are complete on `ip-10-3-70-177`.
See the [follow-up measurements](qwen35_2b_current_node_gpu_util_followup.md).
CPU preparation prefetch reached 80.98% mean sampled utilization on its
matched benchmark window; the other two probes did not improve on that
utilization result.

Date: 2026-11-05. Scope: the eight H200 GPUs on `ip-10-3-120-167`.
These are follow-up experiments for the [full-data run](qwen35_2b_full_data_gpu_optimization.md),
not changes to its active training process. Its one-minute live sample averaged
52.3% GPU utilization; 32.7% of readings were at or below 5%. Peak sampled
memory was already 99.3%, so a larger batch is not the first lever.

1. **Prefetch fully prepared CPU inputs, not just decoded video.** The current
   pipeline predecodes the next batch, but input and image preparation still
   take roughly 3–14 seconds per update in measured full-data steps. Move
   CPU resize, tokenization, collation, and host-to-device staging for the
   next batch into a bounded pipeline that overlaps the current GPU update.
   Keep only a small number of prepared batches queued to control host and
   GPU memory. Measure preparation wait, presentations/s, mean GPU
   utilization, and readings at or below 5% on the same row window.

2. **Balance work across the eight ranks within each planned update.** All
   ranks currently use the same batch size, but frame count, image processing,
   and token length can make one rank finish much later. Measured waits at
   the gradient boundary reached about 14–19 seconds on some updates.
   Assign examples using a per-example cost estimate that includes video
   frames and model tokens, while preserving the exact global example set
   and loss weighting. Compare per-rank step time and synchronization wait,
   plus aggregate throughput and utilization, against the current planner.

3. **Audit and benchmark Qwen3.5's model kernels in an isolated environment.**
   The process prints a fast-path-unavailable warning, yet its later OOM
   stack entered a vendored `fla` gated-delta-rule kernel. Identify which
   attention and Conv1d kernels actually run, then test compatible faster
   variants on the same examples and batch plan. Verify loss and gradients
   against the current path and compare presentations/s, utilization, and
   peak memory. Faster kernels may improve throughput even if CPU gaps still
   limit the utilization percentage.

Run these comparisons **sequentially on this node** after the current
eight-GPU training run, using the same warmed row window, sampling interval,
and batch plan. Keep the current run's checkpoint and data checksums as the
reference. Do not claim an improvement from utilization alone; report
presentations/s and memory headroom as well. The batched vision-encoding
variant already tested slower and is not among these priorities.
