# GPU optimizations

The Qwen3.5-2B J1-S full-data run on eight H200 GPUs used these changes to
improve training throughput:

- Group decisions by estimated context length to reduce padding, then use a
  synchronized batch plan with larger batches for shorter contexts.
- Pack gradients for NCCL reduction. Use the expandable CUDA allocator and
  split large Conv1d batches to avoid an indexing limit.
- Cache frames across episodes, decode video in parallel, and prefetch the
  next batch while the GPUs work on the current one.

See the [full-data optimization report](../experiments/qwen35_2b_full_data_gpu_optimization.md)
for the implementation, capacity checks, throughput measurements, and live-run
status. The earlier [resource profile](../experiments/qwen35_2b_j1s_resource_profile.md)
records the initial bottleneck and batch-size experiments. The full epoch and
judged quality evaluation are still pending; the measured settings are specific
to the tested node.
