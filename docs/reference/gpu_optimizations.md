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
records the initial bottleneck and batch-size experiments. That full-data run
was stopped before completing an epoch; its
[checkpoint quality results](../experiments/qwen35_2b_full_data_findings.md)
are recorded separately. The measured settings are specific to the tested node.

The later [completed mixture experiment](../experiments/qwen35_2b_mixture_12h_results.md)
retained these mechanisms but fixed the effective global batch at 128 rather
than using the earlier large adaptive batches. Its control used two physical
chunks of eight per GPU and measured 30.88–42.26% mean sampled utilization
across production phases, with substantial unused memory. These results do
not reproduce the earlier large-batch benchmark's utilization. Increasing a
physical chunk while retaining batch 128 is a distinct throughput test;
filling memory by increasing effective batch would change the experiment's
optimization contract.
