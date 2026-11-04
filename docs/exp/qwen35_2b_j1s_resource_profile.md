# Qwen3.5-2B J1-S: eight-H200 training resource profile

Date: 2026-11-04. This is a live profile of the full seed-42 training run from
[`qwen35_2b_j1s_web_plan.md`](../qwen35_2b_j1s_web_plan.md). It informs **future**
runs; the current matched experiment retains one example per GPU, accumulation
4, and effective global batch 32.

## Measurement

- Node: `ip-10-3-70-177`, eight NVIDIA H200 GPUs, 143,771 MiB each. Each GPU
  had one training rank and no other compute process in the process listing.
- Window: 2026-11-04 **00:46:54–00:50:06 UTC**, 90 samples per GPU, mean cadence
  2.15 seconds. The raw [CSV](../../outputs/jevany-qwen35-2b-j1s/reports/resource_profile_2026-11-04.csv)
  and [JSON summary](../../outputs/jevany-qwen35-2b-j1s/reports/resource_profile_2026-11-04.json)
  contain all readings.
- The nearby [training log](../../outputs/jevany-qwen35-2b-j1s/logs/train.log)
  records updates 325–375 in 192.8 seconds, or **3.86 s/update**. This includes
  the existing data decode, forward/backward, CPU Gloo gradient reduction, and
  optimizer work. The first 25 updates took 1,431 seconds due to startup and
  kernel warmup and should not be used as steady throughput.

| GPU | Mean compute util. | 95th percentile | Readings at ≤5% | Highest memory used |
|---:|---:|---:|---:|---:|
| 0 | 23.8% | 90.1% | 53.3% | 11,992 MiB |
| 1 | 23.1% | 81.1% | 45.6% | 11,860 MiB |
| 2 | 23.0% | 79.5% | 43.3% | 11,936 MiB |
| 3 | 24.0% | 81.0% | 47.8% | 12,014 MiB |
| 4 | 26.6% | 100.0% | 45.6% | 11,922 MiB |
| 5 | 25.1% | 85.0% | 45.6% | 11,820 MiB |
| 6 | 23.6% | 100.0% | 47.8% | 11,954 MiB |
| 7 | 24.9% | 92.8% | 45.6% | 11,886 MiB |
| **All** | **24.3%** | **88.0%** | **46.8%** | **12,014 MiB** |

Across the 90 simultaneous samples, all eight GPUs were at ≤5% utilization
12.2% of the time. At least one was at ≥80% while another was at ≤5% in 32.2%
of samples. No sample had all eight at ≥50%. Mean memory-controller utilization
was 7.6%; mean board power was 210 W against a 700 W limit. The highest sampled
memory use was **8.36%** of H200 capacity.

## What to try next

The GPUs have substantial observed memory headroom and bursty compute use. A
**physical microbatch of 2 per GPU with accumulation 2** is the first batch-size
experiment to benchmark. It keeps the effective global batch at 32 and should
halve the number of model calls per optimizer update if the trainer is adapted
to collate two examples. Then benchmark microbatch 4 with accumulation 1 if
microbatch 2 improves presentations per second and full 32-frame memory remains
safe. Preserve per-example loss weight, the final partial update's scaling,
causal inputs, and row order when comparing with this run. Changing only
`--accumulation` would increase the global batch without improving model-call
efficiency.

The current trainer processes one variable-length row at a time. Batched
forward passes require padding and per-example decision/option positions, so
memory headroom alone does **not** prove that microbatch 2 or 4 will be faster.
Benchmark them on representative short and full 32-frame windows, recording
updates/s, presentations/s, padding tokens, and peak allocated/reserved GPU
memory. Keep the same global batch and compare judged quality in a separate run.

Also measure time spent in video decode, vision encoding, tokenizer/input
assembly, forward/backward, gradient synchronization, and optimizer step. This
node's NCCL reduction failed with CUDA error 802, so the current run packs about
**242.5 MiB of float32 gradients per rank per update** for CPU Gloo all-reduce.
That transfer and rank-to-rank context-length imbalance are plausible causes
of GPU idle time; this profile does not isolate their shares. For a future run,
first validate NCCL on its node and compare synchronization time. Then decide
whether batching or input preparation gives the larger throughput gain.

The 2.15-second sampling cadence can miss brief memory peaks. The 12,014 MiB
reading is an observed maximum, not a proven worst-case allocation. These
utilization figures cover a three-minute portion of one epoch, not the entire
training or the two-model WEB evaluation.

## Follow-up microbatch benchmark (2026-11-04)

An eight-GPU benchmark on a separate node (`ip-10-3-120-167`) tested physical
batches of 1, 2, and 4 with accumulation 4, 2, and 1 respectively. All kept
global batch 32, used the same ordered 640-row normal-context window, and
excluded five warmup updates. The packed NCCL gradient reduction passed on
this node. [Per-rank results and phase timings](../../outputs/jevany-qwen35-2b-util/reports/summary.md)
are retained under the ignored output tree.

| Physical batch | NCCL presentations/s | Gloo presentations/s | NCCL peak allocated GPU memory |
| ---: | ---: | ---: | ---: |
| 1 | 0.89 | 0.87 | 7.24 GiB |
| 2 | 0.99 | 0.97 | 9.39 GiB |
| 4 | 3.86 | 1.50 | 13.28 GiB |

These are short throughput measurements, not matched full-epoch quality runs.
The completed 2B WEB result still uses physical batch 1 and Gloo; see the
[judged result](qwen35_2b_j1s_web.md). Larger-batch capacity and throughput
experiments are continuing separately.
