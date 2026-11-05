# Qwen3.5-2B J1-S: eight-H200 training resource profile

Date: 2026-11-04. This is a live profile of the full seed-42 training run from
[`qwen35_2b_j1s_web_plan.md`](../plans/qwen35_2b_j1s_web_plan.md). It informs **future**
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

| Physical batch | NCCL presentations/s, throughput-only | NCCL peak allocated GPU memory |
| ---: | ---: | ---: |
| 1 | 2.45 | 7.24 GiB |
| 2 | 2.72 | 9.39 GiB |
| 4 | 3.90 | 13.28 GiB |

These warmed-cache throughput-only runs supersede cold-cache comparisons in the
initial phase profile. The phase-profiled Gloo and NCCL rates in the linked
summary include frequent CUDA synchronizations and are not directly comparable
with the throughput-only rates above. These are short measurements, not matched
full-epoch quality runs. The completed 2B WEB result still uses physical batch 1 and Gloo; see
the [judged result](qwen35_2b_j1s_web.md).

## Larger batches on the current node

The user authorized a larger global batch and full source-data exposure. The
eight-GPU batch sweep on `ip-10-3-120-167` reaches physical batch 48 per GPU,
global batch 384, with packed NCCL gradient reduction. The [full comparison](../../outputs/jevany-qwen35-2b-util/reports/scale_summary.md)
includes ordered and length-sorted runs, per-rank timing, and one-second GPU
samples. Length grouping cuts padding from 61.0% to 8.0% at batch 32 on the
same measured row multiset, raising throughput from 6.19 to 8.01 presentations
per second. A separate batch-48 window reaches 14.00 presentations per second
with 9.5% padding, mean sampled GPU utilization 62.0%, and maximum sampled
memory 123,257 MiB per GPU (85.7% of H200 capacity). Peak PyTorch allocated
and reserved memory were 109.68 and 113.05 GiB respectively.

The default allocator ran out of memory on the final batch-48 mixed-context
step despite free reserved memory. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
completed all 20 updates. Full 32-frame batch-48 checks also completed.
These are capacity and throughput results; an epoch on the expanded training
data and its quality evaluation are still separate work.

On a five-update long-context batch-48 phase profile, the slowest rank spent
about 46 seconds decoding and resizing video, 12 seconds preparing images,
24 seconds encoding vision, 15 seconds assembling inputs, 39 seconds in the
decoder forward pass, 74 seconds in backward, and up to 15 seconds around
gradient synchronization. The profile uses CUDA synchronization at each phase,
so these figures diagnose time distribution rather than predict throughput.
Caching up to 1,024 frames across episodes raised throughput on the same
7,680-row window from 14.00 to 14.82 presentations per second and sampled
utilization from 62.0% to 63.4%.

Removing the stride and episode cap yielded 823,686 episodes and 8,343,088
source decisions from 169,939 eligible annotation records, with the same
causal episode recipe. The full-source [summary](../../outputs/jevany-qwen35-2b-full/manifests/source.summary.json)
records its labels and checksums. The ratio to the matched run is 31.6 times.
A physical batch of 52 (global 416) passed a one-update stress check on 416
long 32-frame decisions, reaching 135,337 MiB sampled memory per H200 (94.1%
of capacity). A mixed-context batch-52 run is required before treating that
capacity result as an epoch-safe training setting.

The expanded J1-S [manifest summary](../../outputs/jevany-qwen35-2b-full/manifests/j1s.summary.json)
contains the same 8,343,088 presentations and 1,304,978 Response labels as
the full source. It replaces easy negatives with 316,314 distinct
post-response Silence decisions and 21,966 grounded Hold decisions while
excluding 45,352 ambiguous memory-only Hold contexts. The builder's large-set
filter was changed to compute the removal union once, allowing the full-data
rewrite to complete without changing its selection rule.

## Full-data length-dependent batch plan

The full 8,343,088-row J1-S manifest was grouped by estimated context length
in bounded 8,320-row windows. Its [batch plan](../../outputs/jevany-qwen35-2b-full/manifests/batch_plan.summary.json)
uses synchronized update sizes across the eight ranks and accounts for every
row. Each tier was stress tested on the current node using long examples in
its context range and the same 1,024-frame cache. A Conv1d batch-axis split
into chunks of 128 avoids a PyTorch 32-bit indexing limit for large physical
batches; a small CPU forward/backward comparison matched the unsplit operation
to floating-point tolerance.

| Frame range | Production physical batch/GPU | Stress batch tested | Peak sampled H200 memory |
|---:|---:|---:|---:|
| 1–4 | 256 | 288 (passed, 320 OOM) | 142,305 MiB at 288 (99.0%) |
| 5–8 | 160 | 160 (192 OOM) | 129,901 MiB (90.4%) |
| 9–16 | 96 | 96 | 134,839 MiB (93.8%) |
| 17–24 | 64 | 64 | 129,387 MiB (90.0%) |
| 25–32 | 52 | 52 | 135,337 MiB (94.1%) |

The short tier uses 256 rather than the passing 288 because the latter left
only about 1% memory headroom. The full plan contains 10,134 updates,
including a final partial batch. These are capacity checks; utilization and
throughput depend on video preparation and the mix of context lengths.
Batch-48 batched vision encoding did not help on its measured window:
13.85 presentations/s and 60.4% sampled utilization versus 14.00/s and
62.0% without it.
