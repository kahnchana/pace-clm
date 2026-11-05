# Qwen3.5-2B full-data GPU optimization on the current eight-H200 node

Date: 2026-11-05. All runs in this report used **only**
`ip-10-3-120-167`, with one local rank on each of its eight H200 GPUs.
This work follows the [original J1-S resource profile](qwen35_2b_j1s_resource_profile.md).

## Outcome and measurement scope

The matched J1-S experiment used physical batch 1 per GPU, accumulation 4,
global batch 32, and 263,760 training presentations. Its sampled mean GPU
utilization was 24.3% and maximum memory use was 12,014 of 143,771 MiB per
GPU, measured on its original node. The experiments here change the training
batch and data exposure; their throughput measurements do **not** establish
matched model quality.

On the current node, a physical batch of 52 per GPU (global batch 416) passed
both a full 32-frame stress update and a 20-update mixed-context run. The
stress update reached 135,337 MiB, or 94.1% of H200 memory. Shorter contexts
allow larger batches, so a synchronized length-dependent plan covers all
8,343,088 full-data presentations in 10,134 updates.

The best sustained-compute result is a same-window batch-48 benchmark:
grouping similar lengths, packing gradients for NCCL, caching frames, and
decoding the next batch with four CPU threads per rank raised throughput to
18.03 presentations/s and mean sampled GPU utilization to 76.7%. The
baseline length-sorted batch-48 path gave 14.00 presentations/s and 62.0%
utilization. The checkpointed full-data training status is reported below.

## Full-source data

The original exposure used video stride 10 and an episode cap of 26,080.
Running the same causal episode builder at stride 1 without the cap produced
the [full-source manifest](../../outputs/jevany-qwen35-2b-full/manifests/source.summary.json):

| Measure | Full source |
|---|---:|
| Eligible annotation records | 169,939 |
| Causal episodes | 823,686 |
| Training decisions | 8,343,088 |
| Response / Silence / Hold | 1,304,978 / 6,814,643 / 223,467 |
| Decisions versus matched run | 31.6× |

The [full J1-S rewrite](../../outputs/jevany-qwen35-2b-full/manifests/j1s.summary.json)
preserves the 8,343,088 decisions and all 1,304,978 Response labels. It adds
316,314 distinct post-response Silence decisions and 21,966 grounded Hold
decisions, while removing easy negatives and 45,352 ambiguous memory-only
Hold contexts. The source and J1-S checksums are recorded in their summaries.
The full-data builder needed one scaling fix: it now computes the union of
removal keys once before filtering millions of rows. This preserves the
selection rule and avoids repeated construction of a large set.

The [length-grouped manifest](../../outputs/jevany-qwen35-2b-full/manifests/train_length_sorted.summary.json)
sorts within bounded 8,320-row windows. Its input checksum equals the J1-S
output checksum, and its row count remains 8,343,088. Grouping within a
window reduces padding while retaining all examples. The [batch plan](../../outputs/jevany-qwen35-2b-full/manifests/batch_plan.summary.json)
uses the same per-rank batch size at every update; its global batch varies
with context length. It accounts for every row, including a final partial
update.

## GPU capacity

These are **one complete forward/backward/optimizer update** on long examples
in each frame range. They use packed NCCL reduction and the expandable CUDA
allocator. The 1–24-frame tests use a 1,024-frame cache; the batch-52 full
32-frame stress check did not, while its mixed-context run did. The production short tier is deliberately
below its passing stress size. A narrow Conv1d batch-axis split into chunks
of 128 avoids a PyTorch 32-bit indexing limit at large physical batches; a
CPU comparison matched unsplit forward and backward to floating-point
tolerance.

| Frames | Planned physical batch/GPU | Stress batch | Highest sampled GPU memory | Result |
|---:|---:|---:|---:|---|
| 1–4 | 256 | 288 | 142,305 MiB (99.0%) | Passed; 320 OOM after Conv1d split |
| 5–8 | 160 | 160 | 129,901 MiB (90.4%) | Passed; 192 OOM |
| 9–16 | 96 | 96 | 134,839 MiB (93.8%) | Passed |
| 17–24 | 64 | 64 | 129,387 MiB (90.0%) | Passed |
| 25–32 | 52 | 52 | 135,337 MiB (94.1%) | Passed |

Batch 320 first hit Conv1d's 32-bit index limit. After Conv1d batch splitting,
it reached a genuine GPU OOM at about 127 GiB PyTorch allocation while
requesting another 9.3 GiB. Batch 288 passed but left only about 1% sampled
headroom, so the full plan uses 256 for that tier. The plan's largest
estimated contexts for batches 256, 160, 96, and 64 are below the respective
stress-set maxima. The batch-52 plan maximum is 0.4% above its stress proxy;
that stress check left about 6% sampled headroom.

Batch 52 also completed a 20-update mixed-context run: 10.40 presentations/s,
44.8% mean GPU utilization, 8.4% token padding, and 133,083 MiB maximum
sampled memory. This window has a different context distribution from the
batch-48 comparison below, so those throughput figures should not be used as
a direct batch-size ranking.

## Sustained throughput and utilization

The [scale report](../../outputs/jevany-qwen35-2b-util/reports/scale_summary.md)
links the per-rank JSON results and one-second `nvidia-smi` samples. The
batch-48 variants below use the same 7,680-row ordered source window, preserve
the first five warmup updates, and measure the same remaining 15 updates.
Length sorting uses the same measured row multiset. Five warmup updates are
excluded from the rates and GPU samples.

| Eight-GPU configuration | Presentations/s | Mean GPU util. | Samples at ≤5% util. | Token padding |
|---|---:|---:|---:|---:|
| Physical 1, global 32, ordered | 2.45 | 47.8% | 40.8% | 0.0% |
| Physical 32, global 256, ordered | 6.19 | 53.0% | 31.9% | 61.0% |
| Physical 32, global 256, length sorted | 8.01 | 42.1% | 35.6% | 8.0% |
| Physical 48, global 384, length sorted | 14.00 | 62.0% | 19.5% | 9.5% |
| Batch 48 plus cross-episode frame cache | 14.82 | 63.4% | 17.6% | 9.5% |
| Batch 48 plus parallel video decode | **16.53** | **72.0%** | **10.4%** | 9.5% |
| Batch 48 plus frame cache and decode prefetch | **18.03** | **76.7%** | **5.1%** | 9.5% |
| Batch 48 plus batched vision encoding | 13.85 | 60.4% | 30.5% | 9.5% |

The batch-32 length-sorted and ordered runs use the same measured rows;
sorting cuts padding from 61.0% to 8.0% and raises throughput by 29%.
Sampled utilization can fall even when throughput rises because the same
work finishes in less time. The batch-48 cache and decode comparisons use
the same rows and settings except for the named option. Decode prefetch
starts the following batch's CPU video work during current GPU work. Batched vision
encoding did not help in this window.

A five-update long-context phase profile at batch 48 attributed about 46 s
to decode/resize, 12 s to image preparation, 24 s to vision encoding, 15 s
to input assembly, 39 s to forward, 74 s to backward, and up to 15 s around
gradient synchronization on the slowest rank. The profile synchronizes CUDA
between phases, so it diagnoses the distribution of time rather than giving
an unperturbed throughput rate. The data path and rank waiting remain the
main reasons sustained GPU utilization is below 100%.

## Training and reproducibility

The eight-rank [batched trainer](../../scripts/train_jevany_2b_batched.py)
collates multimodal decisions, averages each update over its actual global
example count, uses packed NCCL gradients, and checkpoints model adapter,
pointer head, optimizer, RNG, and row progress. Its first five updates run
through the [checkpoint gate](../../scripts/run_qwen35_2b_full_batch_plan_gate.sh);
the [resume runner](../../scripts/run_qwen35_2b_full_batch_plan_resume.sh)
continues the same plan. Both runners check the current-node hostname before
launching any GPU ranks. The full manifest and plan checksums are checked on
resume.

The joint cache and next-batch decode prefetch path was selected from the
same-window benchmark: 18.03 presentations/s, 76.7% mean sampled utilization,
and 5.1% of samples at ≤5% utilization. The first five full-data updates
passed and saved a complete step-5 checkpoint. They processed 6,144 examples
in 289 seconds (21.26 presentations/s), with no OOM or nonfinite loss.
Sampled mean GPU utilization was 57.6%, and maximum memory was 125,735 MiB
(87.5%). The first update includes pipeline fill, and the full-data context
mix differs from the batch-48 benchmark.

The run resumed successfully from that checkpoint for steps 6–10, then saved
the current [step-10 checkpoint](../../outputs/jevany-qwen35-2b-full/train/latest.json).
These five updates covered 2,528 more examples, including 31- and 32-frame
contexts. They averaged 59.8% sampled GPU utilization and peaked at
135,410 MiB (94.2%). Rank 0 waited 2.3 seconds for prefetch on the first
resumed update and near zero thereafter. Its input/vision preparation took
2.9–14.4 seconds per update, model work 26–70 seconds, and the gradient-sync
boundary up to 19.1 seconds when other ranks finished later. On the final
32-frame update, preparation was only 2.9 seconds and model work 45.6
seconds. Rank waiting and model work remain reasons real full-data
utilization is below the 76.7% short-window benchmark. The process printed a
fast-path-unavailable warning, but the later OOM stack entered the vendored
`fla` gated-delta-rule kernel under `outputs/jevany/deps032`; therefore it is
incorrect to describe the entire linear-attention path as a PyTorch fallback.
The exact use of optional `flash-linear-attention` and `causal-conv1d` code
needs an isolated path audit before claiming a faster-path result. Their upstream projects are
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d).

The full epoch resumed from step 10 at **2026-11-05 00:48:10 UTC** on
`ip-10-3-120-167`. The live runner is
[`run_qwen35_2b_full_batch_plan_resume.sh`](../../scripts/run_qwen35_2b_full_batch_plan_resume.sh),
with progress in [its log](../../outputs/jevany-qwen35-2b-full/logs/train_resume.log)
for the initial run.
It reached and saved a complete [step-25 checkpoint](../../outputs/jevany-qwen35-2b-full/train/latest.json)
at **2026-11-05 01:01 UTC**. Steps 11–25 processed 13,792 examples in
801 seconds (17.2 presentations/s), with 57.5% mean sampled GPU utilization,
21.5% of samples at ≤5% utilization, and 134,550 MiB (93.6%) peak sampled
memory. The step-25 update itself used global batch 768; its rank-0 timing
was 10.2 seconds of preparation, 38.4 seconds of model work, and 0.6 seconds
at gradient synchronization. These real-training measurements are lower
than the best controlled batch-48 benchmark and do not establish sustained
near-100% compute utilization.

At **2026-11-05 01:40 UTC**, the latest logged update was step 50: 43,808 of
8,343,088 presentations (0.525%). Steps 11–50 averaged 16.41
presentations/s, while GPU samples since step 25 averaged 55.1% utilization.
Peak sampled memory rose to 142,650 of 143,771 MiB (99.2%) on one GPU;
there was no OOM at this check. Holding the measured rate throughout the
remaining source data would finish around **2026-11-11 UTC**, approximately
5.9 days from this check. This is a projection, since later windows have a
different context mix and the near-full memory leaves little room for a
larger peak. The initial runner saved at step 500 and was configured to save
every 500 steps. This run must remain on the current node and should be checked
by its log and checkpoint path before any quality evaluation.

The full 8.34-million-presentation epoch and judged WEB evaluation are
required before making any quality claim.

### One-minute live utilization check

From **2026-11-05 02:19:49–02:20:49 UTC**, `nvidia-smi` produced 57 readings
per GPU while the eight-rank training run remained active on this node. Mean
sampled compute utilization across all eight GPUs was **52.3%**; per-GPU
means ranged from **49.1% to 59.1%**. **32.7%** of readings were at ≤5%
utilization. The highest memory reading was **142,694 of 143,771 MiB
(99.3%)** on GPU 0. The latest logged progress at this check was step 100
(87,808 presentations); the latest saved checkpoint was still step 25.

### OOM and safe resume

The initial full run logged step 925 at **2026-11-05 13:27 UTC**, then rank 0
failed at **13:34 UTC** with `Triton Error [CUDA]: out of memory` while
loading a linear-attention kernel. Its recent peak was 143,166 of 143,771
MiB on GPU 0, leaving less than 1% headroom. The last durable checkpoint
was step 500, so unsaved work after it had to be replayed. The earlier
November 10–11 finish projection no longer applies.

The trainer now divides each planned per-rank batch into forward/backward
chunks at most 75% of that batch and accumulates gradients before its single
optimizer update. The global batch plan, row order, loss normalization, and
checkpoint identity remain the same. The resume runner now saves every 100
steps to limit future rollback.

A guarded resume from step 500 completed steps 501–505, including a 32-frame
planned batch, and saved a complete [step-505 checkpoint](../../outputs/jevany-qwen35-2b-full/train/latest.json).
Peak sampled memory was 105,866 MiB; mean sampled utilization was 51.1% over
that short gate. The full run resumed from step 505 at **2026-11-05 14:50:39
UTC** on `ip-10-3-120-167`. Its new progress log is
[`train_resume_safe.log`](../../outputs/jevany-qwen35-2b-full/logs/train_resume_safe.log).
At this checkpoint, 426,464 of 8,343,088 presentations (5.11%) are durable.
The next scheduled checkpoint is step 600. Re-estimate the finish time after
a longer representative window of the lower-memory path.

### Stop and WEB evaluation

At the user's request, the resumed training workers were stopped on
**2026-11-05 15:34 UTC**. The latest durable checkpoint remained step 505;
no subsequent training updates were saved. The eight local GPUs then ran the
complete 500-video WEB evaluation from that checkpoint. The judged PAUC-F was
**47.12**, versus **54.67** for the earlier fully trained 2B J1-S controller
under the same WEB protocol. Training remains stopped. See the separate
[step-505 WEB result](qwen35_2b_full_step505_web.md) for the full metrics and
artifact links.
