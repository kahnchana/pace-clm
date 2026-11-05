# Qwen3.5-2B short GPU-utilization follow-up on the current H200 node

Date: 2026-11-05. Host: `ip-10-3-70-177`. All GPU probes used only this
host's eight H200 GPUs. No scheduler job or full-data training run was
launched. These are short throughput experiments, not model-quality results.

## Measurement and node constraint

The probes use the same 7,680-row batch-48 length-sorted window, five warmup
updates, and 15 measured updates. Each update presents 384 examples across
eight ranks. One-second `nvidia-smi` samples were filtered to the synchronized
measurement interval recorded in each per-rank JSON report. The current
host's minimal eight-rank NCCL all-reduce fails with CUDA error 802,
"system not yet initialized." The experiments therefore use the existing
packed CPU-buffer Gloo gradient-reduction path. Their absolute rates should
not be compared with the NCCL results from `ip-10-3-120-167`.

| Short run | Presentations/s | Mean sampled GPU util. | Samples at ≤5% | Peak sampled GPU memory |
|---|---:|---:|---:|---:|
| Decode-only prefetch, original rank assignment | 11.15 | 46.90% | 29.89% | 118,924 MiB |
| **CPU preparation prefetch, original rank assignment** | **19.61** | **80.98%** | **5.82%** | **123,444 MiB** |
| CPU preparation prefetch, cost-balanced ranks | 12.96 | 52.13% | 35.67% | 123,904 MiB |
| CPU preparation prefetch, isolated fast kernels | 20.83 | 79.76% | 6.67% | 123,984 MiB |

The [baseline](../../outputs/jevany-qwen35-2b-util/reports/gloo_b48_prefetch_prepare_baseline.json),
[CPU-prefetch](../../outputs/jevany-qwen35-2b-util/reports/gloo_b48_prefetch_prepare_cpu.json),
[rank-balanced](../../outputs/jevany-qwen35-2b-util/reports/gloo_b48_rank_balanced_prepare.json),
and [fast-kernel](../../outputs/jevany-qwen35-2b-util/reports/gloo_b48_prefetch_prepare_fast.json)
JSON reports record every rank's measured interval, step times, and CUDA
memory peak. Matching one-second GPU samples are in the same experiment's
`logs` directory.

## 1. Prefetch CPU preparation

The benchmark-only implementation uses one queued batch per rank. A worker
decodes and resizes video, runs the image processor, assembles and tokenizes
decision inputs, and collates them on CPU while the current update runs on
GPU. The main thread still runs vision encoding and CPU-to-GPU transfer before
the language-model step. The queue is bounded to one batch; the active
full-data trainer was not changed. The
[paired runner](../../scripts/run_qwen35_2b_prefetch_prepare_probe.sh)
compared this path with decode-only prefetch on the same rows.

CPU preparation prefetch increased throughput by 76% and mean sampled GPU
utilization by 34.08 percentage points. It cut near-idle samples from 29.89%
to 5.82% while increasing peak sampled memory by 4,520 MiB. This is the
strongest measured path and exceeds the requested 80% average utilization
on this short window. Pinned host memory and asynchronous device transfers
were not part of this probe.

## 2. Balance examples across ranks

The [cost-balanced manifest](../../outputs/jevany-qwen35-2b-util/manifests/normal_b48_rank_balanced.summary.json)
keeps every global batch's row multiset and every rank's 48 examples. A greedy
cost estimate using frame count and estimated model tokens reduced mean
estimated rank-load spread from 1.047 to 1.002. Actual throughput fell 34%
relative to CPU prefetch with the original assignment. Reassigning rows may
have reduced per-rank frame-cache locality; this was not separately measured.
The [runner](../../scripts/run_qwen35_2b_rank_balance_probe.sh) and manifest
generator are retained for analysis, but this assignment should not be used
for training.

## 3. Optimized Qwen3.5 kernels

The project dependency path already has `flash-linear-attention` 0.3.2. The
matching `causal-conv1d` 1.5.0.post8 wheel was installed only under `/tmp` for
this probe; the production environment was unchanged. The fast-path log had
no PyTorch-fallback warning. On the same 15 measured updates, throughput rose
6.2% over CPU prefetch, while mean sampled GPU utilization decreased 1.22
percentage points. Peak sampled memory rose 540 MiB.

A [four-example parity probe](../../scripts/probe_qwen35_fast_path_parity.py)
loaded the same step-500 adapter and pointer head for both paths with dropout
disabled. The fast path's loss was 0.64945 versus 0.66063 for the fallback
(1.7% relative difference); checked head and LoRA gradient norms differed by
2.4% and 0.9%. These differences remain unexplained. The fast path should
not be adopted for training based on this throughput result alone. The
[fallback](../../outputs/jevany-qwen35-2b-util/reports/kernel_parity_fallback_checkpoint.json)
and [fast-path](../../outputs/jevany-qwen35-2b-util/reports/kernel_parity_fast_checkpoint.json)
parity outputs record logits and gradient slices.

## Practical conclusion

CPU preparation prefetch was subsequently made the default in the
[checkpointed trainer](../../scripts/train_jevany_2b_batched.py), and both
full-data shell runners now select it explicitly. The original decode-only
mode remains selectable for comparison. No full-data training run was
launched as part of this change.

### Separate 20-step training test

The [local test runner](../../scripts/run_qwen35_2b_prefetch_train_20step.sh)
used the first 20 updates of the full-data batch plan, a 0.75 microbatch
ratio, packed CPU-buffer Gloo reduction, and the default CPU preparation
prefetch mode. It stopped after exactly 20 optimizer updates and saved a
complete [step-20 checkpoint](../../outputs/jevany-qwen35-2b-prefetch-20step/train/step-000020)
in a separate output directory. All eight rank summaries record step 20 and
2,104 examples per rank; the 20 planned global batches total 16,832
presentations. The original full-data checkpoint remained at step 505.

| 20-step test measure | Result |
|---|---:|
| Presentations/s through step 20 | 12.32 |
| Mean sampled GPU utilization | 51.09% |
| Samples at ≤5% GPU utilization | 30.94% |
| Peak sampled GPU memory | 96,618 MiB |
| Total CPU prefetch wait | 8.31 s, all on step 1 |

The test completed without OOM or nonfinite loss. Rank-0 phase timings sum
to 103 seconds of GPU preparation, 1,214 seconds of model work, and 35 seconds
at gradient synchronization across the 20 updates. Step 9 alone spent 19.7
seconds at that synchronization boundary. The one-batch CPU queue was ready
for every update after the first. The 51.09% utilization is from the mixed
full-data plan and should not be equated with the fixed batch-48 benchmark's
80.98%. A matched decode-only 20-step run on this host was not performed, so
the short training test establishes functionality and memory headroom for
these updates, not a measured full-plan speedup. Later full-data contexts may
have a different memory peak.

The [training log](../../outputs/jevany-qwen35-2b-prefetch-20step/logs/train.log),
[GPU samples](../../outputs/jevany-qwen35-2b-prefetch-20step/logs/gpu_samples.csv),
and [recipe](../../outputs/jevany-qwen35-2b-prefetch-20step/train/recipe.json)
record this run.
