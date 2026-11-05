# Qwen3.5-2B full-data experiment: consolidated findings

Date: 2026-11-05. **Status: training stopped at the user's request; latest
saved checkpoint is step 505.** The new training, capacity tests, and WEB
evaluations in this report used only `ip-10-3-120-167` and its eight H200
GPUs. The earlier matched 2B run is a historical comparator measured on its
original node. No completed full-data epoch or final-epoch quality result
exists.

## What we learned

1. Larger, context-dependent batches filled GPU memory and improved a
   controlled eight-GPU benchmark to **18.03 examples/s and 76.7% mean sampled
   GPU utilization**. Real full-data training averaged roughly **52–60%**
   utilization in measured windows. Pushing memory above 99% eventually
   caused an out-of-memory failure.
2. WEB quality changed substantially across early checkpoints on the **same
   500 videos and evaluation protocol**: PAUC-F was **34.89 at step 10**,
   **32.27 at step 25**, and **47.12 at step 505**. The earlier final matched
   2B checkpoint scored **54.67**. The policy's Hold, Response, and Silence
   choices shifted sharply; this is an observed checkpoint effect, not a
   different WEB sample.
3. The first 5.11% of the planned full-data training sequence has a very
   different class mix from the complete manifest. That exposure skew may
   contribute to early-checkpoint behavior. The available runs do not
   establish a causal explanation for the WEB score changes.

## Data, model, and training plan

The controller is `Qwen/Qwen3.5-2B` with a trainable LoRA adapter and pointer
head. It sees up to **32 causal frames at 1 FPS**, each resized to
**384 × 384** and represented by **144 image tokens**. The earlier matched
J1-S run used physical batch 1 per GPU, accumulation 4, global batch 32,
and 263,760 presentations. That count is **3.16%**, not 10%, of the new
full-data manifest: the earlier recipe used video stride 10 and an episode
cap.

Removing those limits yielded 169,939 eligible annotation records, 823,686
causal episodes, and **8,343,088 decisions**. The full J1-S rewrite preserved
all 1,304,978 Response decisions and all 8,343,088 rows while changing
Silence/Hold selection. Its [summary](../../outputs/jevany-qwen35-2b-full/manifests/j1s.summary.json)
records the source and output checksums. Bounded-window length grouping kept
all rows, and the [synchronized batch plan](../../outputs/jevany-qwen35-2b-full/manifests/batch_plan.summary.json)
would cover them in 10,134 optimizer updates:

| Frames | Physical batch per GPU | Global batch | Planned updates |
|---:|---:|---:|---:|
| 1–4 | 256 | 2,048 | 996 |
| 5–8 | 160 | 1,280 | 1,419 |
| 9–16 | 96 | 768 | 3,165 |
| 17–24 | 64 | 512 | 1,688 |
| 25–32 | 52 | 416 | 2,865 |
| Final partial | 18 | 144 | 1 |

The complete training manifest is 81.96% Silence, 15.64% Response, and 2.40%
Hold. The first **426,464 rows**, which are all that the step-505 checkpoint
has seen, are **63.61% Silence, 25.50% Response, and 10.90% Hold**. Response
is 1.63 times and Hold 4.54 times as common in this early prefix as in the
full manifest. The earlier matched manifest's overall class mix was close
to the new full-manifest mix (81.78% Silence, 15.63% Response, 2.59% Hold).
These prefix counts come from streaming the first 426,464 rows of the
[ordered training manifest](../../outputs/jevany-qwen35-2b-full/manifests/train_length_sorted.jsonl).
Grouping within bounded windows preserves rows, but it does not make an
early checkpoint representative of the whole source.

## GPU capacity and throughput

The original matched profile on its original node measured **24.3%** mean
sampled GPU utilization and at most **12,014 of 143,771 MiB (8.4%)** used
per GPU. These are historical measurements, not a same-node throughput
comparison. On the requested current node, complete stress updates passed
at batch 52 per GPU for 32-frame examples, using **135,337 MiB (94.1%)**.
Shorter contexts permitted larger batches; batch 288 at 1–4 frames passed
at 99.0% sampled memory, while batch 320 failed. Batch 192 at 5–8 frames
also failed, so the production plan used 160. The
[capacity and scale report](qwen35_2b_full_data_gpu_optimization.md) has the
full per-tier tests.

On the same 7,680-row, batch-48 benchmark window (five warmup plus 15
measured updates), the main measured changes were:

| Eight-GPU path | Examples/s | Mean GPU util. | Samples at ≤5% util. |
|---|---:|---:|---:|
| Length grouped, packed NCCL | 14.00 | 62.0% | 19.5% |
| Plus cross-episode frame cache | 14.82 | 63.4% | 17.6% |
| Plus parallel video decode | 16.53 | 72.0% | 10.4% |
| Frame cache plus next-batch decode prefetch | **18.03** | **76.7%** | **5.1%** |
| Batched vision encoding variant | 13.85 | 60.4% | 30.5% |

Length grouping at batch 32 cut token padding from 61% to 8% and raised
throughput from 6.19 to 8.01 examples/s on the same rows. The best batch-48
path was selected for full training. Benchmarks on different context windows
are not direct batch-size comparisons.

In real full-data training, the first five planned updates processed 6,144
examples in 289 seconds (21.26/s), with 57.6% sampled utilization and 87.5%
peak sampled memory. Steps 6–10 averaged 59.8% utilization and peaked at
94.2% memory. Steps 11–25 processed 13,792 examples in 801 seconds
(17.2/s), with 57.5% utilization and 93.6% peak memory. A separate fresh
one-minute sample at 02:19:49–02:20:49 UTC averaged **52.3%** across eight
GPUs; **32.7%** of its readings were at or below 5%, while peak memory
reached **99.3%**. Video predecode wait was nearly zero after pipeline fill,
but measured preparation and rank synchronization still left GPU gaps.
High allocated memory does not mean kernels run continuously.

The process printed a fast-path-unavailable warning, but the later OOM stack
entered a vendored `fla` gated-delta-rule kernel. We therefore have not
established which model kernels could be accelerated further. The separate
[GPU-utilization TODO](qwen35_2b_gpu_util_todo.md) records the three
highest-priority follow-ups; those changes were not validated as improvements
to this stopped full-data run.

## OOM, recovery, and saved state

The first uninterrupted full-data attempt logged step 925 and then failed at
13:34 UTC on rank 0 with `Triton Error [CUDA]: out of memory` while loading
a kernel. A recent sampled peak was **143,166 of 143,771 MiB** on that GPU.
The last checkpoint was step 500, so unsaved updates after step 500 were
lost. The original five-to-six-day finish projection ceased to apply.

The trainer was then changed to split each planned per-rank batch into
forward/backward chunks of at most 75% and accumulate gradients before one
optimizer update. This preserves the planned global examples and loss
normalization while lowering peak activation memory. A guarded resume from
step 500 completed steps 501–505, including a 32-frame planned batch, with
**105,866 MiB** peak sampled memory and no OOM or nonfinite loss. The full
resume started from step 505; the user then asked to stop it for WEB
evaluation. No update after step 505 was saved. The revised runner was set
to save every 100 steps, but the stopped run never reached its next
scheduled checkpoint at step 600.

Current full-data checkpoints, all with adapter, pointer head, optimizer,
and eight rank RNG files, are [steps 5, 10, 25, 500, and
505](../../outputs/jevany-qwen35-2b-full/train). The earlier matched run
retains [step 25, every 100 steps through 8,200, and final step
8,243](../../outputs/jevany-qwen35-2b-j1s/full-seed42).

## WEB evaluation and the checkpoint swings

The complete WEB protocol uses 500 identical videos, response margin 0,
the same promoted answer generator, up to 96 generated tokens, 32 causal
frames, and GPT-4.1 `pauc-f-v1` judging. The step-10, step-25, step-505,
and earlier-final protocol manifests differ only in the controller adapter
and pointer-head hashes. Every run produced 500 unique predictions, 8,803
controller decisions, and a complete judged score. Steps 10 and 25 were
evaluated in parallel on disjoint groups of four GPUs on the requested node;
step 505 used all eight GPUs earlier.

| Controller | Training examples | PAUC-F | Fire precision | PAUC recall | Response/fire | Hold | Silence | Useful / wasted fires |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-data step 10 | 8,672 | **34.89** | 45.18% | 28.42 | 301 | 5,639 | 2,863 | 136 / 165 |
| Full-data step 25 | 22,464 | **32.27** | 31.91% | 32.64 | 1,006 | 3,365 | 4,432 | 321 / 685 |
| Full-data step 505 | 426,464 | **47.12** | 59.56% | 38.97 | 863 | 2,712 | 5,228 | 514 / 349 |
| Earlier final 2B J1-S | 263,760 | **54.67** | 70.74% | 44.54 | 991 | 264 | 7,548 | 701 / 290 |

The score change is **not monotonic**: step 25 is 2.62 PAUC-F points below
step 10 despite 705 more fires. It gained some recall while fire precision
fell by 13.27 points and wasted fires rose from 165 to 685. Step 505 is
14.85 points above step 25, yet 7.55 below the earlier final 2B checkpoint.
Step 10 predominantly chose Hold; step 25 moved sharply toward Response;
step 505 chose Silence more often but still used Hold far more than the
earlier final model.

A 1,000-replicate **paired bootstrap over the same 500 WEB IDs** gives these
95% intervals for PAUC-F differences (left model minus right model):

| Comparison | Observed difference | Paired 95% interval |
|---|---:|---:|
| Step 25 − step 10 | -2.62 | [-4.40, -0.87] |
| Step 505 − step 25 | +14.85 | [+13.02, +16.78] |
| Step 505 − step 10 | +12.22 | [+10.04, +14.39] |
| Step 505 − earlier final 2B | -7.55 | [-9.20, -5.95] |

The [bootstrap output](../../outputs/jevany-qwen35-2b-full/reports/web_checkpoint_paired_bootstrap.json)
recomputes the official scores exactly before resampling. The intervals
show that the observed checkpoint differences persist under resampling
these WEB videos. They **do not** measure training-seed variation, repeated
judge variation, or what a finished full-data epoch would score. There is
one training seed and one judged evaluation per checkpoint.

The early training prefix's Response/Hold enrichment is a plausible
contributor to the unusual early policies, and the much larger variable
global batches and far fewer optimizer updates than the earlier matched run
could also matter. None has been isolated by a controlled quality ablation.
The observed actions are consistent with a checkpoint-dependent
action-calibration issue. They do not support a claim that more data or
larger batches inherently reduce quality.

## Source reports and artifacts

- [Detailed current-node GPU optimization and failure chronology](qwen35_2b_full_data_gpu_optimization.md)
- [WEB checkpoint comparison and per-step score artifacts](qwen35_2b_full_checkpoint_web_comparison.md)
- [Earlier fully trained 2B J1-S WEB result](qwen35_2b_j1s_web.md)
- [Original resource profile](qwen35_2b_j1s_resource_profile.md)
- [Full-data manifests and batch plan](../../outputs/jevany-qwen35-2b-full/manifests)

Training remains stopped, and all eight GPUs were released after the WEB
evaluations. There is no valid end-of-epoch ETA for the stopped run.
