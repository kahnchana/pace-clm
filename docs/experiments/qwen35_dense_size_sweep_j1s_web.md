# Qwen3.5 dense-controller size sweep: 0.8B and 4B J1-S on WEB

Date: 2026-11-04. Repeat the completed [2B J1-S experiment](../plans/qwen35_2b_j1s_web_plan.md)
on the same eight-H200 node, once each with official post-trained
`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B` controller checkpoints. Both use the
same frozen promoted `Qwen/Qwen3.5-35B-A3B` answer generator and the final
one-epoch checkpoint. The official controller revisions are
`2fc06364715b967f1860aea9cf38778875588b17` (0.8B) and
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (4B).

The common training source is `outputs/jevany-j1s/manifests/train.jsonl`, SHA-256
`89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222`:
263,760 presentations, including the original J1-S class balance, order,
negative replacement, gold response history, and last-response memory. Use one
seed (42), one epoch, 8 GPUs, microbatch 1, accumulation 4, 8,243 updates,
AdamW defaults, pointer LR `1e-4`, LoRA LR `2e-6`, clipping 1.0, and the
existing action cross-entropy. Keep the 256-wide pointer, matching language
projection LoRA (r64, alpha128, dropout 0.05), frozen vision and base weights,
1 FPS, 32 causal frames, 384px and 144 tokens per frame, direct prefix, and
aged-out last-response v2 memory. Create a fresh adapter and head at each size.

Run 0.8B training first, then start 4B training as soon as the 0.8B epoch
finishes. Overlap 0.8B WEB evaluation with 4B training on the same eight GPUs,
as requested on 2026-11-04. Evaluate 4B after its own final checkpoint. Each
has a separate output
root: `outputs/jevany-qwen35-0p8b-j1s/` and
`outputs/jevany-qwen35-4b-j1s/`. Before full training, check model-specific
loading, all three action gradients, a 32-frame backward pass, checkpoint
resume, short-example overfit, and controller-to-MoE handoffs on training
videos. Record source/dependency/model hashes and matched modules. The node's
NCCL reduction returned CUDA error 802 in the 2B run, so use the validated
eight-rank CPU Gloo gradient average unless a new NCCL preflight succeeds.

Evaluate each final checkpoint on all 500 WEB IDs with response margin 0,
96-token greedy promoted-MoE answers, actual controller history, and separate
model-specific vision features. Merge all eight shards and complete GPT-4.1
judging with `pauc-f-v1`, omega 0.5. Do not select a checkpoint or margin from
WEB. Write each `reports/web_result.md` and a combined
`outputs/jevany-qwen35-size-sweep-j1s/reports/web_comparison.md`. Compare
0.8B, 2B, and 4B on judged PAUC-F, precision, recall, useful and wasted
fires, latency, memory, and training duration, including paired WEB bootstrap
against stored 2B predictions. Historical MoE J1-S started from a PACE
adapter, so its comparison has an initialization difference.

## Execution status (2026-11-04 15:20 UTC)

Both sizes passed eight-GPU 32-frame backward and checkpoint-resume checks,
five-update short-example overfit, and eight-video handoffs to the common
promoted generator. Each handoff produced 16 nonempty answers. Peak allocated
GPU memory in handoff inference was 67.82 GiB for 0.8B and 75.06 GiB for 4B.
The per-size preflight summaries are in `preflight/handoff_summary.json` under
their respective output roots.

The full pipeline started on the current eight-GPU allocation
(Slurm 38226) at 15:20 UTC. A completion-marker watcher launches 4B training
when 0.8B training finishes, while the first pipeline proceeds to 0.8B WEB
evaluation. Per-size file locks prevent duplicate runs. Same-node Slurm job
38381 was queued as a checkpoint-resuming fallback and canceled after both
results completed. Each size records
30-second GPU utilization and memory samples in `logs/gpu_samples.csv` for
batch-size planning after the run. The matched run retains microbatch 1 and
accumulation 4.

## 0.8B result and overlap profile (2026-11-05 00:16 UTC)

The 0.8B model completed 8,243 updates and all 500 WEB videos. Its judged
PAUC-F is **50.1244** (precision 74.80%, PAUC recall 37.69; 623 fires, 466
useful, 157 wasted). This is 4.54 points below the matched 2B controller and
6.39 below historical J1-S. The full result is in
`outputs/jevany-qwen35-0p8b-j1s/reports/web_result.md`.

The 4B training run launched at 2026-11-04 21:57:52 UTC while 0.8B WEB
evaluation was active. In its 30-second samples between 21:58 and 22:23:30,
mean GPU compute utilization across the eight GPUs was 34.8%, mean memory used
was 66,619 MiB, and maximum sampled memory was 95,790 MiB. In the later
4B-only window through 00:15 UTC, mean utilization was 27.4% and maximum
sampled memory was 19,186 MiB. These are utilization samples from all GPUs,
including generator loading and uneven per-video work; they do not isolate
training throughput. The concurrent peak remained below the H200 capacity of
143,771 MiB per GPU.

## Final result (2026-11-05)

Both new controllers completed one epoch and judged all 500 WEB videos with
the same frozen promoted generator. The combined report is
`outputs/jevany-qwen35-size-sweep-j1s/reports/web_comparison.md`.

| Controller | PAUC | PAUC-F | Fire precision | Fires | Useful | Wasted |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 37.69 | 50.12 | 74.80% | 623 | 466 | 157 |
| 2B | 44.54 | 54.67 | 70.74% | 991 | 701 | 290 |
| 4B | 46.73 | 53.78 | 63.32% | 1,216 | 770 | 446 |

The 4B controller has 2.19 points higher raw PAUC than 2B, but 0.89 points
lower PAUC-F because its higher fire rate reduces precision. A paired
1,000-replicate WEB bootstrap for 4B minus 2B PAUC-F gives a 95% interval of
[-2.49, +0.67] points. The 0.8B minus 2B interval is [-6.03, -2.89]. Thus the
observed 4B PAUC-F difference is not resolved by this 500-video sample.
