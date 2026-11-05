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
38381 is queued as a
checkpoint-resuming fallback if that allocation ends. Each size records
30-second GPU utilization and memory samples in `logs/gpu_samples.csv` for
batch-size planning after the run. The matched run retains microbatch 1 and
accumulation 4.

At 21:19 UTC, the 0.8B training log had reached step 7,350 of 8,243; its
latest saved checkpoint was step 7,300. The 4B full run had not started.
Neither size had a WEB result yet. This is a snapshot of the ongoing run, not
an outcome or a checkpoint selection.
