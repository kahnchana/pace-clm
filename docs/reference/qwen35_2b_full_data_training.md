# Qwen3.5-2B full-data training procedure

Status: provisional as of 2026-11-05. The full 8,343,088-presentation epoch and
judged WEB evaluation are pending. This records the procedure exercised on the
eight-H200 node `ip-10-3-120-167`; it is not a validated default for another
node, model size, data mix, or quality objective.

The measurements and live-run history are in
`docs/experiments/qwen35_2b_full_data_gpu_optimization.md`. That report and
the full-data scripts named below are currently local worktree files; commit
and version them before treating this page as a checkout-ready runbook.
The earlier [resource profile](../experiments/qwen35_2b_j1s_resource_profile.md)
documents the smaller matched run and initial capacity work.

## Data and batch-plan contract

1. Build the causal full source with video stride 1 and no episode cap. Apply
   the J1-S rewrite, retaining all Response decisions and the grounded
   post-response Silence and Hold selections. Keep the source and rewrite
   summaries, checksums, and label counts with the manifest.
2. Group the J1-S rows by estimated context length within bounded 8,320-row
   windows. The current `bucket_jevany_manifest.py` uses 384 rows times 20
   batches per window when invoked with its defaults; the recorded full run
   used an 8,320-row window. Preserve every row and record both input and
   output checksums. Verify that the grouped manifest still has 8,343,088
   presentations.
3. Generate a synchronized eight-rank plan with `plan_jevany_batches.py`.
   Its current tiers use physical batches per GPU of 256, 160, 96, 64, and
   52 for progressively longer contexts, with a 40-example fallback for
   unusually long text. The recorded plan covers all rows in 10,134 updates,
   including its final partial batch. Inspect the plan summary and confirm
   that its total row count equals the grouped manifest's row count.

These capacities were stress checked on the named node. A passing short
stress update is insufficient evidence for a full epoch: one live GPU sample
reached 142,650 of 143,771 MiB (99.2%). Repeat capacity checks and adjust the
tiers before using a different node, runtime, or dataset.

## Launch and resume on the tested node

The local launch scripts fix the repository path, Python environment, output
directory, hostname, and eight-rank topology. They set offline model loading,
packed NCCL gradient reduction, the expandable CUDA allocator, a 1,024-frame
cache, Conv1d batch chunks of 128, four video-decode workers, and next-batch
decode prefetch. Check those paths and the node allocation before reuse.

1. Run `bash scripts/run_qwen35_2b_full_batch_plan_gate.sh` for the first five
   optimizer updates. It writes a complete checkpoint and samples GPU memory
   and utilization. Inspect `outputs/jevany-qwen35-2b-full/logs/train_gate.log`,
   its GPU sample CSV, and `train/latest.json` for an OOM, nonfinite loss,
   incomplete checkpoint, or memory too close to capacity.
2. Continue with `bash scripts/run_qwen35_2b_full_batch_plan_resume.sh` only
   after the gate passes. The trainer checks the manifest and batch-plan
   checksums against the checkpoint recipe and verifies row progress against
   the plan. It saves at step 25, every 500 steps, and at a requested stop.
3. For a bounded resume check, set `STOP_AFTER_STEPS` to an absolute step and
   give `RUN_TAG` a distinct value before invoking the resume script. The tag
   selects separate log and GPU-sample filenames; it does not change the
   checkpoint identity.

The output root is `outputs/jevany-qwen35-2b-full/`. Check `train/latest.json`
and the linked checkpoint directory, then compare the latest logged step and
processed-row count with the plan. Monitor per-GPU memory, low-utilization
samples, rank waiting, loss, and checkpoint writes. The observed live rate and
finish date are estimates, since later windows can have a different context
mix. Resume from the latest complete checkpoint after an interrupted run;
do not infer completion from a launcher PID alone.

## Completion gate

Before interpreting model quality, verify that training reached the final
planned update, covered all 8,343,088 presentations, and saved a reloadable
final checkpoint. Then run the frozen WEB evaluation and judging protocol,
recording its exact checkpoint, generator, manifest, and scorer settings.
The measured throughput gains and GPU utilization do not establish a quality
gain. Update this page with any settings that survive the full epoch and
quality evaluation; keep measured outcomes in the experiment report.
