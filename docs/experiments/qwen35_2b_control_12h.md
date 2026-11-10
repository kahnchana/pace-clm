# Qwen3.5-2B control, seed 42

Updated: 2026-11-06. Status: **complete**, including all four 500-video judged
WEB evaluations. Completion: 2026-11-06 14:06:26 UTC. Final checkpoint:
step 3,608, 461,824 presentations, **13.533 actual training hours**, final
PAUC-F **48.70**. See the
[consolidated completed report](qwen35_2b_mixture_12h_results.md) for all A/B
results, uncertainty, runtime issues and the scaling recommendation.

Node: `ip-10-3-120-167`, eight H200 GPUs. Output root:
`outputs/jevany-qwen35-2b-mixture-12h/control-seed42/`.
Launched 2026-11-05 at 22:23 UTC; restarted before any production updates after
adding missing snapshot config files. Detached orchestrator PID `1855040`.
Production training began at 22:26:13 UTC. The first three updates completed
with global batch 128 and finite loss. The recorded historical PID identifies
the completed launch, not a currently running job.

Frozen budget: **3,608 updates / 461,824 presentations**, global batch 128.
Full WEB milestones: **902 / 1,804 / 2,706 / 3,608 updates**. The context-weighted
preflight estimate is 11.977 seconds per update, or 12.004 training hours.
The preflight projection allowed approximately 2–3 additional hours for four
evaluations, plus reload/judge overhead. Measured preflight peak total device
memory was 26.8% of capacity on the busiest GPU. At launch, user allocation
`38177` had almost 30 days remaining.

The user amended the original twelve-training-hour specification to a fixed
optimizer-step estimate. `launch_plan.json` is the authoritative frozen step
count and its four WEB milestones. Elapsed training time is recorded, without
a time-based stop. Four complete 500-video WEB evaluations and judging occur
between segments, outside the estimated training duration.

## Frozen inputs

- Official `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`; fresh seed-42 r64/alpha128/dropout
  0.05 language LoRA and the existing 256-wide pointer head.
- Global batch 128, 16 examples per rank, physical chunks of eight. NCCL
  passed scalar and 63,570,000-element all-reduce tests on all eight ranks.
- AdamW pointer LR 1e-4 and LoRA LR 2e-6, default weight decay 0.01,
  betas (0.9, 0.999), eps 1e-8; gradient clip 1.0; constant learning rate.
- Existing gold histories, labels, memory, 1 FPS, 32 causal frames and
  144 vision tokens per frame. Original row contents and multiplicities remain
  unchanged, with stable estimated-token grouping within 8,320-row windows.

The eligible pool contains 8,173,782 presentations, 4,595,758 unique positions
and 158,103 videos. Exclusions remove 169,306 presentations. The frozen mix is
81.9121% Silence / 15.6593% Response / 2.4285% Hold. Exclusion provenance and
alias checks are saved in `shared/manifests/`; TV/VAD also have a supplemental
zero-overlap proof against the source metadata domain.
The first 4,096 control presentations contain 2,478 Silence / 1,256 Response /
362 Hold (8.84% Hold), confirming meaningful early exposure contrast without
altering the control order.

Eligible source-order SHA256:
`57547c3cf6d8f988456af5728c28a193a9b50dad8c83502d55d05951bb349248`.
Emitted control-order SHA256:
`6ed39b1cf13db9331db9f7b1cb1c74152120fe9365160468462711c33dddb703`.

Initial LoRA SHA256:
`4dbe28e6eeb4c3817852017b576e0f971ce142f3eee43596f046beeb5ccfd54d`.
Initial pointer-head SHA256:
`53acb56d179c2c938d7c8865c97f774cc5458c16a2ea34b6de43aa63ee4d5fed`.
These are byte-identical to the representative arm's initialization artifacts.

## Validation and interpretation

The full test suite passed (16 tests). A real judge connectivity request and
the promoted generator contract check passed. The evaluation-completeness gate
also passed against the historical step-505 full WEB artifacts.
A synthetic 500-video silent fixture also passed through the unchanged official
build/process/score stages and PAUC-F calculation with zero judge requests,
confirming that a silent outcome is reported rather than treated as a job failure.

All eight ranks verified exact restoration of LoRA/head tensors, optimizer,
Torch/CUDA/Python RNG, presentation cursor, step and recorded training elapsed
time. The next resumed update advances optimizer counters and data position.
Independent uninterrupted repetitions have numerical differences in adapter
updates despite matching RNG; subsequent head differences reach about 1.6e-4.
The resumed-update difference is within the independently measured repetition
baseline. This experiment does not promise bitwise reproducible trajectories.

`preflight/report.json` records representative throughput, context-stratified
cost estimates, memory headroom and checkpoint checks. The throughput estimate
weights short/middle/long probe costs by the actual frozen pool distribution.
Actual duration may differ because the consumed prefix and cache reuse differ
from the probe. The fixed step count is never retuned after viewing WEB.

## Operation and artifacts

`scripts/launch_qwen35_control.py` starts a detached four-phase runner with a
private snapshot of the scripts and Python source, so unrelated shared-workspace
edits do not change later phases. `code_snapshot.json` records source hashes.
The trainer verifies the frozen manifest hash and refuses accidental fresh
overwrites. Checkpoint publication uses an atomic latest-pointer replacement.

Inspect `status.json`, `logs/orchestrator.log`, `logs/train_phase1.log` and
`logs/gpu_samples.csv`. The runner releases training allocations, evaluates
each immutable milestone checkpoint, verifies 500 unique prediction/score IDs
and complete valid judge responses, then reloads the full training state.
It fails visibly on runtime or integrity errors. Completed scores and exposure
statistics appear in `phaseN_report.json` and the cumulative `report.json`.
All scheduled evaluations are now complete and independently rechecked.
The completion watcher verified all four checkpoint/result sets and queued
the requested notification into the existing chat; its saved state is
`completion_watch.json` with `completion_queued: true` and `active: false`.

## Completed checkpoint results

| Step | Actual training hours | PAUC-F | Fire precision | PAUC recall | Useful / wasted fires |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 902 | 3.948 | 45.44 | 45.66% | 45.22% | 795 / 946 |
| 1,804 | 7.468 | 49.47 | 61.69% | 41.29% | 615 / 382 |
| 2,706 | 10.637 | 43.20 | 45.40% | 41.21% | 641 / 771 |
| 3,608 | 13.533 | 48.70 | 57.44% | 42.28% | 641 / 475 |

The step-2,706 drop reflects substantially more wasted fires at almost
unchanged recall. Its paired-video 95% interval versus step 1,804 is
−7.79 to −4.79 PAUC-F points. Final versus step 1,804 is inconclusive
(−0.77 points; interval −2.18 to +0.70). The final checkpoint remains the
primary endpoint; no best-WEB-checkpoint selection was substituted.

The control's physical chunks of eight were overly conservative relative to
measured memory headroom. Two chunks accumulate into one batch-128 update,
with one gradient reduction and optimizer step. The unmeasured full-16
alternative might improve throughput but is not a demonstrated speedup.
Production mean sampled utilization ranged from 30.88% to 42.26% across
phases, with 48.84–57.60% of samples at ≤5%. CPU prefetch waits were
negligible; synchronization-boundary timing does not identify the exact
slow-rank cause. The recipe was retained unchanged during the run.

Do not scale this recipe unchanged to the entire manifest based on these
results. The objective/history mismatch and optimization hypotheses require
bounded diagnosis; they are not proven causes, and no new objective or
training continuation has been launched. Unequal budgets and runtimes
preclude a clean causal comparison with B.
