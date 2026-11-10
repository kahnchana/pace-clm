# Qwen3.5-2B representative-mixture experiment B

Updated: 2026-11-06. Status: **complete**, including all four 500-video judged
WEB evaluations; completion recorded at 08:52:58 UTC. Final checkpoint:
step 1,776, 227,328 presentations, **8.914 actual training hours**, final
PAUC-F **47.61**. See the
[consolidated completed report](qwen35_2b_mixture_12h_results.md) for both
arms, uncertainty, deviations and interpretation limits.

Launch preparation: 2026-11-05, node `ip-10-3-70-177` only, eight H200 GPUs.
The user authorized experiment B and requested a measured step estimate instead
of implementing stopping by training hours. The runner freezes four step targets
from the preflight, approximately 3/6/9/12 training hours. Actual training time is
recorded independently, excluding checkpoint and evaluation time.

Specification: [12-hour mixture experiments](../plans/qwen35_2b_12h_mixture_experiments.md).
Output root: `outputs/jevany-qwen35-2b-mixture-12h/representative-seed42/`.
Read `status.json` for the current phase, `schedule.json` for frozen step targets,
and `reports/results.md` / `reports/results.json` for completed evaluations.
Completion is established by `status.json` reporting `complete`, after all four
500-video WEB evaluations are fully judged. Preparation or a running process is
not evidence of completed training.

The common eligible pool contains 8,173,782 presentations after 169,306
benchmark/validation-overlap exclusions, covering 158,103 videos. Its natural
label weighting is 81.9121% Silence / 15.6593% Response / 2.4285% Hold.
The immutable mixture order indexes original byte ranges, preserving row
contents, causal histories, labels and original multiplicities. Every full
update has 104–105 Silence, 19–21 Response and 3–4 Hold presentations. All
4,096-row window and cumulative window-prefix audits pass; full plans are exact
permutations of the same eligible row set. Rank ownership rotates each update
to avoid assigning Hold examples systematically to the same GPUs. The terminal 86 rows remain in the
plan, but cannot form a complete batch of 128 and will not be trained on.

The NCCL collective preflight failed with CUDA error 802, so this run uses the
repository's Gloo CPU gradient-reduction fallback. This backend difference must
be retained when comparing with a control on another node.

The runner checks maximum-context memory headroom, representative throughput,
full-state resume against an uninterrupted update, and a one-video WEB pipeline
smoke check before starting the experiment. It then trains from a shared saved
fresh seed-42 initialization, evaluates immutable checkpoints using the promoted
MoE generator and GPT-4.1 `pauc-f-v1`, and resumes the same optimizer/RNG/data
position. Failures are recorded in `status.json`; they do not silently skip an
evaluation or select a different recipe.

Entry points: `scripts/plan_qwen35_representative.py` and
`scripts/run_qwen35_representative.py`. Existing source manifests and historical
training artifacts are preserved. This runner does not launch experiment A.

Production training started at 2026-11-05 22:14 UTC. Frozen checkpoints: 444,
888, 1,332 and 1,776 updates (227,328 presentations total). The measured
preflight estimate is 24.377 seconds/update, or 12.026 estimated training hours.
All required preflight checks passed; the main experiment and all scheduled
evaluations have completed. Actual training took 8.914 hours rather than the
12.026-hour estimate. The control took 13.533 hours and used more updates and
presentations; these are not equal-time or equal-exposure arms.

The single-video WEB smoke test produced no fires, so it validated rollout and
scoring without making judge requests. A separate optional historical-request
connectivity check was rejected by automatic approval review as an additional
data export and was skipped. The scheduled evaluations use the authorized
experiment outputs and the existing GPT-4.1 scoring pipeline.

## Completed checkpoint results

| Step | Presentations | Actual training hours | PAUC-F | Fire precision | PAUC recall | Useful / wasted fires |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 444 | 56,832 | 2.460 | 40.49 | 83.82% | 26.69% | 57 / 11 |
| 888 | 113,664 | 4.768 | 47.74 | 74.80% | 35.06% | 383 / 129 |
| 1,332 | 170,496 | 6.907 | 51.73 | 77.81% | 38.75% | 512 / 146 |
| 1,776 | 227,328 | 8.914 | 47.61 | 50.44% | 45.08% | 738 / 725 |

The final segment gained useful fires and recall, but increased wasted fires
from 146 to 725 and lowered precision. Final minus step 1,332 is −4.12
PAUC-F points, with a paired-video 95% interval of −5.82 to −2.34 from
1,000 replicates. This uncertainty does not include training seeds or judge
repeatability. Natural class exposure did not establish stable closed-loop
quality; the evidence does not isolate its causal effect from runtime and
budget differences. The final checkpoint is the primary endpoint; the
51.73 intermediate maximum is diagnostic.
