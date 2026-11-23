# Fresh Qwen3.5-2B PACE/ESTP-IT supervised controller

Authorized 2026-11-17. Train from pinned official Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, fresh rank-64/alpha-128 decoder
LoRA and pointer head. Base, vision and promoted MoE answer generator stay
frozen. This is a new SFT lineage, not a continuation of RL step 160.

The latest user instruction puts training on the current node
`ip-10-3-70-177`, allocation 38226, and asynchronous evaluation on
`ip-10-3-120-167`, allocation 38177. Use eight GPUs on each; allocate no other
nodes. This reverses the historical RL placement for this SFT run only.

Use 57,600 seconds of accumulated training-update time, with evaluation
checkpoints at hours 2/4/6/8/10/12/14/16. Initialization, preparation, correctness
gates and checkpoint writing are extra. Do not stop for quality diagnostics.
Correctness failures require recovery from committed model/optimizer/RNG state.

Each global batch contains exactly 16 PACE and 16 ESTP-IT decisions. Preserve
the native PACE training labels. For eligible ESTP-IT standing single questions,
keep all response examples, up to one Hold and four Silence examples per
response. A response is labeled at the first fixed-grid sample inside the native
positive-width answer window; the remainder of that same window is Hold.
Outside annotated answer windows is Silence. Reject overlapping supervised
windows and unreachable targets; do not widen windows or shift sampling using
gold annotations. Preserve the existing source-video train/development/test
assignment and protected evaluation exclusions.

Supervision uses causal teacher-forced past actions and answers, the latest
aged-out response memory and at most 32 frames. The current target answer is
never an observation. The loss is action cross-entropy, head/LoRA learning rates
0.0001/0.000002, AdamW weight decay 0.01 and global gradient clip 1.0.

At each milestone evaluate full WEB 500, full EGO 326 and every eligible
explicit-task ESTP-IT development SQ episode. Freeze actual cohort counts and
identities in the prepared manifest. IT test and ESTP-Bench are not evaluated.
Use the existing promoted MoE greedy generator, 96 answer tokens, self-generated
causal history, raw learned argmax and margin zero. GPT-4.1 judging uses the
existing user-authorized gateway and credentials. Native-window IT PAUC-F is
a development proxy; report pooled and task-macro scores separately. WEB/EGO
historical RL train/heldout subdivisions remain labeled as historical exposure.

The full IT development cohort is much larger than the prior 128-case probe.
Preserve every scheduled evaluation if it backlogs; finish the queue after
training. Do not silently replace full cohorts with subsets.

At hours 8 and 16 also sample four trajectories per episode on a fixed,
task-stratified eight-episode development diagnostic, without updates. Report
all-zero and tied-reward groups separately from greedy evaluation. This small
probe does not establish a population-wide success rate. No RL starts
automatically.

Run: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-seed42-16h`.
Entry point: `scripts/run_pace_estp_sft.sh prepare|initialize|train|eval`.
Use NVMe caches and FSx for durable manifests, code snapshots, checkpoints,
predictions, GPT requests/responses, metrics and final reports.

## Launch verification

The run launched on 2026-11-17. Data preparation froze 263,760 native PACE
decisions and 436,629 ESTP-IT decisions from 32,024 episodes / 516 training
videos. The Explicit IT development cohort has 4,894 cases and 1,447,186
decisions, substantially larger than the earlier 128-case probe.

NCCL failed on this node with CUDA 802 before any optimizer step; the repaired
trainer uses the original SFT's exact packed CPU/Gloo gradient average.
All eight GPUs continue to perform forward/backward. The first update,
eight-observation cross-node gate, committed step-28 optimizer/RNG round trip,
GPT access and complete three-episode inference/scoring smoke all passed.

The large full suite uses validated causal controller batching, retaining
singleton greedy MoE generation. A 401-decision comparison changed zero actions
or answers and had maximum probability error 0.00125731, below the existing
0.005 tolerance. Native-harness prompt parity through aged-out memory and pixel
parity of persistent decoding also passed. Each checkpoint has batch/singleton
checks; failures preserve partial artifacts and fall back to the original
evaluator with identical cases and settings. Sampled readiness remains on the
original evaluator.

The 16-hour budget is time-based. The earlier estimate of 16,000 updates from
PACE-only throughput was too optimistic for this mixed run; use live measured
throughput in `training-status.json`. No training or model-quality completion
claim is made by these startup checks.

## Approved replacement on 2026-11-18

The first physical-batch-one run was stopped at the user's request. Preserve its artifacts; do not resume its monitor. The replacement is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h` and starts again from the pinned base model with a fresh pointer head and decoder LoRA.

Each update contains 192 PACE and 192 ESTP-IT decisions across eight GPUs. Compare physical batches 48 and 24, with two accumulated microbatches for the latter, using the same logical global batch 384. CPU decoding/preparation is prefetched and frozen visual features are batched. The approved target learning rates are 0.00035 for the pointer head and 0.000007 for LoRA, with 50 warmup updates. AdamW, weight decay 0.01 and global gradient clipping 1 remain explicit.

Validate batch/padding and gradient normalization against a full-precision singleton reference before timing. Native BF16 attention kernels depend numerically on batch and sequence geometry; compare their error to the BF16 singleton error against that same reference. Do not optimize the reference or profile weights. Run a long-context capacity update and ten ordinary profile updates per candidate, discarding three warm updates. Use the measured slowest-rank steady update time to freeze a two-hour step interval K; fresh production trains 8K updates. This is an estimated sixteen-hour training budget; inference/judging may produce an evaluation backlog.

Save checkpoint zero and perform a three-case initialization correctness smoke only. The user explicitly removed full evaluation of the random action head. Evaluate complete WEB, EGO and frozen ESTP-IT Explicit development splits at K, 2K, through 8K. Keep raw learned argmax, margin zero, native timing windows, causal self-generated history, the frozen promoted MoE answer generator and authorized GPT scoring. At estimated hours eight and sixteen, run the fixed eight-episode, four-trajectory readiness diagnostic. IT task-macro and pooled metrics remain separate; this is IT development PAUC-F proxy evaluation, not ESTP-Bench or official ESTP-F1. No automatic RL continuation is authorized.

Training uses allocation 38226 on ip-10-3-70-177; evaluation uses allocation 38177 on ip-10-3-120-167. Both supervisors and the new fixed-step monitor honor the replacement run's stop marker. Checkpoints are checksummed, with source-count, optimizer and rank-RNG state; scratch and interrupted artifacts remain auditable. Shared FSx holds durable artifacts and local NVMe holds compiler/temporary caches.
