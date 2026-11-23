# Controller evaluation policy

For the current controller experiments, the user requires decisions to come
directly from the learned action scores. Do not add hand-chosen action-logit
offsets, response margins, temperature calibration, cooldowns, or heuristic
response suppression. Learn desired decision behavior through training.

Use response margin 0 for CLI compatibility; the evaluator must take the raw
learned argmax without modifying scores. Nonzero margins are rejected by
`pace_clm.evaluation_policy`. Prior margin-adjusted experiments are retired for
model selection and scaling decisions; retain their logs as audit records.

Run training for the current RL experiment series only on `ip-10-3-120-167`
(allocation 38177), using its eight GPUs. On 2026-11-11 the user additionally
authorized the assigned eight-GPU node `ip-10-3-70-177` (allocation 38226) for
asynchronous standalone evaluation. Keep learner correctness checks on the
training node; do not train on the evaluation node or allocate other nodes.
On 2026-11-12 the user requested the mixed-run final evaluation on the original
training node, allocation 38177, after training releases its eight GPUs. Keep
earlier evaluation results from allocation 38226; route only the final evaluation
to the training node and wait for training to finish before acquiring its GPUs.
GPT judging uses the user-authorized gateway. The historical full-WEB quality
floor is 54.67 PAUC-F; mixed WEB/EGO training results must distinguish seen
examples from examples held out of the new RL run.

On 2026-11-13 the user authorized the step-32 to step-128 continuation with
ESTP-IT and explicitly required all 128 optimizer steps regardless of evaluation
scores. Quality/coverage/silence diagnostics must not stop this continuation.
Correctness checks remain mandatory; recover technical failures from committed
checkpoints. Preserve ESTP-Bench for evaluation and keep ESTP-IT source-video
training/development/test splits separate. Use the same authorized training and
evaluation allocations above; final evaluation follows GPU release on 38177.

The user reiterated on 2026-11-09 that improvements must come from training.
Do not add inference-time score mixing, token/action logit changes, response
thresholds, cooldowns, or other runtime behavioral heuristics. Supervised replay
and regularization may affect learned weights during training only.

On 2026-11-15 the user authorized a further 32 updates from the committed
step-128 checkpoint through step 160: two original, two WEB, two native EGO
and two ESTP-IT episodes per update, four trajectories per episode, and source
objective weights 0.25 each. Keep the existing RLOO recipe and constant-32 loss
normalization. Native EGO reuses 64 training episodes because all 145 eligible
training episodes were used by step 128; the user explicitly accepted this
reuse. Use fresh episodes from the other three sources and preserve heldouts.
Training remains on allocation 38177 and this new experiment's evaluations at
128/144/160 run on allocation 38226, per the user's explicit two-node request.
Use local NVMe for temporary/compiler caches and shared FSx for durable artifacts.
Recover the prior run's unfinished endpoint evaluation on its original training
node after the continuation releases its GPUs; retain its prior jobs and shards.

On 2026-11-16 the user limited remaining evaluation work to finishing the
currently running `full-step128` endpoint suite in
`mixed-web-ego-estp-episode-balanced-v3-to128`. Complete that same job's inference,
GPT judging and scoring, then run no further evaluations. In particular, do not
enqueue a step-160 ESTP-Bench evaluation or any new checkpoint/benchmark sweep.
Existing completed evaluations remain preserved. Generating completion reports
from existing results is allowed; no new inference or judging jobs follow the
current endpoint suite without a later explicit user instruction.

Later on 2026-11-16 the user explicitly authorized re-running GPT scoring as
needed to calculate ESTP explicit PAUC-F using the stream-gemma protocol. This
permits CPU/GPT-only post-hoc scoring of the already saved step-32 and step-128
ESTP-Bench predictions with unchanged estp-pauc-f-v1; it does not authorize new
model inference, a step-160 ESTP run, or unrelated evaluation sweeps. Preserve
the separate official ESTP-F1 results and their native annotations.

Later on 2026-11-16 the user explicitly authorized TV and VAD evaluation of
the committed RL step-160 checkpoint, using all eight GPUs on the current node
`ip-10-3-70-177` (allocation 38226) only. This authorizes TV/VAD inference,
GPT judging and PAUC-F scoring; it does not authorize other benchmark sweeps.
Keep raw learned argmax decisions with response margin zero, NVMe caches and
durable FSx artifacts.

Later in the same session, the user explicitly authorized step-160 ESTP-Bench
evaluation on this node's eight GPUs (ip-10-3-120-167, allocation 38177), limited
to the explicit proactive split and PAUC-F scoring. This supersedes the earlier
stop instruction only for that checkpoint/split; use unchanged estp-pauc-f-v1
and the authorized GPT gateway, without unrelated evaluations.

Later in the same session, the user explicitly requested a full WEB evaluation
with the committed step-160 controller and `gpt-5.6-sol` as answer generator,
using the same endpoint environment variables and credentials as GPT scoring.
Run the controller on the authorized evaluation node `ip-10-3-70-177`, allocation
38226, using its eight GPUs and raw learned argmax with response margin zero.
Preserve self-generated causal answer history, separate artifacts from the MoE
generator baseline, and report RL training and heldout WEB cohorts separately.

The user then explicitly authorized EGO, TV and VAD evaluation with that same
step-160 controller plus `gpt-5.6-sol` generator, all three suites in parallel,
with three controller processes per GPU on the eight GPUs of `ip-10-3-70-177`
(allocation 38226). This includes GPT judging and CPU timing-only PAUC-F from
saved streams. Preserve the WEB settings and EGO RL training/heldout distinction.

The user subsequently corrected TV input requirements and explicitly requested
an immediate TV rerun with subtitles on this same evaluation node. Follow the
stream-gemma Qwen MoE `tv_subtitles_current/qwen-w1-long-neg` reference: measured
source FPS, 1-FPS samples starting at frame zero, native subtitle user turns
binned causally and present in the current frame only. Apply subtitle evidence
to the step-160 controller and Sol answer generator. This additional eight-worker
TV run may coexist with the already authorized EGO run; preserve earlier
visual-only artifacts and retain raw learned argmax with margin zero.

The user then explicitly approved completing StreamingBench PO and PO-I with
the committed step-160 2B controller and promoted frozen Qwen MoE generator on
the current evaluation node (38226 / ip-10-3-70-177), including smoke validation,
all 250 PO questions, all 50 five-request PO-I episodes and the independent,
append and replace-scoped policies, deterministic scoring and final reporting.
Use eight one-GPU controller/MoE workers, the reference 128-token answer budget,
raw argmax/margin zero, existing request/timing protocols and frozen manifest.
Preserve official and corrected PO timing separately; PO-I primary population
is the 185 differing-output transitions. No GPT judging is required.

The user subsequently authorized WEB, EGO, TV with subtitles, and VAD with the
committed step-160 controller plus `gemini-3.8-flash`, all four suites in parallel
across the eight free GPUs of the current node `ip-10-3-120-167`, allocation
38177. This explicitly changes the node for these Flash evaluations only.
Use one worker per suite per GPU (32 total), raw learned argmax with margin zero,
self-generated causal history, and the same native subtitled-TV protocol as Sol.
This includes access/concurrency checks, GPT judging and unchanged PAUC-F scoring.
Keep existing Sol jobs/artifacts, NVMe caches, durable FSx outputs, and separate
WEB/EGO RL training and heldout cohorts. Monitor through all four completions.

The user subsequently authorized the missing step-160 controller plus promoted
MoE generator TV evaluation with native subtitles, on all eight empty GPUs of
`ip-10-3-70-177` (allocation 38226). Use the same frame-zero, measured-FPS,
current-frame-only subtitle protocol as Sol, raw learned argmax with margin zero,
NVMe caches, and separate durable artifacts; include GPT judging and PAUC-F.

The user then explicitly requested retrying the remaining Gemini Flash WEB
videos on allocation 38177. If an entry still does not work, skip that entry,
continue the other videos, and give the skipped entry a perfect score. Preserve
provider block audits and incomplete episode traces; report user-requested
perfect-score imputations separately from measured model/GPT-judged scores.
Do not alter images, prompts, controller decisions, or inference settings to
work around the provider's content block.

On 2026-11-17 the user authorized a fresh Qwen3.5-2B supervised controller run,
initialized from the pinned official model with fresh LoRA and pointer head,
using equal PACE/ESTP-IT decision presentations for approximately 16 training
hours. Evaluate full WEB, full EGO and the frozen ESTP-IT Explicit development
split at two-hour training milestones, with the existing frozen promoted MoE
generator and authorized GPT scoring gateway. The user's latest instruction
is to train on this session's node, ip-10-3-70-177 / allocation 38226, and evaluate
on the other existing node, ip-10-3-120-167 / allocation 38177. This placement is
an explicit exception for this new SFT run only. Preserve source-video splits,
native windows, raw argmax/margin zero, NVMe caches and durable FSx artifacts.
The approved plan also includes four-trajectory, no-update ESTP-IT development
readiness probes at training hours 8 and 16. Do not start RL automatically.

On 2026-11-18 the user stopped that small-batch SFT run and requested a
high-level plan before restarting with the efficient prefetched batched setup
and appropriately scaled learning rates. Its monitor and Slurm steps 38226.56
and 38177.66 were stopped; both allocations were retained and all sixteen GPUs
verified free. Preserve the aborted run and committed step-489 checkpoint at
`outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-seed42-16h`. Do not restart
that run's automatic monitor. The replacement has not been launched; present
the requested plan before undertaking the restart.

Later on 2026-11-18 the user approved launching and completing the revised
prefetched SFT run from fresh Qwen3.5-2B initialization, with global batch 384
and equal PACE/ESTP-IT decisions. Validate physical batch 48 versus accumulated
24, use the measured steady update time to freeze a two-hour step interval K,
and train for 8K updates (approximately 16 hours). Approved LR targets are
head 0.00035 / LoRA 0.000007 with 50 warm-up updates. Keep the same assigned
nodes: train on 38226 / ip-10-3-70-177, evaluate on 38177 / ip-10-3-120-167.
Save checkpoint zero and run only a small initialization smoke check; the user
explicitly removed full evaluation of the randomly initialized pointer head.
Full WEB/EGO/ESTP-IT Explicit development evaluations occur at K through 8K;
the no-update readiness diagnostics remain at the estimated hours 8 and 16.
Profile weights are temporary and must be discarded before production starts.
Preserve the stopped small-batch run. The revised run is
`outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h`.

Later on 2026-11-18 the user requested continuing the completed mixed SFT run
until one prepared training-data epoch, on this same training node, then fixing
the evaluation slowdown by comparing earlier evaluations. Interpret this as
completing the prepared ESTP-IT pool once in total while preserving the 50:50
mix: resume step 1144 through cumulative step 2275 (1131 additional updates),
not an additional full epoch and not switching to the 8.34M-row PACE source.
Keep physical batch 48/global 384, optimizer/RNG/sampler state and completed
warmup. The continuation is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1`.
Train on 38226 / ip-10-3-70-177 only. Preserve the parent's existing evaluations
on 38177 and their partial predictions. Defer evaluation changes until training
finishes; then diagnose and validate the repair against earlier evaluation
protocols. No inference-time decision heuristics or automatic RL are authorized.

Later in the same session the user corrected the evaluation population:
stop ESTP-IT evaluation, complete WEB for the current SFT checkpoint first,
then evaluate ESTP-Bench Explicit (851 cases, unchanged estp-pauc-f-v1), and
report video/episode counts plus inference and GPT-scoring wall times for
WEB, EGO and ESTP-Bench Explicit. The user also explicitly requested GPT
scoring of the already completed EGO predictions. Use SFT checkpoint 143
for this corrected current evaluation, preserving comparability with EGO.
The obsolete combined evaluator's Slurm step 38177.76 and restart monitor
were stopped; its existing predictions and queued job audit files remain,
with a stop marker to prevent resumption. The corrected workflow is
`outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h/evaluation/corrected-step143`.
Run WEB then ESTP-Bench Explicit on allocation 38177 / ip-10-3-120-167;
CPU/GPT scoring of saved EGO may run concurrently without new inference.
The separate SFT epoch continuation on allocation 38226 remains running.

The user then explicitly requested evaluating the latest checkpoint after the
current corrected checkpoint-143 evaluation. Queue exactly one additional
WEB/EGO/ESTP-Bench Explicit evaluation on allocation 38177, waiting for the
current three suites, GPT scoring and GPU release. At dispatch, choose the
newest fully committed checkpoint in the ongoing SFT epoch continuation and
pin it once for all suites/retries. Do not change checkpoints midway through
an evaluation or resume the cancelled ESTP-IT queue. Preserve the generator,
raw argmax/margin zero, native benchmark protocols, judging and timing reports.
The follow-up queue is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h/evaluation/latest-after-step143`.

The user's latest instruction stops all current/queued evaluations to first
reduce ESTP-Bench Explicit to about 200 episodes and investigate speed, with
a total per-checkpoint evaluation budget below three hours. Corrected step-143
Slurm step 38177.81, its restart monitor, and the latest-checkpoint dispatch
watcher were stopped. Their stop markers and all completed/partial artifacts
remain. Do not restart either old queue. SFT on 38226 continues unchanged.
Technical no-update/no-GPT performance profiling is authorized on 38177.
The proposed fixed subset is 25 episodes per native explicit task, seed 42:
200 episodes over 59 video IDs. It is a diagnostic subset, not the full 851-case
headline. Keep full WEB 500 and EGO 326. Use the same 200 IDs for comparisons.
Artifacts: `outputs/jevany-qwen35-2b-pace-estp-sft/eval-speed-design-v1`.
A one-GPU A/B/A pilot on checkpoint 143 passed 957 decisions with identical
raw probabilities, actions and all 12 generated answers. Warm elapsed time
fell from 246.30 to 190.62 seconds; active GPU utilization rose from 58.8% to
65.9%, resident memory remained 74.54 GiB. Direct GPU generator loading took
166.34 seconds on one GPU; eight-GPU startup scaling is not yet measured.
The ready runner keeps models loaded across one combined, balanced eight-GPU
job, uses guarded cached controller inputs and unchanged singleton raw argmax,
and scores WEB/EGO plus the 200-case ESTP subset with the original protocols.
It requires the passing pilot and source/input hashes, and self-cancels only
its own Slurm step at 179 wall minutes; launch with a three-hour step limit.
A budget timeout preserves partial results and is not a completed evaluation.
No full production evaluation with this new setup has been launched: the
user requested stopping and investigating first. All production evals remain
paused; the separate SFT epoch continuation remains running.

The user subsequently explicitly authorized the newer validated faster
implementation on the latest saved checkpoint, removed any explicit three-hour
limit, and asked to complete all inference/judging and measure actual time.
A new eight-GPU evaluation is running on 38177 with checkpoint 1494 pinned
(the newest committed checkpoint at dispatch; training had reached 1500).
Artifacts: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-001494`.
Use full WEB 500, full EGO 326 and the fixed seed-42 ESTP-Bench Explicit 200
(25 per task, 59 video IDs). It uses the newer direct-GPU loader, guarded cached
controller inputs, one persistent model load per GPU, and balanced combined
inference. Keep original singleton raw argmax/margin zero, histories, native
windows, generator and GPT scoring. All wall-time cutoffs/self-cancel timers
were removed, and the Slurm step was launched without --time. A planning
estimate is not a stop condition. Complete all three suites and judging,
record actual timings and report.md. Technical recovery reuses the same
pinned checkpoint and persistent start time; do not reactivate old cancelled
queues. SFT continuation on 38226 remains active and unchanged.

The faster checkpoint-1494 evaluation completed all WEB 500, EGO 326 and
ESTP-Bench Explicit 200 inference and GPT scoring at 20:20:47 UTC on 2026-11-18.
Actual total time was 6068.242 seconds (101.137 minutes), with no hard cutoff
and zero restarts. Scores: WEB 53.659303; EGO 32.297094; fixed ESTP-Bench
Explicit 200-case task macro 19.118450. All outputs and actual timing report
are in the fast-step-001494 directory above. Evaluation GPUs were verified
released; SFT epoch continuation remains running (1700/2275 at that check).
The earlier generic post-training next-action marker may still describe eval
repair as pending; this repair and checkpoint-1494 evaluation are now complete.
Do not restart cancelled queues or launch extra checkpoint sweeps/RL based on
that old placeholder. Future explicitly requested evaluations should preserve
the fixed 200-case selection and use the validated faster implementation.


On 2026-11-18 the user explicitly requested evaluating the latest SFT checkpoint again on the other node with the efficient setup and the same three sets. Pinned the newest fully committed checkpoint, step 1928, after verifying head/adapter checksums and unchanged validated evaluator hashes. Launched one eight-GPU evaluation on allocation 38177 / ip-10-3-120-167: full WEB 500, full EGO 326, and the same fixed seed-42 ESTP-Bench Explicit 200 episodes (59 videos). Preserve raw argmax/margin zero, frozen promoted MoE generator, native GPT scoring, run-to-completion without a wall-time cutoff and bounded technical recovery at the same checkpoint. Outputs: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-001928`. SFT continuation on 38226 remains unchanged; no additional sweep or RL is authorized by this request.


On 2026-11-19 the user authorized evaluating the latest committed SFT checkpoint on the evaluation node and continuing SFT for another epoch on the training node. Step 2275 is pinned for the separate faster evaluation on 38177 / ip-10-3-120-167: full WEB 500, EGO 326 and the unchanged fixed ESTP-Bench Explicit 200, including native GPT scoring and actual timings, no hard cutoff. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/evaluation/fast-step-002275`.

A separate continuation resumes committed step 2275 through step 4550 (2275 additional updates, one additional prepared ESTP-IT row pass while keeping equal PACE presentations), on 38226 / ip-10-3-70-177. Run: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2`. Restore learned adapter/head, AdamW, eight RNG states and deterministic sampler cursor; retain physical batch 48/global 384, head LR 0.00035, LoRA LR 0.000007 and completed warmup. Data, native labels, sampled negatives, source-video splits, loss and sampling recipe remain unchanged; task balancing/hard negatives discussed earlier are not enabled. Save every 900 accumulated training seconds and at endpoint. Both jobs use NVMe caches and durable FSx artifacts, raw learned argmax/margin zero; independent bounded recovery monitors preserve pinned checkpoints. No automatic later evaluation sweep or RL. Earlier frozen scripts/runs remain intact.


On 2026-11-19 the user explicitly authorized continuing the current SFT through five total epochs: after epoch two finishes at committed step 4550, resume that endpoint for three more prepared ESTP-IT epochs (6825 additional updates) through cumulative step 11375. The user reiterated using the same eight GPUs on this training node, ip-10-3-70-177 / allocation 38226 only. The queued continuation is `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5`. Its persistent monitor waits for epoch-two training-complete, launcher exit and GPU release, verifies every committed endpoint file, then dispatches on allocation 38226 with bounded recovery. Preserve physical48/global384, 50:50 PACE/ESTP-IT decisions, head LR0.00035 / LoRA LR0.000007, AdamW, completed warmup, rank RNG states, learned weights and deterministic sampler cursor. No changes to native labels, negatives, task frequencies, loss or source-video splits. Save every900 accumulated training seconds and final endpoint with durable FSx artifacts and NVMe caches. This later authorization supersedes the prior epoch-two wrapper's generic no-further-training next-action text only for this explicitly requested continuation; do not modify the frozen running epoch-two sources. No new evaluations or RL are authorized by this request. About59 additional training hours after epoch two at recent31sec/update; retained allocation expires2026-12-06, sufficient for the planned work.


On 2026-11-20 the user explicitly requested the epoch-two endpoint evaluation. Launched pinned step4550 on allocation38177 / ip-10-3-120-167, using full WEB500, EGO326 and the unchanged fixed ESTP-Bench Explicit200 with the validated faster evaluator, frozen MoE, raw argmax/margin0, GPT judging and no hard cutoff. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2/evaluation/fast-step-004550`. SFT through five epochs continues separately on38226. Evaluations are explicitly requested rather than periodic; training checkpoints are saved about every900 accumulated training seconds.

The user subsequently authorized exactly one follow-up evaluation after step4550 finishes all inference and scoring. At dispatch select the newest fully committed checkpoint available in the active epoch-five continuation, skip it if already completely evaluated, and otherwise pin it for the same three-set faster suite and all technical retries on38177. Wait for the step4550 job to exit before acquiring evaluation GPUs. Do not turn this into a recurring sweep. Preserve all results and unchanged policies. Queue: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch2/evaluation/latest-after-step4550`.


On 2026-11-20 the user amended the one-shot latest-after-step4550 follow-up: if the newest committed checkpoint is already evaluated, keep checking every600seconds for a new committed checkpoint instead of ending with a skip. Once a new checkpoint is found, pin and launch exactly one three-set evaluation, then stop dispatch polling. Preserve the active step4550 evaluation and five-epoch training. The waiting dispatcher was restarted with this policy; no GPU jobs were interrupted.


On 2026-11-20 the user explicitly required preserving the epoch-one SFT checkpoint at `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch1/train/step-002275`. All files listed in commit.json were verified by SHA256; learned head/adapter, optimizer/training state and all eight rank RNG states are intact. RETAIN.json marks the checkpoint for retention. Do not delete, prune or overwrite this checkpoint without explicit user authorization. Preserve its evaluation artifacts as well. This retention request does not change the active five-epoch training.


On 2026-11-20 the user explicitly requested evaluating the current latest checkpoint. Verified checkpoint and unchanged evaluator hashes, then pinned SFT step6249, the latest committed checkpoint at dispatch, for full WEB500, full EGO326 and the same fixed ESTP-Bench Explicit200. Launched on all eight GPUs of evaluation allocation38177 / ip-10-3-120-167 using the validated faster setup, frozen promoted MoE, raw argmax/margin0, native GPT scoring, no hard cutoff and bounded technical recovery. Output: `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5/evaluation/fast-step-006249`. Training through step11375 on38226 continues unchanged; protected epoch-one step2275 remains retained. No additional follow-up or periodic sweep is queued by this request.


On 2026-11-20 the user requested preparing a continuous latest-checkpoint evaluation script, explicitly requiring confirmation before launch. Created `scripts/watch_pace_sft_latest_eval.py`; IT IS NOT LAUNCHED. Default invocation is preview-only; --run is required for dispatch. After user confirmation it should serially evaluate the newest fully committed checkpoint from the active epoch-five continuation on allocation38177, full WEB500/EGO326/fixed ESTP-Bench Explicit200, using the unchanged validated fast runner, raw argmax/margin0, frozen MoE and GPT scoring. It pins each selection through completion/recovery, skips intermediate checkpoints rather than building a backlog, skips completed matching checkpoint/protocol results, polls every600seconds when caught up, waits for existing evaluations and GPU release, persists a bounded four-attempt budget and stops with needs-attention on technical exhaustion. It evaluates the final training endpoint and exits when caught up after training completion. A STOP file at the training run's evaluation/continuous-latest/STOP stops future dispatches while allowing an active suite to finish; restarting without STOP resumes a persisted active selection. State and shared file locks prevent concurrent copies. No running training/evaluation jobs were changed. Six temporary-artifact scheduler tests passed; preview-only validation selects the current newest commit without starting jobs. Await explicit user confirmation before starting this watcher or new inference/judging.


On 2026-11-20 the user explicitly confirmed launching the prepared continuous evaluation watcher. Started `scripts/watch_pace_sft_latest_eval.py --run`; state/logs under `outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-to-estp-epoch5/evaluation/continuous-latest`. This supersedes the previous preparation-only/no-launch instruction and authorizes serial latest-checkpoint inference and GPT scoring until the final training checkpoint has been evaluated. Use allocation38177 / ip-10-3-120-167 eight GPUs only for evaluation, unchanged full WEB500/EGO326/fixed Explicit200 faster protocol, raw argmax/margin0, frozen MoE and native scoring, no wall-time cutoff. Skip intermediate checkpoints while busy, poll every600seconds when caught up, preserve bounded technical recovery and all artifacts. Training on38226 continues unchanged; protected step2275 remains retained.
