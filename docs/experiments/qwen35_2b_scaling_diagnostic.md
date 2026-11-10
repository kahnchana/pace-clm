# Qwen3.5-2B checkpoint and calibration diagnostic

Status: historical diagnostic. The nonzero-margin sweep was retired by the
user's raw-score inference requirement on 2026-11-08. Its preparation and
orchestration entry points now exit before creating files, launching GPU work
or judging. The design and old artifacts below remain an audit record, not a
supported experiment to rerun. Shared analysis helpers and the development
evaluator remain available to current zero-margin callers.

Question: Does the representative-mixture run's late quality decline reflect a
worse decision ranking, a shifted operating point, or a gap between gold-history
classification and behavior with generated history?

The user authorized this diagnosis on node `ip-10-3-70-177` on 2026-11-06.
No new training experiment is launched by this work.

Compare immutable representative-mixture checkpoints 1,332 and 1,776. Freeze
response margins -2, -1, 0, +1 and +2 before inspecting development results.
Every margin gets a separate closed-loop rollout with the original promoted
MoE generator, greedy 96-token answers, 32 causal frames, original memory
contract and raw action-logit recording. Do not treat rescored cached logits as
counterfactual closed-loop rollouts.

Development: 32 videos, 2,383 decisions, 149 events, deterministically stratified
by video length from the existing reserved selection cohort. Confirmation:
32 separate videos, 2,361 decisions, 129 events from the reserved final cohort.
Both cohort manifests are hashed in `plan.json`; source manifest hashes match
the files used to exclude held-out videos from training. Margin choices must
use development scores only and be frozen before confirmation evaluation.
No WEB data selects margins or checkpoint settings.

Matched-input classification diagnostic: 2,016 native gold-history decisions,
including 1,658 Silence, 291 Response and 67 Hold examples, reproduced from
these development annotations. Ambiguous memory-only Hold examples are
excluded. This is a matched diagnostic sample, not an estimate of population
training loss. Report per-class NLL, accuracy/confusion, Response ranking AP,
calibration and errors by gold memory/history context.

Report physical fires, useful and wasted fires, timing and answer quality,
identical repeated answers, short-gap fires and concrete repeat episodes.
Use paired-video uncertainty intervals; a single seed and small cohorts cannot
identify a unique causal mechanism.

Output root:
`outputs/jevany-qwen35-2b-mixture-12h/scaling-diagnostic/`.
Entry points: `prepare_qwen35_scaling_diagnostic.py`,
`eval_qwen35_development_sweep.py`, `run_qwen35_scaling_diagnostic.py` and
`analyze_qwen35_scaling_diagnostic.py` in `scripts/`.

## Approval boundary

Automatic approval review rejected sending the new development/confirmation
rollout text to the external GPT-4.1 judge, stating that local diagnostics did
not authorize exporting those specific payloads. No such submission was made.
`--local-only` performs GPU diagnostics, computes timing proxies and prepares
reviewable request files on disk. It does not invoke the judge. Semantic margin
selection and confirmation remain pending explicit approval for that export.
Timing-only scores assume in-window answers are correct; they must not be
presented as semantic precision/recall or directly compared with WEB PAUC-F.

## Decision rule

If development-selected calibration also recovers final-checkpoint quality on
confirmation, retain the classification objective and investigate validation
selection/calibration first. If ranking or matched-input quality declines,
test a bounded lower-learning-rate continuation before increasing exposure.
If matched-input quality improves but calibrated closed-loop behavior still
deteriorates, investigate generated-history training contexts. Do not infer
numerical optimizer instability from falling training loss and a worse rollout
score alone. Choose the next experiment after inspecting the diagnostic and
its uncertainty; do not launch another long training run automatically.

## Completed local comparison

Both checkpoints have complete coverage of the same 2,016 decisions. NLL is
0.3256 at step 1,332 versus 0.3194 at step 1,776; the paired delta interval
[-0.0392, +0.0216] includes zero. Response AP is 0.6540 versus 0.6658, with
delta interval [-0.0196, +0.0438]. The final default decision point increases
Response recall and false Response predictions. This does not show numerical
instability or a demonstrated ranking collapse. Closed-loop semantic quality
still needs the queued sweep and approval for judging.

The context-order confound and a bounded single-intervention proposal are in
[the candidate context-order pilot](../plans/qwen35_2b_context_order_pilot.md).
The GPU sweep is queued behind a separate Gemma job on the same node.
