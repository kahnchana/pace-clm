# Three training experiments after JevAny J1

Date: 2026-10-25; J2 results added 2026-10-27; J1-S results added 2026-10-28; J1-S-R v2 and replay results added 2026-11-02

Status: J2, J1-S, and J1-S-R v2 completed; J1-S remains the JevAny-family candidate;
J2-S and J2-R remain unrun; J1-S-R v2 and its source-tail replay were not promoted;
v3 was stopped before a checkpoint

## Objective and evidence

Beat the promoted PACE Qwen `long-neg` controller on **GPT-4.1 judged PAUC-F**
under the existing WEB/EGO protocol: **57.47 WEB** and **46.05 EGO**. Keep
`Silence`, `Hold`, and `Response` as the action space. Do not adjust response
logits, thresholds, cooldowns, or other fire rules at inference. Any change in
when to fire must be learned during training.

The [completed J1 run](jevany_streaming_j1_10pct.md) scored 55.41 WEB and 40.48
EGO. EGO had 3,020 fires, of which 2,030 were judged wasted. J1 trained with
gold prior responses but rolled out with its own generated answers. The
[generator-feedback analysis](controller_generator_feedback.md) found that
swapping the answer path on WEB changed fire counts on 204/500 videos and
reduced judged PAUC-F to 43.31. Because that comparison also changed the
generator prompt, it demonstrates a coupled controller/generator trajectory,
not an isolated effect of adapter weights.

The promoted PACE adapter remains the answer generator for the primary
comparison. It is separate from the trainable controller adapter. We will not
add answer-generation loss to the controller: this round tests whether a
controller can learn the timing policy from video, request, and its **own prior
actions**, without reading provider-written answers.

## Common contract for all three experiments

- Start from the same promoted PACE Qwen initialization, JevAny pointer head,
  1 FPS causal frames, 32-frame window, visual encoding, candidate order, and
  action cross-entropy used by J1. Save the exact source, manifest, prompt,
  model, and generator hashes for each cell.
- The controller sees the standing request, causal frames/timestamps, and
  controller-owned prior action records. A prior `Response` record contains its
  action and time, never its natural-language answer. Replace both in-window
  assistant text and aged-out last-response text with the same action-only
  representation in training and inference. The answer generator may still
  receive its normal history; no generator output is copied into controller
  input. Reword the **controller** system/action instructions to describe this
  representation. Do not change the answer-generation prompt in the primary
  comparison.
- Retain the original human-verified `Silence` and `Response` targets. Treat
  `Hold` as derived supervision requiring a visible prior `Response` action in
  the controller's available history and evidence of event continuation.
  Mask or replace presentations whose context cannot support that meaning;
  record counts and examples before training. Do not silently relabel a
  verified `Response` as `Hold` or `Silence`.
- Use no inference-time response margin (`response_margin=0.0`) and no
  hand-coded cooldown. Keep the promoted PACE answer adapter, decoding, frame
  budget, and judge/scorer settings fixed across compared controllers.
- Keep train/development/test source videos disjoint. Use only training videos
  to construct new examples. The previously examined MarkII selection/final
  sets can serve as development diagnostics; freeze an additional disjoint
  MarkII set for final selection if available. WEB/EGO have already informed
  these hypotheses and are **not untouched selection sets**. Freeze a single
  checkpoint and training recipe before each WEB/EGO rollout.
- Select by **judged validation PAUC-F**, not loss, timing-only PAUC-F, or fire
  count alone. Report judged recall component, useful/wasted fires, repeats,
  Hold recall, fires/video by video-duration bucket, and paired per-video
  differences where raw predictions are available. Reproduce the promoted
  controller's raw predictions if a paired WEB/EGO comparison is claimed.

## Experiment 1 — J2: response-blind controller

**Change:** Train the J1 pointer/LoRA action objective on the original J1
decision exposure after converting every prior response in its controller
prompt/history/memory to an action-and-time record. Keep the original labels,
sampling, loss, update budget, and answer generator. Use the same controller
input construction for rollout. This changes the controller's history and the
instructions needed to interpret it; it does not claim to isolate prompt text
from history representation.

**Hypothesis:** The controller can use visual evidence and its own response
history to learn when another answer is warranted. Its fire schedule should
no longer depend on which model writes the answer.

**Preflight and readouts:**

1. Verify that no answer payload or response-text memory reaches the
   controller, including through recent assistant turns, the aged-out memory
   path, cached prompts, or rollout state. Check train/inference tokenization
   and causal frame parity.
2. On identical held-out videos, swap the answer generator while keeping the
   controller fixed. Require exactly equal action sequences and fire times
   under deterministic execution. Differences indicate an input leak or a
   separate nondeterminism bug, not a tuning opportunity.
3. Evaluate J2 with the promoted generator in closed loop. Compare judged
   validation PAUC-F and recall/precision components with J1 and promoted
   PACE. A fixed-fire-schedule generator replay is a separate diagnostic of
   answer quality, not a controller-training result.

### J2 completed result (2026-10-26)

J2 used the original J1 [manifest](../../outputs/jevany/manifests/train.jsonl)
unchanged: 263,760 presentations, including 41,220 `Response` and 7,038
`Hold`. A mechanical check found a visible earlier `Response` in the
controller's window or aged-out memory for every `Hold` presentation. That
check did not independently establish event-continuation semantics for every
`Hold`. The controller input converted recent assistant emissions and aged-out
response memory into actions and timestamps; the promoted PACE answer adapter
kept its normal prompt and history. The 1 FPS, 32-frame, pointer/LoRA,
action-cross-entropy, seed-42, and zero-margin settings matched J1. The
three-label [GPU smoke](../../outputs/jevany-j2/smoke/report.json) produced a
finite loss and nonzero gradients in all 500 trainable LoRA tensors. Full
training finished 263,760 presentations and 8,243 optimizer steps on eight
H200 GPUs. The frozen [checkpoint](../../outputs/jevany-j2/full-seed42/step-008243),
[training summary](../../outputs/jevany-j2/full-seed42/summary.rank0.json),
and [source/manifest hashes](../../outputs/jevany-j2/source/preflight.json)
identify the run.

Full-split GPT-4.1 judged `pauc-f-v1`, `omega=0.5`; `R` is the scorer's PAUC
recall component, not ordinary event recall. PACE is a historical point
estimate; its raw predictions were not available for a paired comparison.

| Split | Controller | P | R | PAUC-F | Fires | Useful | Wasted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WEB (500) | Promoted PACE `long-neg` | 70.67% | 48.43 | **57.47** | 1,132 | 800 | 332 |
| WEB (500) | J1 | 65.48% | 48.03 | 55.41 | 1,240 | 812 | 428 |
| WEB (500) | J2 response-blind | 63.95% | 46.94 | 54.14 | 1,237 | 791 | 446 |
| EGO (326) | Promoted PACE `long-neg` | 46.14% | 45.96 | **46.05** | 1,634 | 754 | 880 |
| EGO (326) | J1 | 32.78% | 52.90 | 40.48 | 3,020 | 990 | 2,030 |
| EGO (326) | J2 response-blind | 29.51% | 50.71 | 37.30 | 3,213 | 948 | 2,265 |

The J2 score sources are [WEB](../../outputs/jevany-j2/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json)
and [EGO](../../outputs/jevany-j2/evaluation/benchmarks/ego/pauc_f_summary-gpt-4.1-v1.json);
the J1 sources are [WEB](../../outputs/jevany/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json)
and [EGO](../../outputs/jevany/evaluation/benchmarks/ego/pauc_f_summary-gpt-4.1-v1.json).
The promoted PACE figures are in the
[cell-4 report](../../../stream-gemma/agent_notes/docs/experiments/qwen_w1_long_neg.md).

**Decision:** J2 failed to reduce overfiring or improve judged PAUC-F. Against
J1, WEB had nearly the same fire count (−3), but 18 more wasted and 21 fewer
useful fires. EGO had 193 more fires, 235 more wasted, and 42 fewer useful;
PAUC-F fell by 1.27 WEB and 3.18 EGO points. Relative to promoted PACE, J2
was 3.34 points behind on WEB and 8.75 behind on EGO. The response-blind
representation, as trained here, did not yield a better firing policy.

This was one seed. The shortened preflight did **not** run the planned
generator-swap invariance test, held-out judged validation selection, or
event-span audit of all `Hold` labels. Thus the hypothesis that the controller
is fully invariant to answer-generator choice remains unverified. J2 also
changed both history representation and controller instructions, so these
results do not isolate either change. No J2 checkpoint was selected using
WEB/EGO; the only full-run checkpoint was evaluated after training.

## Experiment 2 — J2-S: history-grounded negative sampling

**Change:** Train a fresh J2 architecture from the same initialization, with
the same action-only prompt and objective, but change **which training
decisions are presented**. Increase distinct `Hold` and post-response
`Silence` examples from long videos and repeated-event intervals. Replace
easier negative presentations rather than extending the run. Preserve the
J1 total of 263,760 presentations and its 41,220 `Response` presentations;
report unique video/frame positions, per-video exposure, class counts, and
the exact sampling seed. Do not simply duplicate a small negative set or
raise Hold loss weight: earlier frozen-head work found that extra exposure
and doubled Hold loss could reduce PAUC-F.

**Eligibility:** A `Hold` training example must expose a prior `Response`
action in the available action history and belong to the continuation of that
reported event. A post-response `Silence` example must retain its verified
source target. Exclude ambiguous cases from this cell and replace them with
other eligible negatives so the total exposure stays matched. Check the
event-span and history mapping mechanically and save a duration-stratified
sample for inspection; this experiment does not create new action labels.

**Hypothesis:** Better coverage of post-fire visual states will teach the
model to suppress redundant EGO fires while preserving first detections and
repeated *new* events requested by “whenever/every time” policies.

**Readouts:** Compare J2-S against J2 at equal exposure, especially judged
PAUC-F, value-adding recall, wasted fires, and repeats in long videos. A
smaller fire count without PAUC-F improvement does not pass.

## Experiment 3 — J2-R: closed-loop error training

**Change:** Roll a frozen J2 controller forward on **training videos** with
its action-only history. Mine false fires at verified `Silence` positions and
misses at verified `Response` positions. Train a fresh matched J2-R variant
with a fixed quota of these mined causal prefixes in place of ordinary
presentations, preserving 263,760 total presentations and the 41,220
`Response` presentations used by J1/J2. Keep the source video/query split and
prior-action trace for each example. Version the mining controller, predictions, manifest, and
selection rule; do not mine from WEB/EGO or validation videos.

**Label rule:** Use the original verified `Silence`/`Response` target only
when it remains meaningful under the rolled-out action history. In
particular, do not assume a `Hold` target remains valid if the model missed
the earlier response. Mask cases where an earlier wrong fire changes whether
the current event has already been covered. Record how many examples were
discarded for this reason; do not manufacture new human-verified labels.

**Hypothesis:** Training on the controller's own past actions will reduce the
teacher-forced versus closed-loop gap and correct the false-fire patterns
observed in EGO without an inference rule.

**Readouts:** Compare J2-R against J2 at matched exposure on held-out
closed-loop validation. Measure whether false fires decrease *and* useful
response coverage survives. If J2-S and J2-R each improve independently,
their combination is a later, separately labeled experiment.

## Execution and decision sequence

1. Freeze a common baseline report for J1 and promoted PACE, including the
   same generator and scorer settings. Save a new video-disjoint validation
   split or explicitly mark reused MarkII sets as development only.
2. Implement and smoke J2's response-blind training/rollout path. Run the
   provider-swap invariance test before a full training run. Then complete J2.
3. Build and check the J2-S sampling manifest; run J2-S from the same initial
   weights and with the same presentation budget as J2.
4. Mine training-video rollouts from frozen J2; build and audit J2-R's masked
   examples; run J2-R from the same initial weights and matched budget.
5. Pick one frozen candidate by judged validation PAUC-F. Confirm it on full
   WEB (500) and EGO (326) with `pauc-f-v1`, GPT-4.1, and `omega=0.5`, and
   compare with the promoted PACE targets of 57.47/46.05. Report both splits
   even if only one improves. The historical B0 point estimates alone do not
   support a paired significance claim.

These are three controlled training cells, not three cumulative edits to a
single checkpoint. J2-S and J2-R branch from the J2 input design so their
sampling and closed-loop interventions can be interpreted separately.

## Recommendation after J2

For the **next full training experiment**, run **J1-S: history-grounded
negative sampling on the J1 input design**, rather than J2-S as originally
ordered above. J2-S remains a valid test of whether better negatives can
rescue response blindness, but it inherits a controller that is already below
J1 on both splits and overfires more on EGO. J1-S tests the intervention most
directly aimed at the observed failure while keeping the stronger J1 history
representation fixed. This historical recommendation was completed below.

- Start a fresh model from the same promoted PACE initialization and J1
  pointer/LoRA objective; do not continue the trained J1 or J2 checkpoint.
  Keep J1's controller prompt/history and the promoted answer generator.
- Change only the **presentation manifest**: replace easier negative
  presentations with distinct, verified post-response `Silence` and
  event-continuation `Hold` positions, emphasizing long EGO-like videos and
  repeated-event intervals. Preserve 263,760 total presentations and 41,220
  `Response` presentations. Keep source train videos disjoint from validation,
  WEB, and EGO. Audit `Hold` event spans and visible prior actions before
  training; report per-video exposure, unique positions, class counts, and
  sampling seed. Do not create or relabel verified targets.
- Use the same update budget and `response_margin=0.0`. Pick a frozen
  checkpoint using **judged, closed-loop validation PAUC-F**, while checking
  useful/wasted fires, recall, repeats, and long-video fire rates. Then run
  full WEB/EGO confirmation with the unchanged generator and scorer. A lower
  fire count alone is insufficient: useful coverage and PAUC-F must improve.

If J1-S improves both judged validation PAUC-F and EGO wasted-fire volume,
that would support negative-example coverage as a useful lever independent of
response blindness. It would not establish that response blindness works;
that remains a separate question for a later J2-S or invariance study.

## J1-S completed result (2026-10-28)

J1-S trained a fresh J1-history controller from the promoted PACE adapter,
keeping the J1 pointer/LoRA loss, seed 42, 1 FPS/32-frame input, 263,760
presentations, 41,220 `Response` presentations, 8,243 updates, zero response
margin, and promoted answer generator. The only experiment intervention was
the [presentation manifest](../../outputs/jevany-j1s/manifests/train.jsonl).
The [manifest audit](../../outputs/jevany-j1s/manifests/audit.json) records
10,000 distinct added post-response `Silence` positions, 1,204 added
in-window event-continuation `Hold` positions, and 1,407 excluded memory-only
`Hold` presentations whose event relationship could not be established from
the current window. Unique `Hold` positions increased from 5,258 to 5,733;
the resulting class counts were 215,705 `Silence`, 6,835 `Hold`, and 41,220
`Response`. The [preflight](../../outputs/jevany-j1s/source/preflight.json)
pins source and manifest hashes. The [final checkpoint](../../outputs/jevany-j1s/full-seed42/step-008243)
and [training summary](../../outputs/jevany-j1s/full-seed42/summary.rank0.json)
identify the frozen run. An initial training attempt stopped after a full
`/tmp` Triton cache caused I/O failures; training resumed from step 750 with
the cache on `/dev/shm` and completed the intended presentation budget.

Judged validation used the two reused 200-video MarkII cohorts as development
diagnostics. J1-S scored **74.61** on selection and **71.81** on final,
compared with J1's **76.18 / 72.55**. These cohorts have previously informed
the experiment; they are not untouched final selection sets.

Full-split GPT-4.1 judged `pauc-f-v1`, `omega=0.5`:

| Split | Controller | P | R | PAUC-F | Fires | Useful | Wasted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WEB (500) | Promoted PACE `long-neg` | 70.67% | 48.43 | **57.47** | 1,132 | 800 | 332 |
| WEB (500) | J1 | 65.48% | 48.03 | 55.41 | 1,240 | 812 | 428 |
| WEB (500) | J1-S | 65.36% | 49.77 | 56.51 | 1,357 | 887 | 470 |
| EGO (326) | Promoted PACE `long-neg` | 46.14% | 45.96 | **46.05** | 1,634 | 754 | 880 |
| EGO (326) | J1 | 32.78% | 52.90 | 40.48 | 3,020 | 990 | 2,030 |
| EGO (326) | J1-S | 35.77% | 50.55 | 41.90 | 2,555 | 914 | 1,641 |

The J1-S score sources are [WEB](../../outputs/jevany-j1s/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json)
and [EGO](../../outputs/jevany-j1s/evaluation/benchmarks/ego/pauc_f_summary-gpt-4.1-v1.json).
Against J1, J1-S gained 1.10 WEB and 1.42 EGO PAUC-F points. EGO had 465
fewer fires and 389 fewer wasted fires, but also 76 fewer useful fires. WEB
had 117 more fires, including 75 more useful and 42 more wasted. Within gold
reply windows, extra fires fell by 165 on EGO and rose by 17 on WEB. On a
paired video bootstrap of the **global** PAUC-F difference, the 95% intervals
were −0.08 to 2.45 WEB and −0.48 to 3.35 EGO points; both include zero.
The [paired diagnostics](../../outputs/jevany-j1s/reports/paired_diagnostics.json)
also contain per-video fire-count differences and duration buckets. Promoted
PACE remains a historical point estimate without raw predictions for a paired
comparison. The rollout artifacts do not contain frame-level Hold decisions,
so Hold recall cannot be estimated from these benchmark outputs.

**Decision:** Promote the frozen J1-S checkpoint as the current **JevAny-family
candidate**: it is the best evaluated JevAny controller on both full WEB and
EGO splits. This is a within-family promotion for follow-up experiments, not
a replacement for the production PACE `long-neg` controller. J1-S reduced EGO
overfiring and improved full WEB/EGO scores relative to J1, but did not beat
PACE on either split. Its judged MarkII development scores were lower than
J1's, and the paired benchmark intervals do not establish a reliable gain.

## Recommended next experiment — J1-S-R closed-loop error exposure

**Why this cell:** J1-S cut 389 wasted EGO fires but also lost 76 useful EGO
fires relative to J1. It still made 1,641 wasted fires on EGO, and its WEB
fire count rose. The next intervention should teach the controller from its
own mistaken causal histories while preserving the post-response coverage
that made J1-S the strongest JevAny checkpoint. J2-S and J2-R would instead
branch from the weaker response-blind J2 input design.

**Controlled change:** Freeze the promoted J1-S checkpoint as the mining
controller. Roll it out only on training videos with the promoted answer
generator and its normal J1 text history. Mine verified `Silence` positions
where it fired and verified `Response` positions where it missed. Train a
fresh **J1-S-R** model from the same promoted PACE initialization, keeping
J1-S's prompt, pointer/LoRA objective, training budget, 263,760 total
presentations, 41,220 `Response` presentations, and zero inference margin.
Replace a fixed quota of J1-S manifest presentations with eligible mined
causal prefixes of the **same target class**; retain the J1-S post-response
negative coverage. Do not continue training the J1-S checkpoint or add
answer-generation loss.

**Label and audit gate:** Keep a verified target only when it still means the
same action under J1-S's rolled-out prior actions and generated answers. Mask
missed first detections already covered by an earlier wrong fire, `Hold`
contexts missing their prior response, and other event-coverage ambiguities.
Save the mining predictions, controller/generator hashes, original video and
question IDs, prior-action traces, replacement quota, discarded counts, and
the manifest seed. Check train/development/test video disjointness. If too few
unambiguous errors remain for a diverse quota, stop this cell before full
training rather than duplicating a small set.

**Decision rule:** Freeze a video-disjoint judged validation cohort before
training if one is available; treat the reused MarkII cohorts as development
diagnostics. Select by closed-loop judged PAUC-F while checking useful and
wasted fires, first-event coverage, repeated new events, and long-video fire
rates. Then confirm the frozen checkpoint on full WEB and EGO under the same
GPT-4.1 `pauc-f-v1`, `omega=0.5` protocol. Advance only if useful coverage
survives and judged PAUC-F improves over J1-S; fewer fires alone do not pass.

### J1-S-R mining gate result (2026-10-29)

The frozen J1-S checkpoint and promoted answer generator completed closed-loop
rollouts on 1,024 training videos (84,599 frames), using the normal J1 text
history and zero response margin. A new 200-video MarkII validation cohort was
frozen before mining; its videos and the two reused development cohorts are
disjoint from the training videos. The [rollout traces](../../outputs/jevany-j1sr/mining/rollout)
and [gate audit](../../outputs/jevany-j1sr/mining/gate.json) identify the input
set and errors.

The prespecified manifest replacement quota was 512 `Silence` and 128
`Response` presentations, with at least 64 source videos per class and at
most four replacements from any one video. The conservative audit retained
only **70 `Silence` errors from 56 videos** and **19 `Response` misses from 19
videos**. It discarded 492 `Silence` fires within eight seconds of a verified
`Response`, 1,342 missed `Response` positions with a prior fire in the preceding
32 seconds, and 10 with a prior generated answer overlapping the gold answer.
The 1,024-video sample covers 21% of the 4,857 eligible unambiguous-question
training videos under the 180-frame cap. Extrapolation from this sample is
insufficient to justify the fixed quotas; this is a sampling inference, not an
exhaustive census of all training videos.

**Gate decision:** Stop J1-S-R before full training. No J1-S-R checkpoint or
WEB/EGO benchmark scores exist. Reducing the quota after inspecting mined
errors would turn this into a different, much smaller intervention and would
not meet the planned diversity gate.

### J1-S-R event-aware extension (v2; 2026-10-29)

The stopped pilot remains a separate, immutable cell. The v2 extension keeps
its 512 `Silence` / 128 `Response` replacement quota, four-per-video cap,
64-video-per-class minimum, J1-S mining checkpoint, fresh PACE initialization,
training budget, and frozen 200-video validation cohort. It uses the saved
1,024-video pilot traces and a frozen disjoint top-up set of all 3,833 remaining
eligible videos from the J1-S source training manifest. No evaluation labels
are used to select top-up videos.

The revised gate uses every annotated position in the source episode turns,
including positions not sampled as supervised presentations, and reconstructs
each gold event from a source `Response` onset
and its consecutive annotated `Standby` frames. A model fire is assigned to a
gold event only when it lies in that event's onset-minus-four through
end-plus-four-second window and matches no other event window. A `Silence`
error is eligible only outside all such windows. A missed `Response` is
eligible only if its event has no assigned or ambiguous fire; prior answers
that overlap its gold caption and uncovered prior gold events in the 32-frame
causal prefix are masked. All kept targets are source verified and get the
actual 32-frame self-generated action and answer history. The fixed quota and
diversity gate run on the combined traces before any training.

The pilot re-audit under these rules yields 226 `Silence` candidates from 157
videos and 44 `Response` candidates from 42 videos. These counts do not pass
the prespecified quota on their own; the top-up rollout is required. The v2
protocol and outputs live under `scripts/build_jevany_j1sr_v2.py` and
`outputs/jevany-j1sr-v2/`.

The 3,833-video top-up rollout completed on 2026-10-30. Combined with the
pilot, the event-aware gate found 1,173 eligible `Silence` examples from 864
videos and 210 eligible `Response` misses from 204 videos. The fixed quota
passed: 512 `Silence` replacements were selected from 417 videos and 128
`Response` replacements from 125 videos. This gate preceded the training and
evaluation results below. See the
[v2 gate audit](../../outputs/jevany-j1sr-v2/mining/gate.json).

### J1-S-R v2 result and EGO pause (2026-11-02)

Fresh training completed all 8,243 steps on eight GPUs using 263,760
presentations. On the frozen 200-video validation cohort, J1-S-R made only one
physical response, outside every gold window: 0.005 responses/video and 0.00
judged PAUC-F. J1-S made 699 responses (3.495/video) and scored 71.62 judged
PAUC-F on the same cohort. The J1-S-R judge batch was correctly empty; its
zero judged score used no semantic API requests.

Full WEB confirmed near silence: J1-S-R made two responses across 500 videos
(0.004/video), with one value-adding fire among 1,328 gold turns. Its
timing-only P / PAUC-R / PAUC-F were 50.00 / 25.05 / 33.38; judged GPT-4.1
values were 50.00 / 25.02 / 33.35. J1-S made 1,357 responses (2.714/video),
with 887 value-adding fires, and scored 56.51 judged PAUC-F. PAUC-R has a
25-point no-response floor and precision has only two fires in its denominator,
so the J1-S-R F score alone obscures the near-total loss of useful coverage.
See the [WEB timing gate](../../outputs/jevany-j1sr-v2/reports/web_gate.json)
and [judged WEB summary](../../outputs/jevany-j1sr-v2/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json).

Per the user's conditional request, EGO was paused when WEB showed the same
near-silent trend. The first gate implementation mistakenly also required
PAUC-F below 5, allowing EGO to start because one of two WEB fires was useful.
This was corrected to the response-rate criterion; EGO stopped after 59 of 326
videos and has no complete score. The 59 partial traces are retained for
provenance. **Decision:** Do not promote J1-S-R v2; retain J1-S as the
JevAny-family candidate pending a separate analysis of response collapse.

#### Failure analysis and next controlled check

The [failure report](../../outputs/jevany-j1sr-v2/reports/web_result.md)
records the full frozen recipe, gate, validation and WEB P/R/PAUC-F and
response rates, EGO partial-run status, training-order audit, and proposed
diagnostics. The key implementation finding is that the one-epoch trainer
does not shuffle manifest rows. The v2 builder removed 640 source
presentations at distributed positions but appended all 640 replacements at
the end after sorting by new high episode IDs. Each GPU rank therefore saw
64 mined `Silence` examples followed by 16 mined `Response` examples in its
final 80 presentations, leaving the last roughly 20 optimizer updates
entirely on mined errors. This was an unintended training-order change.

The mined block was also very different in context length: 579/640 mined
rows (90.5%) had 32 frame turns, versus 2,518/263,120 retained source rows
(0.96%). The final nine-presentation mean loss across ranks had median 7.55
for J1-S-R v2 versus 0.19 for J1-S. The loss jump coincides with the
mined-only tail, but does not establish that order alone caused the silent
policy; long-context distribution shift or mislabeled causal histories may
also matter. The final-turn `</Silence>` placeholder is ignored by the
training renderer, which uses the row label as target, so it is not a direct
label contradiction.

Before another full train, evaluate existing checkpoints immediately before
and after the mined tail on the frozen cohort, recording responses/video,
event coverage, and action logits. For a new controlled cell, distribute
mined replacements throughout the epoch, match or stratify context lengths,
semantically audit a sample of selected labels, and use a small closed-loop
response-rate/coverage gate before committing to 8,243 steps. Keep a fresh
PACE initialization and fixed budgets for the first ordering test; changing
initialization, quota, or learning rate should be separate cells.

### Checkpoint and exact-tail diagnosis (2026-11-02)

The [checkpoint diagnosis](../../outputs/jevany-j1sr-v2/diagnostics/checkpoint_diagnosis.md)
located the response collapse in the last 43 updates. On the same frozen 200
videos, step 8,200 made 742 responses and covered 504/659 gold windows; final
step 8,243 made one response and covered none. GPT-4.1 judged MarkII PAUC-F
was 70.74 at step 8,200 versus 71.62 for J1-S, so the pre-tail checkpoint is
a diagnostic, not a newly selected candidate.

An exact replay resumed the step-8,200 optimizer and RNG state and replaced
only the 640 appended mined rows with the 640 source rows removed for v2.
The first 263,120 manifest rows had an identical SHA-256 prefix. On a fixed
64-video subset, the replay made 173 responses, covered 150/241 windows, and
scored 75.20 GPT-4.1 judged PAUC-F. J1-S scored 73.88 on those same 64 videos;
the original mined-tail final checkpoint covered zero windows. This directly
implicates the mined tail in the extreme silence, while leaving ordering,
long-context distribution, and label quality as separate mechanisms. The
replay covered fewer windows than J1-S and is not promoted.

On a matched 16-video action-logit probe, all first divergences between step
8,200 and the original final checkpoint were `Response` versus `Silence` on
identical prior actions and answers. The median Response-minus-best-other
logit margin fell by 17.4 at those decisions. The exact source-tail replay
also suppressed the first pre-tail Response in 13/16 videos, but its median
margin drop was 4.0. See the saved
[logit comparison](../../outputs/jevany-j1sr-v2/diagnostics/logits/comparison.json).

The [label spot check](../../outputs/jevany-j1sr-v2/diagnostics/label_audit.md)
flagged one visually questionable mined positive. A fresh interleaved manifest
contains exactly the same v2 presentations and class counts, with mined rows
distributed across the source replacement positions. The next cell uses a
step-1,000 closed-loop response/coverage safety gate before completing its
matched 8,243-step budget.

The v3 canary was started, then stopped at the user's request before a
checkpoint or gate result so the source-tail replay could receive benchmark
evaluation first. No v3 quality result is available.

#### Source-tail replay benchmark result and EGO stop

The frozen step-8,243 source-tail replay used zero response margin and the
promoted answer generator. The complete 500-video WEB rollout and GPT-4.1
judge used `pauc-f-v1` with `omega=0.5`, matching the earlier benchmark
protocol. Its [judged score](../../outputs/jevany-j1sr-v2/diagnostics/source-tail-replay/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json)
was **44.67 PAUC-F**, with 205 fires (0.410/video), 174 value-adding fires,
31 wasted fires, 84.88% fire precision, and a 30.31 PAUC recall component.
The [complete predictions](../../outputs/jevany-j1sr-v2/diagnostics/source-tail-replay/evaluation/benchmarks/web/predictions.jsonl)
and judge outputs are saved with that summary.

| WEB controller | Videos | Fires | Useful | Wasted | Judged PAUC-F |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted PACE `long-neg` | 500 | 1,132 | 800 | 332 | 57.47 |
| J1-S | 500 | 1,357 | 887 | 470 | 56.51 |
| J1-S-R v2 final mined tail | 500 | 2 | 1 | 1 | 33.35 |
| J1-S-R v2 source-tail replay | 500 | 205 | 174 | 31 | 44.67 |

The replay improved on the near-silent mined-tail checkpoint by 11.32 points,
but remained 11.84 below J1-S and 12.80 below promoted PACE. Its response
rate was far below J1-S's 2.714 fires/video, explaining why high fire
precision did not translate into competitive event coverage. The 75.20
judged score on the first 64 frozen MarkII videos was a useful diagnosis of
the tail failure, not a reliable estimate of full WEB performance. The
replay is **not promoted**.

At the user's request, the replay EGO rollout was stopped after 38/326
videos had produced saved [partial rank traces](../../outputs/jevany-j1sr-v2/diagnostics/source-tail-replay/evaluation/benchmarks/ego/manifest.json).
The launcher and all eight GPU workers exited. No merged EGO prediction file,
GPT-4.1 judgment, or complete EGO score exists for this replay. Do not compare
the partial traces numerically with full EGO results.
