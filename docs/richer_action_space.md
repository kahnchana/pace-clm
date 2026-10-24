# Richer, Variable Action Spaces for Streaming Video Agents

## Overview

PACE-CLM currently chooses among three fixed actions: **Silence**, **Hold**, and
**Response**. This document proposes a more consequential direction: use the
streaming VLM to choose a **language-defined communicative action**—for example,
monitor, notify, alert, request evidence, guide, update, correct, or escalate—then
generate content only when that action calls for communication.

The central experiment is a **variable action policy**. The same controller sees
different candidate actions for proactive QA, an instructional assistant, or a
factory watcher. Success is not merely better three-way classification; it is
transfer to new policy sets, new action descriptions, and held-out actions while
remaining grounded in causal video evidence, conversation history, and the
user's standing request.

## Why this is a new hypothesis

The completed PACE-CLM study is an important negative result, not evidence for
this proposal:

- Three fixed CLM prototypes did not reliably beat a capacity-matched MLP.
- The selected frozen MLP reduced firing but lost useful-response coverage and
  underperformed promoted PACE on judged WEB/EGO PAUC-F.
- With a fixed set of three labels, a contrastive action head is largely a
  metric-learning reparameterization of a classifier.

The richer-action hypothesis is therefore:

> A language-conditioned action space becomes valuable when the available
> actions, their descriptions, or their policies change across tasks—and the
> model must ground those descriptions in the current stream rather than learn
> a permanent output index for each class.

This should be tested against a fixed classifier, not assumed from the CLM
architecture. If variable or unseen actions do not transfer, the richer model
has not established an advantage.

## Application motivation: Glimpse-style watcher behavior

The Glimpse harness already behaves more like an intervention router than an
answer gate. Its watcher distinguishes ordinary scene notes, product readiness,
serial-label detection, readable serial values, known blocking issues, and
unreadable warnings. Deterministic code then routes these signals differently,
suppresses duplicates, and chooses whether to observe, ask for better evidence,
coach the user, or initiate troubleshooting.

Relevant implementation references:

- `/Users/kranasinghe/repo/glimpse/server/harness/conversation.js`
  defines and parses watcher outputs such as `PRODUCT_READY`, `SERIAL`,
  `SERIAL_LABEL`, `ISSUE`, and `ERROR`.
- `/Users/kranasinghe/repo/glimpse/server/harness/routes/watch.js` converts those
  outputs into structured application signals.
- `/Users/kranasinghe/repo/glimpse/public/harness/turn.js` deduplicates issues,
  prioritizes a known issue over an unreadable error, and launches the relevant
  follow-up path.

That application suggests a general streaming-agent decomposition:

```text
causal video + standing request + interaction history + available actions
                              |
                              v
                    semantic action selector
                              |
               +--------------+---------------+
               |                              |
        nonverbal/control action       communicative action
        monitor, hold, inspect         notify, alert, guide,
                                      update, correct, escalate
                                               |
                                               v
                                  action-conditioned generation
```

The selector decides **what kind of intervention is warranted**. The generator
decides **what to say**. Application code retains authority over routing,
deduplication, safety constraints, and tool calls.

## Proposed action ontology

The first ontology should stay small enough to label consistently while being
meaningfully richer than `Response`.

| Action | Operational meaning | Example |
| --- | --- | --- |
| **Monitor** | Continue observing; nothing warrants an intervention. | No relevant change is visible. |
| **Hold** | Relevant evidence persists, but it has already been communicated or is not yet actionable. | The same event continues after an earlier notification. |
| **Notify** | Report a useful, non-urgent state change. | “The cycle has finished.” |
| **Alert** | Immediately report a blocking, dangerous, or time-critical condition. | “The machine is showing an overheat warning.” |
| **RequestEvidence** | Ask for missing visual or textual evidence. | “Please point the camera at the display.” |
| **Guide** | Give the next grounded procedural instruction. | “Now attach the milk tube.” |
| **Update** | Add material information to an earlier valid response. | “The display now shows error 17.” |
| **Correct** | Repair an earlier incorrect interpretation. | “Correction—the container is not fully seated.” |
| **Escalate** | Hand off because safe or grounded assistance is unavailable. | “This condition requires a qualified technician.” |

`Answer` should not necessarily remain one universal action. In proactive QA it
can be the domain-facing name for `Notify` or a specialized action whose
generated content directly answers the query. The important separation is
between the **action type** and the **natural-language payload**.

### Factorized alternative

A flat nine-way taxonomy may create sparse combinations. A later model can
factor an intervention into:

1. **Communication decision:** monitor, hold, communicate.
2. **Speech act:** answer, notify, warn, request, guide, correct, escalate.
3. **Urgency:** routine, attention, immediate.
4. **Relation to history:** new, continuation, update, correction.

For the first experiment, use atomic natural-language candidates. Factorization
is an ablation or follow-up, not an excuse to change several modeling choices at
once.

## Variable policies

The most compelling experiment exposes one controller to different candidate
sets. Candidate descriptions, not hard-coded indices, specify the active
policy.

| Policy | Candidate actions | Typical standing request |
| --- | --- | --- |
| **Proactive QA** | `Monitor`, `Hold`, `Answer`, `Update`, `Correct` | “Tell me when you have enough evidence to answer, and update me if the answer changes.” |
| **Instructional assistant** | `Monitor`, `Hold`, `Guide`, `RequestEvidence`, `Correct`, `Escalate` | “Guide me through this procedure, but ask for a better view rather than guessing.” |
| **Factory watcher** | `Monitor`, `Hold`, `Notify`, `Alert`, `RequestEvidence`, `Escalate` | “Watch this station; notify me of completion and alert me immediately to blocking or unsafe conditions.” |

These policies deliberately overlap without being identical. `Monitor`, `Hold`,
and `RequestEvidence` may be reusable, while `Answer`, `Guide`, and `Alert`
encode different user expectations.

### Critical policy principle

Action labels cannot be inferred from pixels alone. The same visual event may
correctly produce different actions:

| Visual event | Standing request | Correct action |
| --- | --- | --- |
| Machine cycle completes | “Tell me when it finishes.” | `Notify` |
| Machine cycle completes | “Only interrupt me for unsafe conditions.” | `Monitor` |
| Warning indicator appears | “Warn me about blocking conditions.” | `Alert` |
| Warning text is unreadable | Same request, but poor camera view | `RequestEvidence` |
| Previously reported warning remains unchanged | Same request and history | `Hold` |

Every training and evaluation example must therefore include a standing
request or policy. Otherwise `Notify` versus `Alert` becomes an inconsistent
visual classification problem.

## Model formulation

The existing PACE-CLM mechanism can generalize mathematically to any number of
candidate actions:

```text
h_s = frozen_or_adapted_streaming_vlm(state)
h_a = action_encoder(candidate_description)

z_s = normalize(state_head(h_s))
z_a = normalize(action_head(h_a))
score(s, a) = exp(logit_scale) * z_s^T z_a
```

At inference, encode only the actions allowed by the current policy and choose
among them. An action such as `Notify` or `Guide` invokes the unchanged response
generator with the chosen action description included as a control instruction.
`Monitor` and `Hold` do not generate text.

### What must differ from the completed MVP

- Candidate sets vary across examples or policies.
- Training uses multiple semantically equivalent descriptions per action.
- At least one evaluation holds out an action, a policy composition, or both.
- The model is compared with a fixed classifier and with a classifier retrained
  for each policy.
- The action encoder must preserve enough semantic structure to score a novel
  description; freely learning three prototype vectors is insufficient.

The current `ContrastiveActionModel` already accepts an arbitrary action matrix,
but `Action`, label mapping, configs, datasets, losses, and evaluators assume
exactly three actions. Do not silently modify `actions_v1.yaml`; introduce a
versioned dynamic-action contract when implementation begins.

## Data strategy

Most richer action labels do not exist natively. Use three evidence tiers and
report them separately.

| Tier | Source | Examples | Use |
| --- | --- | --- | --- |
| **Direct** | Existing human timing/content annotations | PACE Silence, response onset, event continuation | High-confidence anchors and evaluation |
| **Derived** | Deterministic transformation of annotations/history | Hold, later response as Update, mismatched policy as Monitor | Scalable training after validation |
| **Synthetic** | Controlled policy, history, or visibility intervention | Correct, RequestEvidence, Alert, Escalate | Training augmentation; requires audit |

### PACE conversion

PACE provides strong causal timing and response content. It is best suited to:

- `Monitor`: existing irrelevant or non-event intervals.
- `Hold`: ongoing-event frames after an already warranted response.
- `Answer`/`Notify`: first sufficient-evidence response onset.
- `Update`: a later distinct response to the same query that adds information.

PACE alone is unlikely to provide enough trustworthy `Alert`,
`RequestEvidence`, or `Escalate` supervision. Those require explicit policy and
counterfactual construction.

Recommended conversion procedure:

1. Preserve the original causal frames, timestamps, query, answer, and video
   split.
2. Generate several standing-request variants with an explicit policy.
3. Assign a communicative action using the original evidence interval plus the
   new request and prior interaction history.
4. Generate or reuse the payload text only after the action is fixed.
5. Verify that a causal prefix supports both the action and its content.
6. Store provenance, transformation version, label tier, and verifier outcome.

### Controlled counterfactuals

Counterfactual histories and observations create labels that are otherwise rare:

- **Correct:** insert a plausible but false prior answer, then reveal evidence
  supporting the gold answer.
- **Update:** insert a valid incomplete prior answer, then reveal additional
  evidence.
- **RequestEvidence:** crop, blur, occlude, or omit the decisive view while
  keeping the standing request fixed.
- **Escalate:** provide conflicting evidence or an unsupported condition under a
  policy that forbids guessing.
- **Alert versus Notify:** keep the event fixed but change the standing policy
  and urgency requirement.

Counterfactual generation must not leak future gold text into the state. Avoid
easy artifacts such as always using one phrase for corrupted memories or one
image degradation for `RequestEvidence`.

### VideoChat3 and MMDuet2

- [VideoChat3-OL617K](https://huggingface.co/datasets/MCG-NJU/VideoChat3-OL617k)
  converts video QA into causal streams with localized clue intervals and
  explicit `Silence`, `Standby`, and `Response` states. Its timing supervision
  is well suited to policy-conditioned relabeling, but it does not natively
  provide the richer speech acts.
- [MMDuet2](https://openreview.net/pdf?id=rxQnMSNCUs) provides multi-turn
  proactive interaction and trains response timing with SFT and RL. It is useful
  for histories, induced states, and response-density behavior, but its core
  action distinction is still respond versus remain silent. Rich action labels
  would need post-labeling or transformation.

### Proposed `PACE-Acts` record

Each derived example should make the policy and provenance explicit:

```json
{
  "video_id": "...",
  "decision_time": 42.0,
  "causal_frame_ids": ["..."],
  "standing_request": "Warn me immediately if a blocking condition appears.",
  "policy_id": "factory_v1",
  "candidate_actions": [
    {"id": "monitor", "description": "Continue observing without interrupting."},
    {"id": "alert", "description": "Warn immediately about a blocking or unsafe condition."},
    {"id": "request_evidence", "description": "Ask for a clearer view before deciding."}
  ],
  "gold_action_id": "alert",
  "payload": "The display is showing an overheat warning.",
  "history": ["..."],
  "label_tier": "synthetic_verified",
  "source_annotation": "...",
  "transformation_version": "pace-acts-v1"
}
```

Split at the source-video level before producing variants. All variants of one
video must remain in one split.

## Experiments

### Core comparison

| ID | Method | Candidate behavior | Purpose |
| --- | --- | --- | --- |
| **B0** | Existing PACE token decision | Fixed `Silence/Hold/Response` | Preserve the promoted reference |
| **B1** | Capacity-matched MLP | Fixed output indices | Test ordinary classification |
| **B2** | Fixed-action CLM | Same candidates every example | Reproduce the completed formulation |
| **V0** | Variable-action CLM | Policy-specific candidate descriptions | Test language-conditioned selection |
| **V1** | Variable-action classifier | Policy ID plus union of fixed output indices | Strong non-language baseline |
| **V2** | Per-policy classifier | Separate classifier trained for each policy | Upper-cost specialist baseline |

All methods must use the same causal visual evidence, histories, train/test
videos, generator, and action payloads where applicable.

### Generalization tests

1. **Description paraphrase:** train and test with disjoint phrasings of known
   actions.
2. **Unseen candidate composition:** train on individual policy sets, then test
   a new mixture of known actions.
3. **Held-out action:** exclude one action from controller training and introduce
   only its description at test time.
4. **Few-shot action adaptation:** evaluate the held-out action after 10, 50,
   100, and 500 labeled examples.
5. **Cross-domain policy transfer:** train mainly on proactive QA, then test
   instructional and factory-style policies with limited adaptation.
6. **Policy counterfactual:** hold the video state fixed, change only the
   standing request, and verify that the selected action changes appropriately.

The held-out-action result is the strongest test of a language-defined action
space. Zero-shot success is not guaranteed: the completed three-action training
does not provide enough evidence to expect it. Few-shot gains may be the more
realistic and still valuable result.

## Evaluation

### Selection metrics

- Macro-F1 and per-action precision/recall.
- Calibration/NLL over the available candidate set.
- Pairwise policy-consistency accuracy on identical video states.
- Paraphrase robustness.
- Seen-action, held-out-action, and few-shot adaptation accuracy.
- Confusion by label tier: direct, derived, and synthetic.

### Streaming behavior

- Judged PAUC-F for actions that communicate an answer or useful update.
- Useful intervention recall and intervention precision.
- Detection/notification latency.
- False alerts per video or hour.
- Duplicate notifications and repeated alerts.
- Correct use of `Hold` after an intervention.
- Request-evidence success: whether a later clearer view produces the correct
  downstream action.
- Correction success and time-to-repair after an injected wrong history.

### Compatibility with current downstream benchmarks

WEB/EGO and the existing PACE evaluation principally score when a response is
made and whether its text is useful. Preserve a benchmark-compatible mapping:

```text
Monitor, Hold, RequestEvidence without supported interaction -> no benchmark fire
Answer, Notify, Update, Correct, Alert, Guide                  -> benchmark fire
```

This mapping permits continuity reporting but does **not** validate the richer
action type. A model could select the wrong speech act and still receive answer
credit. Rich action claims therefore require a dedicated typed-action validation
set, ideally with human-reviewed direct or counterfactual labels.

Factory and instructional evaluation should additionally report policy-specific
costs. A false `Alert` is more serious than a missed routine `Notify`; a guessed
`Guide` can be worse than `RequestEvidence`. Do not collapse these outcomes into
accuracy alone.

## Recommended first scoped study

Avoid launching all nine actions immediately. Begin with three policies and six
shared/domain-specific actions:

```text
Common:          Monitor, Hold, RequestEvidence
Proactive QA:    Answer, Update
Instructional:   Guide, Correct
Factory:         Notify, Alert
```

The union contains nine actions, but each example exposes only five or six. The
first dataset can combine:

- direct PACE anchors for Monitor/Hold/Answer;
- derived PACE examples for Update;
- visibility counterfactuals for RequestEvidence;
- procedural subsets for Guide;
- history perturbations for Correct;
- policy-rewritten state changes for Notify/Alert.

Before large-scale training:

1. Hand-author a small action-definition guide with positive and negative
   examples.
2. Build a few hundred examples spanning all policy/action pairs.
3. Have humans audit an evaluation subset for action and payload correctness.
4. Verify tiny-set overfit and policy-counterfactual behavior.
5. Compare V0 against B1/V1 on held-out videos and paraphrases.
6. Scale only if variable candidates provide a measurable advantage.

## Failure modes and guardrails

- **Synthetic shortcuts:** transformations may reveal the label through wording,
  corruption style, or image quality rather than semantics.
- **Undefined urgency:** `Alert` versus `Notify` is a policy decision, not an
  inherent property of a frame. Record the user policy explicitly.
- **Future leakage:** evidence intervals may guide label creation, but future
  frames or gold responses must never enter the causal model state.
- **Action/content confounding:** evaluate the selected action separately from
  generated payload quality.
- **Candidate-order bias:** randomize or balance order during training and test
  sensitivity explicitly.
- **Prototype memorization:** disjoint train/test paraphrases and held-out
  compositions are mandatory.
- **Unsafe routing:** the learned controller proposes an action; deterministic
  policy still governs actual tools, escalation, and high-risk alerts.
- **Benchmark overclaiming:** current WEB/EGO scores validate response timing and
  content, not an open-vocabulary intervention ontology.

## Prior work and positioning

| Work | Relevant contribution | Relation to this direction |
| --- | --- | --- |
| [VideoLLM-online](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_VideoLLM-online_Online_Video_Large_Language_Model_for_Streaming_Video_CVPR_2024_paper.html) | Uses streaming EOS prediction to decide whether to produce language at a frame. | Establishes causal streaming response gating, but not a variable communicative-action space. |
| [VideoChat3 / OL617K](https://huggingface.co/datasets/MCG-NJU/VideoChat3-OL617k) | Converts localized clue intervals into `Silence`, `Standby`, and `Response` supervision. | Provides scalable causal timing anchors that can be transformed into policy-conditioned actions. |
| [MMDuet2](https://openreview.net/pdf?id=rxQnMSNCUs) | Uses multi-turn SFT/RL for timely proactive response without relying only on precise timestamp labels. | Motivates history-aware policy learning; released behavior remains primarily respond/silence. |
| [Proact-VL](https://openreview.net/attachment?id=k9PKgV0L4C&name=originally_submitted_PDF) | Applies a lightweight speaking classifier to a final-layer decision-token representation. | A close fixed-head baseline; richer CLM must show benefits beyond this design. |
| [CLM](https://github.com/Contrastive-LM/CLM) | Learns normalized state/action projections over a frozen language-model feature space. | Supplies the current action-scoring mechanism, but the completed fixed-three-action study did not show a reliable MLP advantage. |
| [JevAny](https://github.com/SimpleJev/JevAny) | Candidate-conditioned decision modeling with multimodal decoder adaptation. | Offers an alternative contextual option readout and is already covered by the separate JevAny experiment plan. |
| [SayCan](https://arxiv.org/abs/2204.01691) | Scores natural-language skills using semantic relevance plus grounded affordance. | Conceptual precedent for selecting language-described actions subject to environment feasibility. |

The proposed contribution is not merely “use an LLM embedding and projection
head.” It is:

> Learn a causal streaming policy over language-defined communicative actions,
> conditioned on user intent and interaction history, and demonstrate transfer
> across action descriptions, policy compositions, and application domains.

## Implementation notes for future agents

- Read `docs/design.md`, `docs/experiment_log.md`, and
  `agent_notes/experiments/status.md` before changing code.
- Preserve `configs/actions_v1.yaml` and completed outputs as the immutable
  three-action baseline.
- Create a new schema/config version for dynamic candidates and policy IDs.
- Extend datasets and losses to carry per-example candidate sets and candidate
  masks; do not assume one global action ordering.
- Keep PACE/Stream-Gemma causal input construction, response generation, and
  benchmark scoring as the source of truth.
- Store the standing request, exact action descriptions, action order, label
  provenance, and transformation version in every artifact.
- Keep controller and response-generator evaluation separable.
- Freeze a human-reviewed typed-action validation/test manifest before model
  selection.
- Do not inspect WEB/EGO to invent or tune the richer taxonomy; they have already
  been used in prior project evaluation.
- Treat zero-shot novel-action behavior as a hypothesis. Report failure rather
  than silently adding examples and continuing to call the action unseen.

## Decision criterion

Continue this direction only if a variable-action model demonstrates at least
one capability that a fixed classifier does not provide economically:

- robust selection under unseen action paraphrases;
- useful zero- or few-shot behavior on a held-out action;
- transfer to a new composition of known actions without retraining;
- reuse of one controller across proactive QA, instructional, and factory
  policies with competitive typed-action and streaming utility metrics.

If performance requires a separately trained fixed classifier for every policy
and offers no transfer advantage, the richer CLM formulation has not justified
its complexity.
