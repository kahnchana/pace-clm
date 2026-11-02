# Generator feedback in the J1 streaming controller

Date: 2026-10-25

Status: observed WEB result and proposed controller design; no new controller trained

## What the WEB rerun showed

J1 uses a Qwen decoder with a trained LoRA and pointer head to choose `Silence`,
`Hold`, or `Response`. On `Response`, a separate answer path writes text. The
original J1 evaluation used the promoted PACE Qwen `long-neg` adapter for that
text. The [base-answer WEB rerun](../../outputs/jevany/evaluation/benchmarks/web-base-answer-native/manifest.json)
kept the J1 checkpoint and decision settings, but disabled LoRA during answer
generation and used the unadapted Qwen base model. It needed a plain-language
answer prompt with Qwen's native generation suffix: the PACE forced
`</Response>` prefix sometimes produced an empty base-model answer. This prompt
change is part of the intervention, so the score difference cannot be assigned
to generator weights alone.

| Full WEB, 500 videos | Promoted PACE | J1 + promoted answers | J1 + base answers |
| --- | ---: | ---: | ---: |
| GPT-4.1 judged PAUC-F | 57.47 | 55.41 | 43.31 |
| Physical fires | 1,132 | 1,240 | 1,773 |
| Value-adding / wasted fires | 800 / 332 | 812 / 428 | 773 / 1,000 |
| Judged fire precision | 70.67% | 65.48% | 43.60% |

Sources: [promoted PACE record](../../../stream-gemma/agent_notes/docs/experiments/qwen_w1_long_neg.md),
[original J1 score](../../outputs/jevany/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json),
and [base-answer score](../../outputs/jevany/evaluation/benchmarks/web-base-answer-native/pauc_f_summary-gpt-4.1-v1.json).
The base-answer judge completed 1,039/1,039 requests with no reported errors.
Promoted PACE's raw WEB predictions are unavailable locally, so its comparison
uses reported full-split point estimates, without a paired interval.

Replacing answers changed the later action trajectory: 204 of 500 videos had a
different fire count between the two J1 rollouts. In
`eaU9xQkP_jE.26.mp4`, both controllers first fired at 1 second. With a
promoted PACE answer, the next fire was at 9 seconds and the video had three
fires. With a base-model answer, the next fire was at 2 seconds and the video
had ten fires. These are [saved predictions](../../outputs/jevany/evaluation/benchmarks/web-base-answer-native/predictions.jsonl),
not a per-frame logit attribution. This short clip had no aged-out response
memory; its prior answer was still present as an assistant turn in the causal
window.

## Why answer text affects decisions

The [rollout loop](../../../stream-gemma/src/stream_gemma/inference/qwen_stream.py)
adds each emitted answer to later assistant history. After it leaves the
32-frame window, the `last-response` memory path can also put its text next to
the question. The [J1 driver](../../scripts/eval_jevany_streaming.py) then
runs the controller on that self-generated context at every frame. Therefore a
different answer can produce different later action logits even though the
controller weights, video, and question are fixed. J1 chooses an action via
pointer logits; it does not generate the answer tokens itself. The two J1
rollouts have identical decision inputs until their first generated answer,
assuming deterministic execution.

This is a closed-loop test of a *controller and generator pair*. A fixed fire
schedule with different answers would isolate answer quality, but would not
test how the deployed system behaves after its own answers enter history.

## What generator independence should mean

For a controller intended to work with Qwen, Gemini, or another answer
provider, make its **decision inputs independent of provider-written text**.
It should still use the video, question, timestamps, and its own prior actions:
the last fire time and an event or coverage state are needed to avoid repeated
responses. That state can be derived by the controller from video evidence and
its own actions, rather than from the wording of a provider's answer. The
answer generator receives the current causal evidence and returns text, but
its text does not feed back into the portable controller.

Under a deterministic policy with identical video, question, and controller
state, swapping generators would then leave the action schedule unchanged by
construction. This is a useful invariance check. It also has a limit: a
response-blind controller cannot know whether a generator refused, erred, or
omitted an important detail. If adapting to answer quality is needed, add a
separate, provider-neutral structured feedback signal and measure its effect
explicitly; that relaxes schedule invariance.

Generator independence is narrower than complete model independence. **J1's
controller still requires the pinned Qwen multimodal backbone and its LoRA.**
It could call Gemini as a black-box *answer generator* while Qwen makes
decisions, but it is not a controller that runs on Gemini's hidden states. A
controller that can be deployed without Qwen needs its own fixed visual/text
encoder or a standard feature interface, plus a separate action head and
provider-specific answer adapters.

## Closed-model generators and next experiment

A closed model does not need to be fine-tuned to serve as the answer generator.
At a controller fire, an API call can send the allowed causal frames and
question and return answer text. Its latency, price, failures, and context
limits then become part of the system evaluation. Closed-model rollouts on
training or development videos can also provide examples or feedback for
training a **separate** controller, without gradients through the API. Keep
WEB/EGO as evaluation data; do not tune the controller or threshold from their
judged results.

Recommended test sequence:

1. On training/development videos, replay one frozen fire schedule with
   promoted Qwen, base Qwen, and any closed generator. This isolates answer
   quality and provider cost at common decision times.
2. Build a response-blind controller input that retains prior action times and
   controller-owned coverage state. Run a provider-swap check: action times
   should match exactly on the same videos when answer text is hidden.
3. Evaluate each controller–generator pair in closed loop on held-out
   validation, then freeze choices before WEB/EGO. Report judged PAUC-F,
   useful and wasted fires, repeated events, answer latency, and API cost.

The present WEB result argues against treating J1 as generator agnostic. It
does not show that a separate response-blind controller would fail.
