# Project Docs Overview

Agent note-keeping for `pace-clm`. Read this first when resuming work; prefer
these checked-in notes over chat history.

## How to use

- Keep verified architecture and source-contract decisions in
  [`../docs/design.md`](../docs/design.md).
- Update [`../docs/experiment_log.md`](../docs/experiment_log.md) only when a
  check, experiment, or decision is complete.
- Keep the implementation sequence and acceptance criteria in
  [`todo.md`](todo.md) accurate as milestones land.
- Store deterministic data selections under [`../manifests/`](../manifests/).
- Do not run or inspect WEB/EGO pilot results until the PACE-200 design and
  hyperparameters are frozen.

## Doc map

| Doc | Purpose |
| --- | --- |
| [`todo.md`](todo.md) | Full MVP handoff, execution order, evaluation protocol, and completion criteria |
| [`../README.md`](../README.md) | User-facing project status, pinned contract, and repository map |
| [`../docs/design.md`](../docs/design.md) | Promoted PACE-Qwen audit, integration boundary, label semantics, feature contract, and attribution |
| [`../docs/experiment_log.md`](../docs/experiment_log.md) | Chronological record of completed and verified work |
| [`../configs/qwen_mvp.yaml`](../configs/qwen_mvp.yaml) | Pinned model, adapter hash, streaming protocol, projection architecture, and seed |
| [`../configs/actions_v1.yaml`](../configs/actions_v1.yaml) | Exact versioned action descriptions used for action features |
| [`../manifests/README.md`](../manifests/README.md) | Manifest policy and WEB/EGO pilot guardrail |

## Project goal

Evaluate whether CLM-style contrastive state/action alignment improves
streaming `Response`, `Silence`, and `Hold` decisions over ordinary classifiers
when every method uses the same frozen promoted PACE-Qwen state feature and the
unchanged Stream-Gemma evaluation harness.

The frozen PACE VLM must retain its existing response-generation behavior. Only
the state projection head, action projection head, and scalar logit scale are
trainable in the CLM-style method.

## Pinned promoted model

| Field | Value |
| --- | --- |
| Base | `Qwen/Qwen3.5-35B-A3B` |
| Revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| PACE adapter | `../stream-gemma/outputs/lora-qwen-w1-long-neg` |
| Protocol | v24/v25, 1 FPS, causal 32-frame rolling window |
| Memory | last-response, instruction v2, k=1, aged-out selection |
| Prefix | direct |
| Paper label map | `Silence -> Silence`, `Standby -> Hold`, `Response -> Response` |

## Current status

The PACE-200 pilot, 10% learning curve, data-mixture study, PACE-exposure head
scale-up, and fresh gate follow-through are complete. The latest independent
200-video timing-only cohort found the frozen run 5 MLP at 83.38 mean PAUC-F
across three seeds versus 82.08 for PACE token generation, with fewer fires.
The CLM head did not show a repeatable advantage over the MLP, and longer
repeated exposure did not help. The newly dev-selected balanced-MLP margin
reduced final fires but lowered PAUC-F.

Full WEB/EGO transfer and GPT-4.1 judging are now complete. The frozen run 5
MLP scored 53.50 WEB and 43.95 EGO judged PAUC-F, below the promoted PACE
Qwen token decision's 57.47 and 46.05. It fired less often but lost judged
recall. The CLM-style head has not demonstrated an advantage over the ordinary
MLP, and this MLP has not demonstrated a WEB/EGO advantage over PACE. See
[`experiments/pace_clm_gate_followthrough.md`](experiments/pace_clm_gate_followthrough.md)
and [`experiments/pace_clm_web_ego_full.md`](experiments/pace_clm_web_ego_full.md).
Reproducible checkpoints, manifests, predictions, judge outputs, scorer
outputs, and provenance are under `outputs/`.
