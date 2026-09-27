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

PACE-200 development is complete through unchanged-harness timing PAUC-F:

- real-model extraction checks passed on the pinned promoted PACE-Qwen model;
- deterministic, video-disjoint train and frozen PACE-200 manifests are saved;
- reusable features were extracted and the tiny-overfit gate passed at 100%;
- all five required methods and the feasible width/shared-head/short-label
  variants were trained and evaluated, with three seeds for trained heads;
- the existing PACE token baseline remains strongest at 83.70 macro-F1 and
  80.10 timing PAUC-F; best CLM timing PAUC-F is 65.14 (256-d);
- reproducible checkpoints, per-example predictions, metrics, scorer outputs,
  and provenance are under `outputs/`.

Semantic judged PAUC-F is intentionally out of scope for this timing-only
pilot. Do not start WEB/EGO; the PACE-200 evidence does not justify advancing
the CLM method unchanged.
