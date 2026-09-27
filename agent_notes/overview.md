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

Repository foundation is complete at initial commit `cfa1a92`:

- standalone Apache-2.0 repository and package scaffold;
- promoted checkpoint/configuration audit and validation;
- versioned MVP/action configurations;
- Stream-Gemma integration boundary and causal message builder;
- frozen backbone feature-extraction and pooling primitives;
- independent projection heads and learned temperature;
- eight passing CPU contract tests;
- clean Ruff lint and formatting checks.

No H200 job has been launched yet. The tests avoid loading the 35B model.

## Next execution milestone

Run a small real-model feature-extraction smoke test through the promoted
PACE-Qwen adapter. It must confirm:

1. the selected final language position follows the exact direct-prefix chat
   rendering used by the promoted runtime;
2. multimodal and text-only action extraction return the configured hidden
   width and never invoke the vocabulary head;
3. all VLM and PACE adapter parameters remain frozen;
4. cached and uncached action features are numerically identical;
5. strictly causal frame access holds on hand-inspected episodes.

After that, build the deterministic balanced PACE training subset and require a
tiny-set overfit before starting the full MVP head training.

