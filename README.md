# pace-clm

Proof-of-concept CLM-style state/action projection heads over the frozen,
promoted PACE-Qwen streaming model.

The repository is intentionally separate from `stream-gemma`. Stream-Gemma
remains the source of truth for PACE model loading, causal streaming inputs,
state construction, benchmark adapters, and PAUC-F scoring. This package adds a
small, tested integration boundary and does not fork benchmark behavior.

## Pinned MVP contract

- Base: `Qwen/Qwen3.5-35B-A3B` at revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`
- PACE adapter: `../stream-gemma/outputs/lora-qwen-w1-long-neg`
- Inference: v24/v25 streaming, 1 FPS, 32 frames, 144 vision tokens/frame
- Memory: `last-response`, instruction `v2`, `k=1`
- Qwen generation prefix: `direct`
- Paper-facing state map: `Standby -> Hold`

See [docs/design.md](docs/design.md) for evidence and source locations.

## Current status

The repository foundation is complete:

- Initialized as a standalone Apache-2.0 project.
- Audited and pinned the promoted PACE-Qwen base model, revision, adapter, and
  inference protocol.
- Recorded the promoted adapter SHA-256 in the resolved MVP configuration.
- Added versioned action descriptions and an explicit `Standby -> Hold` label
  boundary.
- Added a narrow Stream-Gemma integration layer for source-contract validation
  and strictly causal message construction.
- Added frozen-backbone feature extraction below the vocabulary LM head.
- Added causally safe final-language-token pooling.
- Added independent normalized state/action projection heads with a learned
  logit scale.
- Added provenance helpers and guarded entry points for extraction, training,
  PACE-200 evaluation, and the frozen WEB/EGO pilot.
- Added tests for source-contract validation, causal inputs, label mapping,
  pooling position, backbone freezing, LM-head bypass, and head gradients.

The initial contract suite contains eight passing tests. It uses synthetic
models and validates the local promoted Stream-Gemma artifact without loading
the 35B model.

Real-model feature extraction, tiny-set overfitting, MVP training, baselines,
PACE-200 development evaluation, and the frozen WEB/EGO pilot are intentionally
still pending. Their required order and completion criteria are tracked in
[agent_notes/todo.md](agent_notes/todo.md).

## Repository map

| Path | Purpose |
| --- | --- |
| [`configs/`](configs/) | Versioned model, streaming, projection, label, and action contracts |
| [`src/pace_clm/`](src/pace_clm/) | Feature extraction, pooling, heads, labels, and Stream-Gemma boundary |
| [`tests/`](tests/) | CPU contract and gradient tests |
| [`scripts/`](scripts/) | Guarded execution entry points |
| [`docs/design.md`](docs/design.md) | Source audit, design decisions, and attribution |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Chronological verified work log |
| [`agent_notes/overview.md`](agent_notes/overview.md) | Project status and documentation index |
| [`agent_notes/todo.md`](agent_notes/todo.md) | Full implementation handoff and execution order |
| [`manifests/`](manifests/) | Future deterministic train/dev/test manifests |

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

The unit tests use synthetic models and do not allocate a GPU. A real 35B
feature-extraction smoke test is deliberately a separate, explicit step.
