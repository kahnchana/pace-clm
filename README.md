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

The PACE-200 CLM pilot is complete through timing-only PAUC-F:

- Initialized as a standalone Apache-2.0 project.
- Audited and pinned the promoted PACE-Qwen base model, revision, adapter, and
  inference protocol.
- Verified real-model final-token extraction, hidden width 2048, frozen
  backbone/adapter parameters, LM-head bypass, and cached-action parity.
- Built deterministic disjoint manifests and reusable features for a balanced
  768-example PACE-train subset and the frozen 200-video validation set.
- Passed the 24-example tiny-set overfit gate at 100% accuracy.
- Trained and evaluated raw cosine, linear, capacity-matched MLP, CLM
  128/256/512-d, shared-head, and short-label variants over three seeds.
- Evaluated the existing PACE token-generation decision baseline on the same
  10,658 causal streaming decisions.
- Scored seed-42 predictions through the unchanged Stream-Gemma timing PAUC-F
  implementation and saved per-example predictions, checkpoints, metrics,
  confusion matrices, manifests, and provenance.
- Exercised the original PACE response-generation handoff on eight
  deterministic CLM-selected fires; all eight returned non-empty,
  query-relevant text for qualitative inspection.

The existing PACE token-generation decision remains strongest (83.70 macro-F1,
80.10 timing PAUC-F). The best CLM timing score is 65.14 PAUC-F (256-d); the
best CLM Response F1 across seeds is 60.49 ± 1.75 (short labels). See
[`agent_notes/experiments/pace200_clm_pilot.md`](agent_notes/experiments/pace200_clm_pilot.md)
for the concise result table. Semantic judged PAUC-F is intentionally out of
scope for this timing-only pilot; no WEB/EGO pilot has been run.

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
| [`manifests/`](manifests/) | Deterministic train/dev/test manifests and selection metadata |

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

The unit tests use synthetic models and do not allocate a GPU. Real-model
extraction and evaluation artifacts live under the ignored `outputs/` tree.
