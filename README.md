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

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

The unit tests use synthetic models and do not allocate a GPU. A real 35B
feature-extraction smoke test is deliberately a separate, explicit step.

