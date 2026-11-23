# Jevany checkpoint archives and node caches

Storage maintenance authorized 2026-11-15. Evaluations and automatic retry
loops were stopped at the user's request. Do not restart them as part of
storage maintenance.

Older checkpoint directories are archived under
`/fsx/sfr/data/kanchana/pace-clm/checkpoints/outputs/`, preserving their original
relative paths. Every file is SHA-256 verified before its original directory is
replaced by a symlink. Existing checkpoint commit manifests remain unchanged.
The original base SFT checkpoint
`outputs/jevany-qwen35-2b-j1s/full-seed42/step-008243` and the final checkpoint
`outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128/step-000128`
stay in their original directories and are independently hashed before/after.
All other checkpoint contents are retained, including optimizer and RNG state.

Both authorized nodes have local instance NVMe mounted at `/opt/dlami/nvme`.
Use `/opt/dlami/nvme/kranasinghe/pace-clm/node-cache` for disposable compilation
and temporary caches. The approximately 100 GB root filesystems must not hold
growing experiment kernel caches. Original `/tmp/pace-…triton…` cache paths are
redirected to NVMe for compatibility with frozen launch scripts. Current ESTP
kernel caches are preserved on NVMe; obsolete compiled caches are discarded.

For new commands, use:

```bash
scripts/with_jevany_nvme_cache.sh <command> [arguments ...]
```

The wrapper sets TMPDIR, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR,
CUDA_CACHE_PATH, TORCH_EXTENSIONS_DIR, MPLCONFIGDIR and NUMBA_CACHE_DIR. It
does not change Hugging Face model-cache locations, pretrained weights, model
scores, evaluation protocol, or training recipes. Local NVMe is disposable;
checkpoints, predictions, judgments, logs and audit records remain on FSx.

Inventory, protected-checkpoint hashes, individual copied-file manifests,
move journal, per-node cleanup records and the completion report are under
`outputs/storage-maintenance/20261115/`.
