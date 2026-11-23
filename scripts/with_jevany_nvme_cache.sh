#!/usr/bin/env bash
set -euo pipefail
PACE_NODE_CACHE=/opt/dlami/nvme/kranasinghe/pace-clm/node-cache
mountpoint -q /opt/dlami/nvme
mkdir -p "$PACE_NODE_CACHE"/{tmp,triton,torchinductor,cuda,torch-extensions,matplotlib,numba}
export TMPDIR="$PACE_NODE_CACHE/tmp"
export TRITON_CACHE_DIR="$PACE_NODE_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$PACE_NODE_CACHE/torchinductor"
export CUDA_CACHE_PATH="$PACE_NODE_CACHE/cuda"
export TORCH_EXTENSIONS_DIR="$PACE_NODE_CACHE/torch-extensions"
export MPLCONFIGDIR="$PACE_NODE_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$PACE_NODE_CACHE/numba"
# Keep the existing Hugging Face cache configuration and model weights intact.
exec "$@"
