#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
[[ "$(hostname)" == ip-10-3-120-167 ]]
export LD_LIBRARY_PATH=
export PATH="/fsx/home/kranasinghe/google-cloud-sdk/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export STREAM_GEMMA_GCLOUD_TOKEN_CACHE=/opt/dlami/nvme/kranasinghe/pace-clm/node-cache/flash-token.json
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
exec bash scripts/with_jevany_nvme_cache.sh "$ROOT/../stream-gemma/.venv/bin/python" -u scripts/run_jevany_flash_parallel.py "$@"
