#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
[[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]]
export LD_LIBRARY_PATH= OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0 STREAM_GEMMA_VISION_FEATURES=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
exec bash scripts/with_jevany_nvme_cache.sh "$ROOT/../stream-gemma/.venv/bin/python" -u scripts/profile_pace_sft_fast_eval.py
