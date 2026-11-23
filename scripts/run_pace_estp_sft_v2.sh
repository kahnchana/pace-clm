#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
PY="$ROOT/../stream-gemma/.venv/bin/python"
RUN="$ROOT/outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-b384-prefetch-v2-16h"
case "${1:-}" in
 train) [[ "$(hostname)" == ip-10-3-70-177 && "${SLURM_JOB_ID:-}" == 38226 ]] ;;
 eval) [[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]] ;;
esac
exec bash scripts/with_jevany_nvme_cache.sh "$PY" -u scripts/run_pace_estp_sft_v2.py "${1:?mode required}" --run "$RUN"
