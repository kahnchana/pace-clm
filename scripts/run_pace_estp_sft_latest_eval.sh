#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
[[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]]
export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
exec bash scripts/with_jevany_nvme_cache.sh "$ROOT/../stream-gemma/.venv/bin/python" -u scripts/eval_pace_estp_sft_three_suites.py "$@"
