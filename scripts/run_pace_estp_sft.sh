#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export LD_LIBRARY_PATH=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
PY="$ROOT/../stream-gemma/.venv/bin/python"
RUN="$ROOT/outputs/jevany-qwen35-2b-pace-estp-sft/base-50-50-seed42-16h"
case "${1:-}" in
 prepare) exec bash scripts/with_jevany_nvme_cache.sh "$PY" -u scripts/prepare_pace_estp_sft.py --run "$RUN" ;;
 initialize) exec bash scripts/with_jevany_nvme_cache.sh "$PY" -u scripts/run_pace_estp_sft.py initialize --run "$RUN" ;;
 train)
  [[ "$(hostname)" == ip-10-3-70-177 && "${SLURM_JOB_ID:-}" == 38226 ]]
  exec bash scripts/with_jevany_nvme_cache.sh "$PY" -u scripts/run_pace_estp_sft.py train --run "$RUN" ;;
 eval)
  [[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]]
  exec bash scripts/with_jevany_nvme_cache.sh "$PY" -u scripts/run_pace_estp_sft.py eval --run "$RUN" ;;
 *) echo 'usage: run_pace_estp_sft.sh prepare|initialize|train|eval' >&2;exit 2 ;;
esac
