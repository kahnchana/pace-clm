#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/tmp/pace-rl-triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ROOT/../stream-gemma/.venv/bin/python"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'RL profiling requires an unused node' >&2
  exit 2
fi
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/probe_qwen35_nccl.py > "$ROOT/outputs/jevany-2b-rl/nccl-${SLURM_JOB_ID:-local}.json"
exec "$PY" -u scripts/run_jevany_rl.py "$@"
