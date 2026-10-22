#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/cache_exp10_teacher.py --save-every-videos 10
