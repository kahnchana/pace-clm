#!/usr/bin/env bash
set -euo pipefail
ROOT=${PACE_CONTROL_CODE_ROOT:-/fsx/home/kranasinghe/repo/pace-clm}
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-control-triton
export TMPDIR=/dev/shm/pace-qwen35-control-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
PY="$ROOT/../stream-gemma/.venv/bin/python"
case "${1:-}" in
  train)
    shift
    exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_2b_batched.py --backend nccl --microbatch 16 \
      --require-global-batch 128 --microbatch-ratio 0.5 --frame-cache 1024 \
      --conv-batch-chunk 128 --predecode-workers 4 --prefetch-mode prepare \
      --save-every 250 --log-every 1 "$@"
    ;;
  python)
    shift
    exec "$PY" "$@"
    ;;
  *) echo 'usage: run_qwen35_control_command.sh train|python [arguments]' >&2; exit 2 ;;
esac
