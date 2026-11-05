#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
DATA="$ROOT/outputs/jevany-qwen35-2b-full"
PY="$ROOT/../stream-gemma/.venv/bin/python"
test "$(hostname)" = ip-10-3-120-167 || { echo 'wrong node' >&2; exit 2; }
cd "$ROOT"
mkdir -p "$DATA/train" "$DATA/logs"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_jevany_2b_batched.py \
  --manifest "$DATA/manifests/train_length_sorted.jsonl" \
  --batch-plan "$DATA/manifests/batch_plan.jsonl" \
  --output "$DATA/train" --frame-cache 1024 --conv-batch-chunk 128 \
  --predecode-workers 4 --prefetch-mode prepare \
  --save-every 500 --stop-after-steps 5 \
  > "$DATA/logs/train_gate.log" 2>&1 &
training_pid=$!
(
  while kill -0 "$training_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "$DATA/logs/train_gate_gpu_samples.csv" 2>/dev/null &
sampler_pid=$!
if ! wait "$training_pid"; then
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  exit 1
fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
