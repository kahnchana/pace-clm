#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-qwen35-2b-util"
PY="$ROOT/../stream-gemma/.venv/bin/python"
test "$(hostname)" = ip-10-3-120-167 || { echo 'wrong node' >&2; exit 2; }
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/benchmark_qwen35_2b_training.py \
  --manifest "$OUT/manifests/normal_b48_length_sorted.jsonl" \
  --output "$OUT/reports/nccl_b48_length_sorted_predecode4.json" \
  --backend nccl --microbatch 48 --predecode-workers 4 --warmup 5 --updates 15 \
  --throughput-only > "$OUT/logs/nccl_b48_length_sorted_predecode4.log" 2>&1 &
training_pid=$!
(
  while kill -0 "$training_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "$OUT/logs/nccl_b48_length_sorted_predecode4_gpu_samples.csv" 2>/dev/null &
sampler_pid=$!
if ! wait "$training_pid"; then
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  exit 1
fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
