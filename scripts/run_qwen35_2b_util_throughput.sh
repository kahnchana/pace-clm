#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-qwen35-2b-util"
PY="$ROOT/../stream-gemma/.venv/bin/python"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR" "$OUT/logs" "$OUT/reports"
for batch in 1 2 4; do
  echo "$(date -Is) throughput-only nccl batch=$batch"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/benchmark_qwen35_2b_training.py \
    --manifest "$OUT/manifests/normal.jsonl" \
    --output "$OUT/reports/nccl_b${batch}_throughput.json" \
    --backend nccl --microbatch "$batch" --warmup 5 --updates 15 \
    --throughput-only > "$OUT/logs/nccl_b${batch}_throughput.log" 2>&1 &
  training_pid=$!
  (
    while kill -0 "$training_pid" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "$OUT/logs/nccl_b${batch}_gpu_samples.csv" 2>/dev/null &
  sampler_pid=$!
  if ! wait "$training_pid"; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    exit 1
  fi
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
done
