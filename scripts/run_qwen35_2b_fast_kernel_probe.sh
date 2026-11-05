#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-qwen35-2b-util"
PY="$ROOT/../stream-gemma/.venv/bin/python"
KERNELS=/tmp/pace-qwen-kernels/site-packages
test "$(hostname)" = ip-10-3-70-177 || { echo 'wrong node' >&2; exit 2; }
test -d "$KERNELS/causal_conv1d" || { echo 'isolated kernel wheel missing' >&2; exit 2; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPUs are in use; short probes require all eight GPUs free' >&2
  exit 2
fi
cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/reports"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$KERNELS:$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name=gloo_b48_prefetch_prepare_fast
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/benchmark_qwen35_2b_training.py \
  --manifest "$OUT/manifests/normal_b48_length_sorted.jsonl" \
  --output "$OUT/reports/${name}.json" \
  --backend gloo --microbatch 48 --frame-cache 1024 \
  --predecode-workers 4 --prefetch-prepare \
  --warmup 5 --updates 15 --throughput-only \
  > "$OUT/logs/${name}.log" 2>&1 &
benchmark_pid=$!
(
  while kill -0 "$benchmark_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "$OUT/logs/${name}_gpu_samples.csv" 2>/dev/null &
sampler_pid=$!
if ! wait "$benchmark_pid"; then
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  exit 1
fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
