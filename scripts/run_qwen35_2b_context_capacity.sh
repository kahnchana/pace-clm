#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-qwen35-2b-util"
PY="$ROOT/../stream-gemma/.venv/bin/python"
test "$(hostname)" = ip-10-3-120-167 || { echo 'wrong node' >&2; exit 2; }
case "${1:-}" in
  short) batch=272; manifest=context_1_4_b272_stress.jsonl ;;
  medium) batch=160; manifest=context_5_8_b160_stress.jsonl ;;
  longish) batch=96; manifest=context_9_16_b96_stress.jsonl ;;
  midlong) batch=64; manifest=context_17_24_b64_stress.jsonl ;;
  *) echo 'usage: run_qwen35_2b_context_capacity.sh {short|medium|longish|midlong}' >&2; exit 2 ;;
esac
name="nccl_${1}_b${batch}_capacity"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/benchmark_qwen35_2b_training.py \
  --manifest "$OUT/manifests/$manifest" --output "$OUT/reports/$name.json" \
  --backend nccl --microbatch "$batch" --frame-cache 1024 \
  --conv-batch-chunk 128 --warmup 0 --updates 1 \
  --throughput-only > "$OUT/logs/$name.log" 2>&1 &
training_pid=$!
(
  while kill -0 "$training_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "$OUT/logs/${name}_gpu_samples.csv" 2>/dev/null &
sampler_pid=$!
if ! wait "$training_pid"; then
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  exit 1
fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
