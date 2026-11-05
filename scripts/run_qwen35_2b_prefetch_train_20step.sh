#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
DATA="$ROOT/outputs/jevany-qwen35-2b-full"
OUT="$ROOT/outputs/jevany-qwen35-2b-prefetch-20step"
PY="$ROOT/../stream-gemma/.venv/bin/python"
test "$(hostname)" = ip-10-3-70-177 || { echo 'wrong node' >&2; exit 2; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPUs are in use; the 20-step test requires all eight GPUs free' >&2
  exit 2
fi
if [[ -e "$OUT/train/latest.json" ]]; then
  echo 'test output already contains a checkpoint; refusing to overwrite it' >&2
  exit 2
fi
cd "$ROOT"
mkdir -p "$OUT/train" "$OUT/logs"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_jevany_2b_batched.py \
  --manifest "$DATA/manifests/train_length_sorted.jsonl" \
  --batch-plan "$DATA/manifests/batch_plan.jsonl" \
  --output "$OUT/train" --frame-cache 1024 --conv-batch-chunk 128 \
  --microbatch-ratio 0.75 --backend gloo \
  --save-every 20 --stop-after-steps 20 --log-every 1 \
  > "$OUT/logs/train.log" 2>&1 &
training_pid=$!
(
  while kill -0 "$training_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) > "$OUT/logs/gpu_samples.csv" 2>/dev/null &
sampler_pid=$!
if ! wait "$training_pid"; then
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  exit 1
fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
