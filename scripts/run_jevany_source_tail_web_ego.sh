#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1sr-v2/diagnostics/source-tail-replay"
CHECKPOINT="$OUT/full-seed42/step-008243"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/tmp/pace-jevany-source-tail-bench-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"
test -f "$CHECKPOINT/controller_adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/pointer_head.pt"

for benchmark in web ego; do
  dir="$OUT/evaluation/benchmarks/$benchmark"
  echo "Starting $benchmark at $(date -Is)" >> "$OUT/logs/benchmarks.log"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted >> "$OUT/logs/benchmarks.log" 2>&1
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8 \
    >> "$OUT/logs/benchmarks.log" 2>&1
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir" \
    >> "$OUT/logs/benchmarks.log" 2>&1
  echo "Completed $benchmark at $(date -Is)" >> "$OUT/logs/benchmarks.log"
done
date -Is > "$OUT/evaluation/benchmarks/completed"
