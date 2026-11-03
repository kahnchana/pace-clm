#!/usr/bin/env bash
# Continue the locally started smoke run into the full WEB Gemini rollout.
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
CKPT="$ROOT/outputs/jevany-j1s/full-seed42/step-008243"
OUT="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/web-gemini38"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/tmp/pace-j1s-gemini38-local
export STREAM_GEMMA_GCLOUD_TOKEN_CACHE=/tmp/pace-j1s-gemini38-token-local.json
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"

# The smoke was launched separately on this node. Wait for its torchrun parent.
while kill -0 3341054 2>/dev/null; do sleep 10; done
"$PY" scripts/merge_jevany_benchmark.py --input-dir "$OUT/smoke8" --world-size 8 \
  >> "$OUT/logs/rollout.log" 2>&1

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_jevany_streaming.py --controller "$CKPT" \
  --benchmark web --output-dir "$OUT/full" \
  --response-margin 0.0 --answer-model gemini-3.8-flash \
  >> "$OUT/logs/rollout.log" 2>&1
"$PY" scripts/merge_jevany_benchmark.py --input-dir "$OUT/full" --world-size 8 \
  >> "$OUT/logs/rollout.log" 2>&1
date -Is > "$OUT/rollout-completed"
