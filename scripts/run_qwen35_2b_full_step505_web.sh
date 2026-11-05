#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-qwen35-2b-full"
CHECKPOINT="$OUT/train/step-000505"
WEB="$OUT/evaluation/benchmarks/web_step505"

test "$(hostname)" = ip-10-3-120-167 || { echo 'wrong node' >&2; exit 2; }
test -f "$CHECKPOINT/controller_adapter/adapter_model.safetensors"
test -f "$CHECKPOINT/pointer_head.pt"
cd "$ROOT"
mkdir -p "$WEB" "$OUT/logs"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PACE_DISTRIBUTED_BACKEND=gloo
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-step505-web-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-step505-web-tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

echo "WEB rollout started $(date -u -Is) from $CHECKPOINT" >> "$OUT/logs/web_step505_eval.log"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" --benchmark web \
  --output-dir "$WEB" --response-margin 0.0 --max-new-tokens 96 \
  --answer-model promoted >> "$OUT/logs/web_step505_eval.log" 2>&1
"$PY" scripts/merge_jevany_benchmark.py --input-dir "$WEB" --world-size 8 \
  >> "$OUT/logs/web_step505_score.log" 2>&1
"$PY" scripts/score_exp10_web_ego.py web --pred-dir "$WEB" \
  >> "$OUT/logs/web_step505_score.log" 2>&1
echo "WEB evaluation completed $(date -u -Is)" >> "$OUT/logs/web_step505_score.log"
