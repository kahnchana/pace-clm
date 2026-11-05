#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-qwen35-2b-full"

test "$(hostname)" = ip-10-3-120-167 || { echo 'wrong node' >&2; exit 2; }
(( $# > 0 )) || { echo 'expected checkpoint steps, e.g. 10 25' >&2; exit 2; }
gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
eval_ranks=${EVAL_RANKS:-8}
[[ "$eval_ranks" =~ ^[1-8]$ ]] || { echo "invalid EVAL_RANKS: $eval_ranks" >&2; exit 2; }
[[ $(awk -F, '{print NF}' <<< "$gpu_ids") == "$eval_ranks" ]] || {
  echo "GPU_IDS count must equal EVAL_RANKS" >&2; exit 2;
}
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PACE_DISTRIBUTED_BACKEND=gloo
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
mkdir -p "$OUT/logs"

for step in "$@"; do
  [[ "$step" =~ ^[0-9]+$ ]] || { echo "invalid step: $step" >&2; exit 2; }
  checkpoint="$OUT/train/step-$(printf '%06d' "$step")"
  web="$OUT/evaluation/benchmarks/web_step${step}"
  eval_log="$OUT/logs/web_step${step}_eval.log"
  score_log="$OUT/logs/web_step${step}_score.log"
  export TRITON_CACHE_DIR="/dev/shm/pace-qwen35-2b-web-step${step}-triton"
  export TMPDIR="/dev/shm/pace-qwen35-2b-web-step${step}-tmp"
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
  test -f "$checkpoint/controller_adapter/adapter_model.safetensors"
  test -f "$checkpoint/pointer_head.pt"
  test -f "$checkpoint/training_state.pt"
  mkdir -p "$web"
  echo "WEB rollout started $(date -u -Is) from $checkpoint on GPU_IDS=$gpu_ids" >> "$eval_log"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$eval_ranks" \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" --benchmark web \
    --output-dir "$web" --response-margin 0.0 --max-new-tokens 96 \
    --answer-model promoted >> "$eval_log" 2>&1
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$web" --world-size "$eval_ranks" \
    >> "$score_log" 2>&1
  "$PY" scripts/score_exp10_web_ego.py web --pred-dir "$web" \
    >> "$score_log" 2>&1
  echo "WEB evaluation completed $(date -u -Is)" >> "$score_log"
done
