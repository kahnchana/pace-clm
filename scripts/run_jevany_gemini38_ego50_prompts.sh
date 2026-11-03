#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50"
CKPT="$ROOT/outputs/jevany-j1s/full-seed42/step-008243"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export STREAM_GEMMA_GCLOUD_TOKEN_CACHE=/dev/shm/pace-j1s-g38-ego50-token.json
mkdir -p "$OUT/logs"
test -f "$CKPT/controller_adapter/adapter_model.safetensors"
test -f "$OUT/selected_ids.json"

variants=(event_boundary no_premature_absence stable_action_words current_event_only)
gpu_pairs=(0,1 2,3 4,5 6,7)
pids=()
for index in 0 1 2 3; do
  variant=${variants[$index]}
  pair=${gpu_pairs[$index]}
  (
    export CUDA_VISIBLE_DEVICES="$pair"
    export TRITON_CACHE_DIR="/dev/shm/pace-j1s-g38-ego50-${variant}-triton"
    export TMPDIR="/dev/shm/pace-j1s-g38-ego50-${variant}-tmp"
    mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/eval_jevany_streaming.py --controller "$CKPT" --benchmark ego \
      --selection-ids "$OUT/selected_ids.json" \
      --output-dir "$OUT/$variant" --response-margin 0.0 \
      --answer-model gemini-3.8-flash --gemini-prompt-variant "$variant"
    "$PY" scripts/merge_jevany_benchmark.py --input-dir "$OUT/$variant" --world-size 2
    date -Is > "$OUT/$variant/generation-completed"
  ) > "$OUT/logs/$variant.rollout.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more prompt rollouts failed; inspect per-variant logs" >&2
  exit 1
fi
date -Is > "$OUT/all-generation-completed"
