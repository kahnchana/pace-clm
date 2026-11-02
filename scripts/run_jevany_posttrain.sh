#!/usr/bin/env bash
# Evaluate the frozen seed-42 controller after the full eight-GPU pass finishes.
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
TRAIN="$ROOT/outputs/jevany/full-j1-seed42"
OUT="$ROOT/outputs/jevany/evaluation"
cd "$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-triton

restarts=0
while [[ ! -f "$TRAIN/summary.rank0.json" ]]; do
  # torchrun's supervisor can exit while its eight workers keep training.
  # Count the workers themselves so a watcher never starts a second model
  # on GPUs that are still in use.
  workers=$("$PY" scripts/count_jevany_workers.py)
  if (( workers > 0 && workers != 8 )); then
    echo "Partial training worker set ($workers/8); refusing competing restart" >&2
    exit 1
  fi
  if (( workers == 0 )); then
    sleep 20
    workers=$("$PY" scripts/count_jevany_workers.py)
    if [[ ! -f "$TRAIN/summary.rank0.json" ]] && (( workers == 0 )); then
      restarts=$((restarts + 1))
      if (( restarts > 3 )); then
        echo "Training exited four times without completing; inspect the recovery log" >&2
        exit 1
      fi
      echo "Training stopped before completion; recovery attempt $restarts at $(date -Is)" >&2
      bash scripts/run_jevany_full_current_node.sh \
        >> "$ROOT/outputs/jevany/logs/full-recovery.log" 2>&1
    fi
  fi
  sleep 60
done
checkpoint=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$TRAIN/latest.json")
sleep 60  # Let the training workers release GPU memory after writing the summary.
if [[ ! -f "$checkpoint/controller_adapter/adapter_model.safetensors" ]]; then
  echo "Incomplete final controller checkpoint: $checkpoint" >&2
  exit 1
fi

for benchmark in web ego; do
  dir="$OUT/benchmarks/$benchmark"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --benchmark "$benchmark" --output-dir "$dir"
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8
done

for benchmark in web ego; do
  dir="$OUT/benchmarks/$benchmark"
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir"
done

for split in selection final; do
  manifest="$ROOT/outputs/pace-clm-mixtures/validation/$split.jsonl"
  gold="$ROOT/outputs/pace-clm-mixtures/validation/$split.gold.json"
  dir="$OUT/validation/$split"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --validation-manifest "$manifest" --output-dir "$dir"
  "$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$dir" \
    --manifest "$manifest" --gold "$gold" --world-size 8
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$dir"
done
"$PY" scripts/summarize_jevany_results.py
echo "JevAny seed-42 validation and WEB/EGO evaluation completed at $(date -Is)"
