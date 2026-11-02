#!/usr/bin/env bash
#SBATCH --job-name=pace-jevany-j1s
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-j1s/logs/full-%j.out
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1s"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-j1s-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"
test -f "$OUT/manifests/train.summary.json"
if [[ ! -f "$OUT/full-seed42/summary.rank0.json" ]]; then
  for attempt in $(seq 1 10); do
    resume_args=()
    if [[ -f "$OUT/full-seed42/latest.json" ]]; then resume_args=(--resume); fi
    echo "Training attempt $attempt at $(date -Is)" >> "$OUT/logs/train.log"
    if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_streaming.py --mode J1 --stage full --seed 42 \
      --manifest "$OUT/manifests/train.jsonl" --output "$OUT/full-seed42" \
      --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 100 \
      "${resume_args[@]}" >> "$OUT/logs/train.log" 2>&1; then
      break
    fi
    echo "Training attempt $attempt failed at $(date -Is)" >> "$OUT/logs/train.log"
    sleep 30
  done
fi
test -f "$OUT/full-seed42/summary.rank0.json"
checkpoint=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/full-seed42/latest.json")
test -f "$checkpoint/controller_adapter/adapter_model.safetensors"
for split in selection final; do
  manifest="$ROOT/outputs/pace-clm-mixtures/validation/$split.jsonl"
  gold="$ROOT/outputs/pace-clm-mixtures/validation/$split.gold.json"
  dir="$OUT/evaluation/validation/$split"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --validation-manifest "$manifest" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted
  "$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$dir" \
    --manifest "$manifest" --gold "$gold" --world-size 8
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$dir"
done
for benchmark in web ego; do
  dir="$OUT/evaluation/benchmarks/$benchmark"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir"
done
date -Is > "$OUT/evaluation/completed"
