#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-heads
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/remaining-heads-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/merge_exp10_teacher.py
"$PY" scripts/merge_exp10_temporal.py

run_pair() {
  local script="$1" first="$2" second="$3"
  mkdir -p "$OUT/$first" "$OUT/$second"
  CUDA_VISIBLE_DEVICES=0 "$PY" "scripts/$script" --run "$first" > "$OUT/$first/train.log" 2>&1 &
  local first_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" "scripts/$script" --run "$second" > "$OUT/$second/train.log" 2>&1 &
  local second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
}

run_pair train_exp10_distill.py E03 E04
run_pair train_exp10_replay.py E05 E06
run_pair train_exp10_temporal.py E07 E08
run_pair train_exp10_utility.py E09 E10
