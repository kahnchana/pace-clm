#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-selection
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/selection-runs-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10/selection"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

for run in E01 E02 E03 E04 E05 E06 E07 E08 E09 E10; do
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_exp10_rollout.py \
    --manifest "$ROOT/manifests/exp10/selection.jsonl" \
    --checkpoint "$ROOT/outputs/pace-clm-exp10/$run/checkpoint.pt" \
    --output-dir "$OUT/$run"
  "$PY" scripts/prepare_exp10_scoring.py \
    --rollout-dir "$OUT/$run" \
    --manifest "$ROOT/manifests/exp10/selection.jsonl" \
    --gold "$ROOT/manifests/exp10/selection.gold.json"
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$OUT/$run"
  "$PY" scripts/mark_exp10_completion.py --run "$run"
done
"$PY" scripts/select_exp10_finalists.py
