#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-train-rollout
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/train-rollout-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/freeze_exp10_training_rollout.py
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_exp10_rollout.py \
  --manifest "$ROOT/manifests/exp10/training_rollout.jsonl" \
  --checkpoint "$ROOT/outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt" \
  --capture-features-manifest "$ROOT/outputs/pace-clm-mixtures/manifests/run5.jsonl" \
  --output-dir "$OUT/train_rollout_smoke" --max-videos 8
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_exp10_rollout.py \
  --manifest "$ROOT/manifests/exp10/training_rollout.jsonl" \
  --checkpoint "$ROOT/outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt" \
  --capture-features-manifest "$ROOT/outputs/pace-clm-mixtures/manifests/run5.jsonl" \
  --output-dir "$OUT/train_rollout"
"$PY" scripts/prepare_exp10_scoring.py \
  --rollout-dir "$OUT/train_rollout" \
  --manifest "$ROOT/manifests/exp10/training_rollout.jsonl" \
  --gold "$ROOT/manifests/exp10/training_rollout.gold.json" \
  --skip-judge
