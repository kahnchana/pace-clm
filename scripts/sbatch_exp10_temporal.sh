#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-temporal
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/temporal-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10/cache/temporal"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/freeze_exp10_temporal.py
mkdir -p "$OUT" "$ROOT/outputs/pace-clm-exp10/cache/temporal-smoke"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/extract_feature_shards.py --split train \
  --train-manifest "$ROOT/manifests/exp10/temporal_features.jsonl" \
  --output-dir "$ROOT/outputs/pace-clm-exp10/cache/temporal-smoke" \
  --max-videos-per-rank 1 --save-every-videos 1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/extract_feature_shards.py --split train \
  --train-manifest "$ROOT/manifests/exp10/temporal_features.jsonl" \
  --output-dir "$OUT" --save-every-videos 10
"$PY" scripts/merge_feature_shards.py --input-dir "$OUT" --split train \
  --world-size 8 --output-name temporal --manifest "$ROOT/manifests/exp10/temporal_features.jsonl"
