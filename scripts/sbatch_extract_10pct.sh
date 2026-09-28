#!/usr/bin/env bash
#SBATCH --job-name=pace-clm-10pct-features
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/logs/extract-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct
PYTHON=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/features" "$OUT/logs"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/extract_feature_shards.py --split train \
    --train-manifest "$ROOT/manifests/pace_train_10pct_decisions.jsonl" \
    --output-dir "$OUT/features"
"$PYTHON" scripts/merge_feature_shards.py --input-dir "$OUT/features" --split train --world-size 8
cp "$ROOT/outputs/features/v1/pace200.pt" "$OUT/features/pace200.pt"
cp "$ROOT/outputs/features/v1/action_variants.pt" "$OUT/features/action_variants.pt"
