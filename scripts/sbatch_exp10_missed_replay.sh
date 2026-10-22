#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-missed-replay
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/missed-replay-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/freeze_exp10_missed_windows.py
"$PY" scripts/build_exp10_replay.py
mkdir -p "$OUT/preflight/e01" "$OUT/cache/e01"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/extract_feature_shards.py --split train \
  --train-manifest "$ROOT/manifests/exp10/e01_features.jsonl" \
  --output-dir "$OUT/preflight/e01" --max-videos-per-rank 1 --save-every-videos 1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/extract_feature_shards.py --split train \
  --train-manifest "$ROOT/manifests/exp10/e01_features.jsonl" \
  --output-dir "$OUT/cache/e01" --save-every-videos 10
"$PY" scripts/merge_feature_shards.py --input-dir "$OUT/cache/e01" \
  --split train --world-size 8 --output-name e01 \
  --manifest "$ROOT/manifests/exp10/e01_features.jsonl"
for run in E01 E02; do
  "$PY" scripts/train_exp10_missed.py --run "$run"
done
