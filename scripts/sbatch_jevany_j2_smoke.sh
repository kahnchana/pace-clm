#!/usr/bin/env bash
#SBATCH --job-name=pace-jevany-j2-smoke
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-j2/logs/smoke-%j.out
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pace-jevany-j2-triton"
mkdir -p "$TRITON_CACHE_DIR"
exec "$ROOT/../stream-gemma/.venv/bin/python" scripts/smoke_jevany_streaming.py \
  --manifest "$ROOT/outputs/jevany/manifests/train.jsonl" \
  --output "$ROOT/outputs/jevany-j2/smoke/report.json"
