#!/usr/bin/env bash
#SBATCH --job-name=pace-jevany-j2-train
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-j2/logs/train-%j.out
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pace-jevany-j2-triton"
mkdir -p "$TRITON_CACHE_DIR"
resume_args=()
if [[ -f outputs/jevany-j2/full-seed42/latest.json ]]; then resume_args=(--resume); fi
exec "$ROOT/../stream-gemma/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/train_jevany_streaming.py \
  --mode J2 --stage full --seed 42 \
  --manifest "$ROOT/outputs/jevany/manifests/train.jsonl" \
  --output "$ROOT/outputs/jevany-j2/full-seed42" \
  --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 250 \
  "${resume_args[@]}"
