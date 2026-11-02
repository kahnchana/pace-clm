#!/usr/bin/env bash
#SBATCH --job-name=pace-jevany-j1
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany/logs/full-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/../stream-gemma/src"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pace-jevany-triton"
mkdir -p "$TRITON_CACHE_DIR" "$ROOT/outputs/jevany/full-j1-seed42"
echo "JevAny reference b273aea682e5b748a2605808e52ccafb7a378924"
echo "PACE commit $(git rev-parse HEAD) job ${SLURM_JOB_ID:-none}"
resume_args=()
if [[ -f "$ROOT/outputs/jevany/full-j1-seed42/latest.json" ]]; then
  resume_args=(--resume)
fi
exec "$ROOT/../stream-gemma/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/train_jevany_streaming.py \
  --mode J1 --stage full --seed 42 \
  --manifest "$ROOT/outputs/jevany/manifests/train.jsonl" \
  --output "$ROOT/outputs/jevany/full-j1-seed42" \
  --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 \
  "${resume_args[@]}"
