#!/usr/bin/env bash
# Run directly inside the existing eight-GPU allocation.
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/../stream-gemma/src"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-triton
mkdir -p "$TRITON_CACHE_DIR" "$ROOT/outputs/jevany/full-j1-seed42"
echo "Started $(date -Is) on $(hostname), parent Slurm allocation 36781"
echo "PACE commit $(git rev-parse HEAD), JevAny b273aea682e5b748a2605808e52ccafb7a378924"
resume_args=()
if [[ -f "$ROOT/outputs/jevany/full-j1-seed42/latest.json" ]]; then
  resume_args=(--resume)
fi
exec "$ROOT/../stream-gemma/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/train_jevany_streaming.py \
  --mode J1 --stage full --seed 42 \
  --manifest "$ROOT/outputs/jevany/manifests/train.jsonl" \
  --output "$ROOT/outputs/jevany/full-j1-seed42" \
  --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 250 \
  "${resume_args[@]}"
