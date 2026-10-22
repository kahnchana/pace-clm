#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-utility-labels
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/utility-labels-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/freeze_exp10_utility_candidates.py
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/cache_exp10_utility.py --output-dir "$OUT/preflight/utility" \
  --max-videos-per-rank 1 --save-every-videos 1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/cache_exp10_utility.py --output-dir "$OUT/cache/utility" \
  --save-every-videos 10
"$PY" scripts/prepare_exp10_utility_judge.py
"$PY" scripts/judge_exp10_utility.py
