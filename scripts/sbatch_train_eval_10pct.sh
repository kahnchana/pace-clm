#!/usr/bin/env bash
#SBATCH --job-name=pace-clm-10pct-train
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/logs/train-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct
PYTHON=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
"$PYTHON" scripts/train_10pct_curve.py --feature-dir "$OUT/features" --output-dir "$OUT/results"
"$PYTHON" scripts/eval_10pct_frozen.py --feature-dir "$OUT/features" --result-dir "$OUT/results"
"$PYTHON" scripts/benchmark_serving_10pct.py --feature-dir "$OUT/features" --result-dir "$OUT/results"
"$PYTHON" scripts/finalize_10pct.py --result-dir "$OUT/results"
"$PYTHON" scripts/write_10pct_provenance.py --output-root "$OUT"
