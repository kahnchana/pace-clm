#!/usr/bin/env bash
#SBATCH --job-name=pace-jevany-j2-eval
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-j2/logs/eval-%j.out
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j2/evaluation/benchmarks"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pace-jevany-j2-triton"
mkdir -p "$TRITON_CACHE_DIR"
test -f outputs/jevany-j2/full-seed42/summary.rank0.json
checkpoint=$("$PY" -c 'import json; print(json.load(open("outputs/jevany-j2/full-seed42/latest.json"))["checkpoint"])')
test -f "$checkpoint/controller_adapter/adapter_model.safetensors"
for benchmark in web ego; do
  dir="$OUT/$benchmark"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir"
done
