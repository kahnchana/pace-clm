#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-references
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/selection-references-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10/selection"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

for reference in run5 PACE; do
  if [[ "$reference" == run5 ]]; then
    POLICY=(--checkpoint "$ROOT/outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt")
  else
    POLICY=(--token-baseline)
  fi
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_exp10_rollout.py \
    --manifest "$ROOT/manifests/exp10/selection.jsonl" \
    "${POLICY[@]}" --output-dir "$OUT/$reference"
  "$PY" scripts/prepare_exp10_scoring.py \
    --rollout-dir "$OUT/$reference" \
    --manifest "$ROOT/manifests/exp10/selection.jsonl" \
    --gold "$ROOT/manifests/exp10/selection.gold.json"
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$OUT/$reference"
done
