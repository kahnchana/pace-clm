#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-confirmation
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/confirmation-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10/confirmation"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mapfile -t finalists < <("$PY" -c 'import json; print("\n".join(json.load(open("manifests/exp10/selection_decision.json"))["finalists"]))')
if (( ${#finalists[@]} == 0 )); then
  "$PY" scripts/confirm_exp10_finalists.py
  exit 0
fi

for name in run5 PACE "${finalists[@]}"; do
  if [[ "$name" == run5 ]]; then
    POLICY=(--checkpoint "$ROOT/outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt")
  elif [[ "$name" == PACE ]]; then
    POLICY=(--token-baseline)
  else
    POLICY=(--checkpoint "$ROOT/outputs/pace-clm-exp10/$name/checkpoint.pt")
  fi
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_exp10_rollout.py \
    --manifest "$ROOT/manifests/exp10/confirmation.jsonl" \
    "${POLICY[@]}" --output-dir "$OUT/$name"
  "$PY" scripts/prepare_exp10_scoring.py \
    --rollout-dir "$OUT/$name" \
    --manifest "$ROOT/manifests/exp10/confirmation.jsonl" \
    --gold "$ROOT/manifests/exp10/confirmation.gold.json"
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$OUT/$name"
done
"$PY" scripts/confirm_exp10_finalists.py
