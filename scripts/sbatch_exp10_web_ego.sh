#!/usr/bin/env bash
#SBATCH --job-name=pace-exp10-web-ego
#SBATCH --account=team-agent
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/pace-clm-exp10/logs/web-ego-%j.out

set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10/full"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mapfile -t advanced < <("$PY" -c 'import json; print("\n".join(json.load(open("manifests/exp10/confirmation_decision.json"))["advanced_to_web_ego"]))')
if (( ${#advanced[@]} == 0 )); then
  echo "No candidate passed confirmation; WEB/EGO candidate evaluation correctly skipped."
  "$PY" scripts/report_exp10.py
  exit 0
fi

for benchmark in web ego; do
  for name in PACE "${advanced[@]}"; do
    if [[ "$name" == PACE ]]; then
      POLICY=(--token-baseline)
    else
      POLICY=(--checkpoint "$ROOT/outputs/pace-clm-exp10/$name/checkpoint.pt")
    fi
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/eval_web_ego_full.py "$benchmark" \
      "${POLICY[@]}" --output-dir "$OUT/$benchmark/$name"
    "$PY" scripts/merge_web_ego_full.py "$OUT/$benchmark/$name"
    "$PY" scripts/score_exp10_web_ego.py "$benchmark" \
      --pred-dir "$OUT/$benchmark/$name"
  done
done
"$PY" scripts/bootstrap_exp10_web_ego.py
"$PY" scripts/report_exp10.py
