#!/usr/bin/env bash
# Judge WEB as soon as its full eight-GPU rollout is merged, while EGO runs.
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
DIR="$ROOT/outputs/jevany/evaluation/benchmarks/web"
cd "$ROOT"
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
while [[ ! -f "$DIR/predictions.jsonl" ]]; do
  sleep 60
done
"$PY" scripts/score_exp10_web_ego.py web --pred-dir "$DIR"
echo "WEB GPT-4.1 scoring completed at $(date -Is)"
