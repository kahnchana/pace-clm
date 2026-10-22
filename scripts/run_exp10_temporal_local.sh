#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-exp10"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/logs/temporal-local.exit"' EXIT
cd "$ROOT"
bash scripts/sbatch_exp10_temporal.sh
"$PY" scripts/merge_exp10_temporal.py
