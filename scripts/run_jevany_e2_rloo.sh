#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
exec "$ROOT/../stream-gemma/.venv/bin/python" -u scripts/run_jevany_e2_rloo.py "$@"
