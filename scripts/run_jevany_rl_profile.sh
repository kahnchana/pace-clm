#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
OUTPUT=$1
shift
bash scripts/run_jevany_rl_gate.sh --output "$OUTPUT" "$@"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
"$ROOT/../stream-gemma/.venv/bin/python" -u scripts/verify_jevany_rl_resume.py --run "$OUTPUT"
