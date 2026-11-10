#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
OUTPUT=$1
shift
PY="$ROOT/../stream-gemma/.venv/bin/python"
"$PY" scripts/jevany_rl_kernel_cache.py stage --audit "$OUTPUT/kernel-cache.jsonl"
cleanup() {
  local result=$?
  trap - EXIT
  "$PY" scripts/jevany_rl_kernel_cache.py save --audit "$OUTPUT/kernel-cache.jsonl" || true
  exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
bash scripts/run_jevany_rl_profile.sh "$OUTPUT" "$@"
