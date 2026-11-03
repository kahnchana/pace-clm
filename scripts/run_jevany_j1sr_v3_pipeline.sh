#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-j1sr-v3"
mkdir -p "$OUT/logs"
started=$(date -Is)
trap 'status=$?; printf "exit_code=%s\nfinished=%s\n" "$status" "$(date -Is)" > "$OUT/pipeline_status.txt"' EXIT
printf 'started=%s\n' "$started" > "$OUT/pipeline_status.txt"

"$ROOT/scripts/run_jevany_j1sr_v3_canary.sh"
"$ROOT/scripts/run_jevany_j1sr_v3_finish.sh"
