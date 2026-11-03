#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-j1sr-v3"
GATE="$OUT/validation/canary64_gate_result.json"
deadline=$(( $(date +%s) + 43200 ))
trap 'status=$?; printf "exit_code=%s\nfinished=%s\n" "$status" "$(date -Is)" > "$OUT/finish_status.txt"' EXIT
printf 'waiting_since=%s\n' "$(date -Is)" > "$OUT/finish_status.txt"

until [[ -f "$GATE" ]]; do
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "Timed out waiting for canary gate" >&2
    exit 1
  fi
  sleep 30
done

"$ROOT/scripts/run_jevany_j1sr_v3_finish.sh"
