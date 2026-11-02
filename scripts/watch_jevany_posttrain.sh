#!/usr/bin/env bash
# Relaunch resumable WEB/EGO evaluation if its tmux worker exits early.
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
REPORT="$ROOT/outputs/jevany/reports/full-results.json"
LOG="$ROOT/outputs/jevany/logs/posttrain.log"
attempts=0
while [[ ! -f "$REPORT" ]]; do
  if ! tmux has-session -t jevany-posttrain 2>/dev/null; then
    attempts=$((attempts + 1))
    if (( attempts > 3 )); then
      echo "Posttrain evaluation exited repeatedly; inspect $LOG" >&2
      exit 1
    fi
    echo "Restarting resumable posttrain evaluation, attempt $attempts at $(date -Is)" >> "$LOG"
    tmux new-session -d -s jevany-posttrain \
      "cd '$ROOT' && bash scripts/run_jevany_posttrain.sh >> '$LOG' 2>&1"
  fi
  sleep 120
done
echo "Final result report exists: $REPORT"
