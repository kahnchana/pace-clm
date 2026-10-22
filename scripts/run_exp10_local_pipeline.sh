#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/pace-clm-exp10"
cd "$ROOT"
mkdir -p "$OUT/logs"

while [[ ! -f "$OUT/logs/temporal-local.exit" ]]; do
  sleep 30
done
if [[ "$(cat "$OUT/logs/temporal-local.exit")" != 0 ]]; then
  echo "Temporal cache failed; stopping before any head training." >&2
  exit 1
fi

run_stage() {
  local name="$1" script="$2"
  if [[ -f "$OUT/logs/$name.exit" && "$(cat "$OUT/logs/$name.exit")" == 0 ]]; then
    echo "[$(date -u +%FT%TZ)] already completed $name" | tee -a "$OUT/logs/pipeline.log"
    return
  fi
  echo "[$(date -u +%FT%TZ)] starting $name" | tee -a "$OUT/logs/pipeline.log"
  if /bin/bash "scripts/$script" >> "$OUT/logs/$name.log" 2>&1; then
    echo 0 > "$OUT/logs/$name.exit"
    echo "[$(date -u +%FT%TZ)] completed $name" | tee -a "$OUT/logs/pipeline.log"
  else
    local status=$?
    echo "$status" > "$OUT/logs/$name.exit"
    echo "[$(date -u +%FT%TZ)] failed $name (exit $status)" | tee -a "$OUT/logs/pipeline.log"
    exit "$status"
  fi
}

run_stage train-rollout sbatch_exp10_training_rollout.sh
run_stage utility-labels sbatch_exp10_utility_labels.sh
run_stage missed-replay sbatch_exp10_missed_replay.sh
run_stage remaining-heads sbatch_exp10_remaining_heads.sh
run_stage selection-references sbatch_exp10_selection_references.sh
run_stage selection-runs sbatch_exp10_selection_runs.sh
run_stage confirmation sbatch_exp10_confirmation.sh
run_stage web-ego sbatch_exp10_web_ego.sh
