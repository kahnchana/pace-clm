#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
BASE="$ROOT/outputs/jevany-qwen35-2b-mixture-12h"
OUT="$BASE/control-seed42"
INITIAL="$BASE/shared/initial-seed42/step-000000"
COMMAND="$ROOT/scripts/run_qwen35_control_command.sh"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits
    sleep 1
  done
) >> "$OUT/logs/preflight_gpu.csv" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
if [[ ! -f "$OUT/preflight/continuous-attempt1/summary.rank0.json" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/smoke.jsonl" \
    --output "$OUT/preflight/continuous-attempt1" --initialize-from "$INITIAL" \
    --stop-after-steps 2 > "$OUT/logs/preflight_continuous-attempt1.log" 2>&1
fi
if [[ ! -f "$OUT/preflight/continuous/summary.rank0.json" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/smoke.jsonl" \
    --output "$OUT/preflight/continuous" --initialize-from "$INITIAL" \
    --stop-after-steps 2 --save-every 1 > "$OUT/logs/preflight_continuous.log" 2>&1
fi
if [[ ! -f "$OUT/preflight/resumed/step-000001/training_state.pt" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/smoke.jsonl" \
    --output "$OUT/preflight/resumed" --initialize-from "$INITIAL" \
    --stop-after-steps 1 > "$OUT/logs/preflight_resumed.log" 2>&1
fi
if [[ ! -f "$OUT/preflight/resumed/summary.rank0.json" ]] || [[ ! -f "$OUT/preflight/resumed/step-000002/training_state.pt" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/smoke.jsonl" \
    --output "$OUT/preflight/resumed" --resume \
    --stop-after-steps 2 >> "$OUT/logs/preflight_resumed.log" 2>&1
fi
if [[ ! -f "$OUT/preflight/restore/verified.rank7.json" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/smoke.jsonl" \
    --output "$OUT/preflight/restore" --initialize-from "$OUT/preflight/resumed/step-000001" \
    --verify-restore-only > "$OUT/logs/preflight_restore.log" 2>&1
fi
while [[ ! -f "$BASE/shared/manifests/audit.json" ]]; do sleep 5; done
if [[ ! -f "$OUT/preflight/representative/summary.rank0.json" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/manifests/probe.jsonl" \
    --output "$OUT/preflight/representative" --initialize-from "$INITIAL" \
    --stop-after-steps 24 > "$OUT/logs/preflight_representative.log" 2>&1
fi
if [[ ! -f "$OUT/preflight/stress/summary.rank0.json" ]]; then
  bash "$COMMAND" train --manifest "$BASE/shared/manifests/stress.jsonl" \
    --output "$OUT/preflight/stress" --initialize-from "$INITIAL" \
    --stop-after-steps 1 > "$OUT/logs/preflight_stress.log" 2>&1
fi
bash "$COMMAND" python scripts/finalize_qwen35_control_preflight.py \
  > "$OUT/logs/preflight_finalize.log" 2>&1
