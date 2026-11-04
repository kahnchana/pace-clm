#!/usr/bin/env bash
set -euo pipefail
cd /fsx/home/kranasinghe/repo/pace-clm
bash scripts/run_qwen35_dense_j1s_web.sh 0p8b &
small_pid=$!
train_complete=outputs/jevany-qwen35-0p8b-j1s/full-seed42/completed_epoch.txt
while [[ ! -f "$train_complete" ]]; do
  if ! kill -0 "$small_pid" 2>/dev/null; then
    wait "$small_pid"
    echo "0.8B run exited before training completed" >&2
    exit 1
  fi
  sleep 30
done
bash scripts/run_qwen35_dense_j1s_web.sh 4b &
large_pid=$!
wait "$small_pid"
wait "$large_pid"
export PYTHONPATH="$PWD/../stream-gemma:$PWD/../stream-gemma/src:$PWD/scripts"
../stream-gemma/.venv/bin/python scripts/report_qwen35_j1s_size_sweep.py
date -Is > outputs/jevany-qwen35-size-sweep-j1s/completed
