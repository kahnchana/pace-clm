#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/web-gemini38"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$SG/src:$SG"

while true; do
  count=$("$PY" - "$OUT/full" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(sum(sum(1 for _ in p.open()) for p in root.glob('predictions.rank*.jsonl')))
PY
  )
  if [[ "$count" -eq 500 ]]; then break; fi
  if [[ "$count" -gt 500 ]]; then echo "More than 500 predictions: $count" >&2; exit 1; fi
  sleep 20
done
"$PY" scripts/merge_jevany_benchmark.py --input-dir "$OUT/full" --world-size 8 \
  >> "$OUT/logs/rollout.log" 2>&1
date -Is > "$OUT/rollout-completed"
