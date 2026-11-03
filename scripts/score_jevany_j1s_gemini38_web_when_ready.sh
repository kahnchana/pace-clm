#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/web-gemini38"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$SG/src:$SG"

while [[ ! -f "$OUT/rollout-completed" ]]; do sleep 20; done
"$PY" - "$OUT/full/predictions.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.open() if line.strip()]
if len(rows) != 500 or len({row['question_id'] for row in rows}) != 500:
    raise SystemExit(f'incomplete or duplicate WEB predictions: {len(rows)} rows')
PY
"$PY" scripts/score_exp10_web_ego.py web --pred-dir "$OUT/full" \
  >> "$OUT/logs/score.log" 2>&1
date -Is > "$OUT/scoring-completed"
