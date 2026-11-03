#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$SG/src:$SG"
mkdir -p "$OUT/logs"

while [[ ! -f "$OUT/all-generation-completed" ]]; do sleep 30; done
for variant in event_boundary no_premature_absence stable_action_words current_event_only; do
  "$PY" scripts/score_exp10_web_ego.py ego --pred-dir "$OUT/$variant" \
    --expected-videos 50 >> "$OUT/logs/$variant.score.log" 2>&1
  date -Is > "$OUT/$variant/scoring-completed"
done
date -Is > "$OUT/all-judging-completed"
"$PY" scripts/report_jevany_gemini38_ego50_prompts.py
