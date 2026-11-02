#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1sr-v2"
CHECKPOINT="$OUT/full-seed42/step-008243"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-j1sr-v2-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"

wait_for_shards() {
  local directory="$1" expected="$2" deadline=$(( $(date +%s) + 14400 ))
  local count shards shard
  while true; do
    count=0
    shards=0
    for shard in "$directory"/predictions.rank*.jsonl; do
      if [[ -f "$shard" ]]; then
        shards=$((shards + 1))
        count=$((count + $(wc -l < "$shard")))
      fi
    done
    if [[ "$shards" -eq 8 && "$count" -eq "$expected" ]]; then
      sleep 30
      return 0
    fi
    if [[ $(date +%s) -ge "$deadline" ]]; then
      echo "Timed out waiting for $directory: $count/$expected rows" >&2
      return 1
    fi
    sleep 60
  done
}

wait_for_file() {
  local target="$1" deadline=$(( $(date +%s) + 14400 ))
  until [[ -f "$target" ]]; do
    if [[ $(date +%s) -ge "$deadline" ]]; then
      echo "Timed out waiting for $target" >&2
      return 1
    fi
    sleep 60
  done
}

wait_for_shards "$OUT/evaluation/validation/j1sr" 200
"$PY" scripts/prepare_exp10_scoring.py \
  --rollout-dir "$OUT/evaluation/validation/j1sr" \
  --manifest "$OUT/validation/frozen.jsonl" \
  --gold "$OUT/validation/frozen.gold.json" --world-size 8 \
  >> "$OUT/logs/eval-j1sr.log" 2>&1
"$PY" scripts/judge_exp10_rollout.py \
  --rollout-dir "$OUT/evaluation/validation/j1sr" \
  >> "$OUT/logs/eval-j1sr.log" 2>&1
wait_for_file "$OUT/evaluation/validation/j1s/official/results/exp10_judged_summary.json"

for benchmark in web ego; do
  dir="$OUT/evaluation/benchmarks/$benchmark"
  echo "Starting $benchmark at $(date -Is)" >> "$OUT/logs/eval-$benchmark.log"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted >> "$OUT/logs/eval-$benchmark.log" 2>&1
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8 \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir" \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
done
"$PY" scripts/report_jevany_j1sr_v2.py >> "$OUT/logs/report.log" 2>&1
date -Is > "$OUT/evaluation/completed"
