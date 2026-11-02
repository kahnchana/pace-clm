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
mkdir -p "$TRITON_CACHE_DIR" "$OUT/reports"

wait_for_shards() {
  local directory="$1" expected="$2" deadline=$(( $(date +%s) + 72000 ))
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

score_timing() {
  local benchmark="$1" dir="$OUT/evaluation/benchmarks/$1"
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8 \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
  (cd "$SG" && "$PY" -m zs_infer.scoring web timing \
    --pred-dir "$dir" --gold "$SG/datasets/proactive-videoqa-$benchmark/anno.json" \
    --benchmark-name "ProactiveVideoQA ${benchmark^^}") \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
  (cd "$SG" && "$PY" -m zs_infer.pauc_f \
    --predictions "$dir/predictions.jsonl" \
    --scores "$dir/pauc_llm_scores-timing.json" \
    --output "$dir/pauc_f_summary-timing-v1.json" --omega 0.5 \
    --judge-model timing) >> "$OUT/logs/eval-$benchmark.log" 2>&1
}

score_judged() {
  local benchmark="$1" dir="$OUT/evaluation/benchmarks/$1"
  (cd "$SG" && "$PY" -m zs_infer.scoring web build \
    --pred-dir "$dir" --gold "$SG/datasets/proactive-videoqa-$benchmark/anno.json" \
    --judge-model gpt-4.1 --benchmark-name "ProactiveVideoQA ${benchmark^^}") \
    >> "$OUT/logs/judge-$benchmark.log" 2>&1
  if [[ -s "$dir/pauc_batch_input-gpt-4.1.jsonl" ]]; then
    "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir" \
      >> "$OUT/logs/judge-$benchmark.log" 2>&1
  else
    : > "$dir/pauc_batch_output-gpt-4.1.jsonl"
    (cd "$SG" && "$PY" -m zs_infer.scoring web process \
      --pred-dir "$dir" --gold "$SG/datasets/proactive-videoqa-$benchmark/anno.json" \
      --judge-model gpt-4.1) >> "$OUT/logs/judge-$benchmark.log" 2>&1
    (cd "$SG" && "$PY" -m zs_infer.pauc_f \
      --predictions "$dir/predictions.jsonl" \
      --scores "$dir/pauc_llm_scores-gpt-4.1.json" \
      --output "$dir/pauc_f_summary-gpt-4.1-v1.json" --omega 0.5 \
      --judge-model gpt-4.1) >> "$OUT/logs/judge-$benchmark.log" 2>&1
  fi
  echo "$benchmark judged at $(date -Is)" >> "$OUT/logs/benchmarks-status.log"
}

web_dir="$OUT/evaluation/benchmarks/web"
ego_dir="$OUT/evaluation/benchmarks/ego"
wait_for_shards "$web_dir" 500
score_timing web
echo "WEB timing complete at $(date -Is)" >> "$OUT/logs/benchmarks-status.log"
score_judged web
if "$PY" scripts/check_jevany_j1sr_v2_web_gate.py \
    >> "$OUT/logs/benchmarks-status.log" 2>&1; then
  echo "WEB gate passed; proceeding to EGO" >> "$OUT/logs/benchmarks-status.log"
else
  code=$?
  if [[ "$code" -eq 3 ]]; then
    echo "WEB gate matched validation collapse; EGO paused at $(date -Is)" \
      >> "$OUT/logs/benchmarks-status.log"
    date -Is > "$OUT/evaluation/ego_paused"
    exit 0
  fi
  exit "$code"
fi

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
  --benchmark ego --output-dir "$ego_dir" --response-margin 0.0 \
  --answer-model promoted >> "$OUT/logs/eval-ego.log" 2>&1
wait_for_shards "$ego_dir" 326
score_timing ego
echo "EGO timing complete at $(date -Is)" >> "$OUT/logs/benchmarks-status.log"

"$PY" scripts/report_jevany_j1sr_v2_timing.py \
  >> "$OUT/logs/benchmarks-status.log" 2>&1
date -Is > "$OUT/evaluation/timing_completed"
score_judged ego
"$PY" scripts/report_jevany_j1sr_v2.py >> "$OUT/logs/report.log" 2>&1
date -Is > "$OUT/evaluation/completed"
