#!/usr/bin/env bash
set -euo pipefail
SIZE=${1:?expected 0p8b or 4b}
case "$SIZE" in
  0p8b) MODEL=official-0p8b; MODEL_NAME=Qwen/Qwen3.5-0.8B ;;
  4b) MODEL=official-4b; MODEL_NAME=Qwen/Qwen3.5-4B ;;
  *) echo "unsupported size: $SIZE" >&2; exit 2 ;;
esac
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-qwen35-$SIZE-j1s"
TRAIN="$OUT/full-seed42"
WEB="$OUT/evaluation/benchmarks/web"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PACE_DISTRIBUTED_BACKEND=gloo
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-size-sweep-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs" "$OUT/reports"
exec 9>"$OUT/pipeline.lock"
flock 9
if [[ -f "$OUT/completed" ]]; then exit 0; fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 > "$OUT/logs/gpu_samples.csv" 2> "$OUT/logs/gpu_samples.err" &
gpu_sampler=$!
trap 'kill "$gpu_sampler" 2>/dev/null || true' EXIT
printf '%s  %s\n' 89bcd02cb2668750a16a2d4e89d35badd41c3b40e87699648b80ea3308622222 \
  "$ROOT/outputs/jevany-j1s/manifests/train.jsonl" | sha256sum --check
if [[ ! -f "$TRAIN/summary.rank0.json" ]]; then
  mkdir -p "$TRAIN"
  if [[ ! -f "$TRAIN/started_epoch.txt" ]]; then date +%s > "$TRAIN/started_epoch.txt"; fi
  for attempt in $(seq 1 10); do
    resume_args=()
    if [[ -f "$TRAIN/latest.json" ]]; then resume_args=(--resume); fi
    echo "Training attempt $attempt at $(date -Is)" >> "$OUT/logs/train.log"
    if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_streaming.py --mode J1 --stage full \
      --controller-model "$MODEL" --seed 42 \
      --manifest "$ROOT/outputs/jevany-j1s/manifests/train.jsonl" \
      --output "$TRAIN" --head-lr 0.0001 --lora-lr 0.000002 \
      --accumulation 4 --save-every 100 "${resume_args[@]}" \
      >> "$OUT/logs/train.log" 2>&1; then break; fi
    sleep 30
  done
fi
"$PY" - "$TRAIN" "$MODEL_NAME" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); s = json.loads((p/'summary.rank0.json').read_text())
assert (s['world'], s['optimizer_steps'], s['global_presentations']) == (8, 8243, 263760), s
latest = json.loads((p/'latest.json').read_text())
assert latest['step'] == 8243, latest
identity = json.loads((Path(latest['checkpoint'])/'model_identity.json').read_text())
assert identity['model_name'] == sys.argv[2], identity
PY
if [[ ! -f "$TRAIN/completed_epoch.txt" ]]; then date +%s > "$TRAIN/completed_epoch.txt"; fi
checkpoint=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$TRAIN/latest.json")
for attempt in $(seq 1 10); do
  echo "WEB evaluation attempt $attempt at $(date -Is)" >> "$OUT/logs/web_eval.log"
  if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" --benchmark web \
    --output-dir "$WEB" --response-margin 0.0 --max-new-tokens 96 \
    --answer-model promoted >> "$OUT/logs/web_eval.log" 2>&1; then break; fi
  sleep 30
done
"$PY" scripts/merge_jevany_benchmark.py --input-dir "$WEB" --world-size 8 \
  >> "$OUT/logs/web_score.log" 2>&1
"$PY" scripts/score_exp10_web_ego.py web --pred-dir "$WEB" \
  >> "$OUT/logs/web_score.log" 2>&1
"$PY" scripts/report_qwen35_dense_j1s_web.py --output-root "$OUT" \
  >> "$OUT/logs/web_score.log" 2>&1
date -Is > "$OUT/completed"
