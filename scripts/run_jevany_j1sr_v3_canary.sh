#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1sr-v3"
MANIFEST="$OUT/manifests/train.jsonl"
CHECKPOINT="$OUT/full-seed42/step-001000"
ROLL="$OUT/evaluation/validation/canary64-step-001000"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/tmp/pace-jevany-j1sr-v3-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"

"$PY" - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/fsx/home/kranasinghe/repo/pace-clm')
manifest = root / 'outputs/jevany-j1sr-v3/manifests/train.jsonl'
audit = json.loads((root / 'outputs/jevany-j1sr-v3/manifests/train.audit.json').read_text())
h = hashlib.sha256(manifest.read_bytes()).hexdigest()
if h != audit['interleaved_manifest_sha256']:
    raise ValueError('interleaved manifest hash differs')
if audit['mined_last_640'] != 3:
    raise ValueError('unexpected tail placement')
PY

if [[ ! -f "$CHECKPOINT/training_state.pt" ]]; then
  for attempt in $(seq 1 5); do
    resume_args=()
    if [[ -f "$OUT/full-seed42/latest.json" ]]; then resume_args=(--resume); fi
    echo "Canary training attempt $attempt at $(date -Is)" >> "$OUT/logs/canary-train.log"
    if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_streaming.py --mode J1 --stage full --seed 42 \
      --manifest "$MANIFEST" --output "$OUT/full-seed42" \
      --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 100 \
      --stop-after-steps 1000 "${resume_args[@]}" >> "$OUT/logs/canary-train.log" 2>&1; then
      break
    fi
    sleep 30
  done
fi
test -f "$CHECKPOINT/training_state.pt"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
  --validation-manifest "$OUT/validation/canary64.jsonl" --output-dir "$ROLL" \
  --response-margin 0.0 --answer-model promoted --record-action-logits \
  >> "$OUT/logs/canary-eval.log" 2>&1
"$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$ROLL" \
  --manifest "$OUT/validation/canary64.jsonl" \
  --gold "$OUT/validation/canary64.gold.json" --world-size 8 --skip-judge \
  >> "$OUT/logs/canary-eval.log" 2>&1
"$PY" scripts/check_jevany_canary_gate.py \
  --gate "$OUT/validation/canary64_gate.json" \
  --timing "$ROLL/official/results/timing_prejudge.json" \
  --output "$OUT/validation/canary64_gate_result.json" \
  | tee -a "$OUT/logs/canary-eval.log"
