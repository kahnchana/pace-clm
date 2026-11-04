#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
PY="$ROOT/../stream-gemma/.venv/bin/python"
OUT="$ROOT/outputs/jevany-qwen35-2b-util"
cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/reports" "$OUT/manifests"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-qwen35-2b-util-triton
export TMPDIR=/dev/shm/pace-qwen35-2b-util-tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

"$PY" - "$OUT/manifests" <<'PY'
import json, sys
from pathlib import Path
source = Path('outputs/jevany-j1s/manifests/train.jsonl')
out = Path(sys.argv[1])
normal = []
full = []
with source.open() as handle:
    for i, line in enumerate(handle):
        if 120000 <= i < 120640:
            normal.append(line)
        if len(full) < 32 and len(json.loads(line)['turns']) == 32:
            full.append(line)
if len(normal) != 640 or len(full) != 32:
    raise ValueError('benchmark manifests are incomplete')
(out / 'normal.jsonl').write_text(''.join(normal))
(out / 'full32.jsonl').write_text(''.join(full))
PY

nccl=0
if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/probe_qwen35_nccl.py > "$OUT/logs/nccl_probe.log" 2>&1; then
  nccl=1
fi
echo "NCCL probe success: $nccl" | tee "$OUT/logs/status.log"

for backend in gloo nccl; do
  if [[ "$backend" == nccl && "$nccl" != 1 ]]; then continue; fi
  for batch in 1 2 4; do
    if [[ -f "$OUT/reports/${backend}_b${batch}_normal.json" ]]; then
      echo "SKIP completed backend=$backend batch=$batch normal" | tee -a "$OUT/logs/status.log"
    else
    echo "$(date -Is) backend=$backend batch=$batch normal" | tee -a "$OUT/logs/status.log"
    if ! "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/benchmark_qwen35_2b_training.py \
      --manifest "$OUT/manifests/normal.jsonl" \
      --output "$OUT/reports/${backend}_b${batch}_normal.json" \
      --backend "$backend" --microbatch "$batch" --warmup 5 --updates 15 \
      > "$OUT/logs/${backend}_b${batch}_normal.log" 2>&1; then
      echo "FAILED backend=$backend batch=$batch normal" | tee -a "$OUT/logs/status.log"
      continue
    fi
    fi
    if [[ -f "$OUT/reports/${backend}_b${batch}_full32.json" ]]; then
      echo "SKIP completed backend=$backend batch=$batch full32" | tee -a "$OUT/logs/status.log"
      continue
    fi
    echo "$(date -Is) backend=$backend batch=$batch full32" | tee -a "$OUT/logs/status.log"
    if ! "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/benchmark_qwen35_2b_training.py \
      --manifest "$OUT/manifests/full32.jsonl" \
      --output "$OUT/reports/${backend}_b${batch}_full32.json" \
      --backend "$backend" --microbatch "$batch" --warmup 0 --updates 1 \
      > "$OUT/logs/${backend}_b${batch}_full32.log" 2>&1; then
      echo "FAILED backend=$backend batch=$batch full32" | tee -a "$OUT/logs/status.log"
    fi
  done
done
"$PY" scripts/report_qwen35_2b_util_benchmarks.py \
  > "$OUT/logs/report.log" 2>&1
if [[ ! -f "$OUT/reports/gloo_b1_normal.json" ]]; then
  echo "Baseline benchmark failed; inspect logs" | tee -a "$OUT/logs/status.log"
  exit 1
fi
date -Is > "$OUT/completed"
