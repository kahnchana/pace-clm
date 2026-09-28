#!/usr/bin/env bash
# Run on the always-on eight-H200 node; the Slurm extraction covers the head.
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct
PYTHON=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/features" "$OUT/results" "$OUT/logs"

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/extract_feature_shards.py --split train \
    --train-manifest "$ROOT/manifests/pace_train_10pct_decisions.jsonl" \
    --video-start 450 --shard-suffix tail --output-dir "$OUT/features"
touch "$OUT/features/tail.done"

while ! "$PYTHON" - "$OUT/features" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
for rank in range(8):
    path = root / f"train.rank{rank:02d}.pt"
    if not path.is_file():
        raise SystemExit(1)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    if len(shard["completed_video_ids"]) < 450:
        raise SystemExit(1)
PY
do
    echo "waiting for low-priority extraction to cover first 450 videos per rank"
    sleep 60
done

# The head is fully covered and this node owns the disjoint tail.
scancel 36566 || true
"$PYTHON" scripts/merge_10pct_parallel.py --input-dir "$OUT/features" --split-index 450
cp "$ROOT/outputs/features/v1/pace200.pt" "$OUT/features/pace200.pt"
cp "$ROOT/outputs/features/v1/action_variants.pt" "$OUT/features/action_variants.pt"
export OMP_NUM_THREADS=4
"$PYTHON" scripts/train_10pct_curve.py --feature-dir "$OUT/features" --output-dir "$OUT/results"
"$PYTHON" scripts/eval_10pct_frozen.py --feature-dir "$OUT/features" --result-dir "$OUT/results"
"$PYTHON" scripts/benchmark_serving_10pct.py --feature-dir "$OUT/features" --result-dir "$OUT/results"
"$PYTHON" scripts/finalize_10pct.py --result-dir "$OUT/results"
"$PYTHON" scripts/write_10pct_provenance.py --output-root "$OUT"
touch "$OUT/results/complete.done"
