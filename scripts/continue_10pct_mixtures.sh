#!/usr/bin/env bash
# Continue after all eight train.union.rankXX.pt shards have been written.
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/pace-clm-mixtures"
OLD=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/results" "$OUT/checkpoints" "$OUT/logs"

# The training-union extraction is started separately so its live progress can
# be monitored. Advance only after all eight workers finish their shards.
while true; do
  FINISHED=$("$PY" -c 'import pathlib,sys;print(sum("wrote " in line and "train.union.rank" in line for line in pathlib.Path(sys.argv[1]).read_text().splitlines()))' "$OUT/logs/extract-union.log")
  if [ "$FINISHED" -ge 8 ]; then
    break
  fi
  sleep 30
done

if [ ! -f "$OUT/results/feature_overlap.json" ]; then
  "$PY" scripts/merge_feature_shards.py --input-dir "$OUT/features" --split train \
    --shard-suffix union --output-name union --manifest "$OUT/manifests/union.jsonl" \
    > "$OUT/logs/merge-union.log" 2>&1
  "$PY" scripts/check_10pct_feature_overlap.py --old "$OLD/features/train.pt" \
    --new "$OUT/features/union.pt" --output "$OUT/results/feature_overlap.json" \
    > "$OUT/logs/check-feature-overlap.log" 2>&1
fi

"$PY" scripts/train_10pct_mixtures.py --old-feature-dir "$OLD/features" \
  --mixture-feature-dir "$OUT/features" --manifest-dir "$OUT/manifests" \
  --output-dir "$OUT/checkpoints" --conditions run3 run4 run5 \
  > "$OUT/logs/train-run3-5.log" 2>&1

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/extract_feature_shards.py \
  --split selection --validation-manifest "$OUT/validation/selection.jsonl" \
  --output-dir "$OUT/features" --save-every-videos 5 \
  > "$OUT/logs/extract-selection.log" 2>&1
"$PY" scripts/merge_feature_shards.py --input-dir "$OUT/features" --split selection \
  > "$OUT/logs/merge-selection.log" 2>&1

"$PY" scripts/eval_10pct_mixtures.py --set selection \
  --conditions control run1 run2 run3 run4 run5 \
  --feature-dir "$OUT/features" --validation-dir "$OUT/validation" \
  --checkpoint-dir "$OUT/checkpoints" --old-result-dir "$OLD/results" \
  --old-feature-dir "$OLD/features" --output-dir "$OUT/results" \
  > "$OUT/logs/eval-selection.log" 2>&1
"$PY" scripts/select_10pct_mixture.py \
  --selection-results "$OUT/results/selection.results.json" \
  --output "$OUT/results/selected_mixture.json" \
  > "$OUT/logs/select-mixture.log" 2>&1
SELECTED=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_mixture"])' "$OUT/results/selected_mixture.json")

"$PY" scripts/train_10pct_mixture_scaleup.py --selected "$SELECTED" \
  --old-feature-dir "$OLD/features" --mixture-feature-dir "$OUT/features" \
  --manifest-dir "$OUT/manifests" --output-dir "$OUT/scaleup" \
  > "$OUT/logs/train-scaleup.log" 2>&1

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/extract_feature_shards.py \
  --split final --validation-manifest "$OUT/validation/final.jsonl" \
  --output-dir "$OUT/features" --save-every-videos 5 \
  > "$OUT/logs/extract-final.log" 2>&1
"$PY" scripts/merge_feature_shards.py --input-dir "$OUT/features" --split final \
  > "$OUT/logs/merge-final.log" 2>&1

"$PY" scripts/eval_10pct_mixtures.py --set final --scaleup \
  --conditions selected balanced --feature-dir "$OUT/features" \
  --validation-dir "$OUT/validation" --checkpoint-dir "$OUT/scaleup" \
  --old-result-dir "$OLD/results" --old-feature-dir "$OLD/features" \
  --output-dir "$OUT/results" \
  > "$OUT/logs/eval-final.log" 2>&1
"$PY" scripts/report_10pct_mixtures.py --root "$OUT" \
  > "$OUT/logs/report.log" 2>&1
touch "$OUT/results/complete.done"
