#!/usr/bin/env bash
# Run after all eight final feature shards are complete.
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
OLD=/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct
PY=/fsx/home/kranasinghe/repo/stream-gemma/.venv/bin/python
OUT="$ROOT/outputs/pace-clm-gate"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

"$PY" scripts/merge_feature_shards.py --input-dir "$OUT/features" --split final \
  > "$OUT/merge-final.log" 2>&1
"$PY" - <<'PY'
import json
import torch
from pathlib import Path
root = Path('outputs/pace-clm-gate')
data = torch.load(root / 'features/final.pt', map_location='cpu', weights_only=False)
manifest = [json.loads(line) for line in (root / 'validation/final.jsonl').read_text().splitlines()]
assert len(data['metadata']) == sum(row['inference_frame_count'] for row in manifest)
assert {row['video_id'] for row in data['metadata']} == {row['video_id'] for row in manifest}
print(f"verified {len(manifest)} videos and {len(data['metadata'])} decisions")
PY

"$PY" scripts/eval_10pct_mixtures.py --set final \
  --conditions control run5 run5_early run5_hold2 \
  --feature-dir "$OUT/features" --validation-dir "$OUT/validation" \
  --checkpoint-dir "$ROOT/outputs/pace-clm-mixtures/checkpoints" \
  --early-checkpoint-dir "$OUT/early" --hold2-checkpoint-dir "$OUT/hold2" \
  --old-result-dir "$OLD/results" --old-feature-dir "$OLD/features" \
  --output-dir "$OUT/final_eval" > "$OUT/final-eval.log" 2>&1

"$PY" scripts/eval_10pct_mixtures.py --set final --scaleup \
  --conditions selected balanced --feature-dir "$OUT/features" \
  --validation-dir "$OUT/validation" \
  --checkpoint-dir "$ROOT/outputs/pace-clm-mixtures/scaleup" \
  --old-result-dir "$OLD/results" --old-feature-dir "$OLD/features" \
  --output-dir "$OUT/scaleup_eval" > "$OUT/scaleup-eval.log" 2>&1

"$PY" scripts/test_temporal_gate.py --phase test \
  --feature-dir "$OUT/features" --validation-dir "$OUT/validation" \
  --old-result-dir "$OLD/results" \
  --mixture-checkpoint-dir "$ROOT/outputs/pace-clm-mixtures/checkpoints" \
  --early-checkpoint-dir "$OUT/early" --hold2-checkpoint-dir "$OUT/hold2" \
  --output-dir "$OUT/results" > "$OUT/gate-test.log" 2>&1

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_pace_token_shards.py --validation-manifest "$OUT/validation/final.jsonl" \
  --output-dir "$OUT/token" --save-every-videos 5 \
  > "$OUT/token-final.log" 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('outputs/pace-clm-gate')
manifest = [json.loads(line) for line in (root / 'validation/final.jsonl').read_text().splitlines()]
rows = [json.loads(line) for path in sorted((root / 'token').glob('pace_token.rank*.jsonl'))
        for line in path.read_text().splitlines()]
assert len(rows) == sum(row['inference_frame_count'] for row in manifest)
assert len({(row['video_id'], row['frame_index']) for row in rows}) == len(rows)
assert {row['video_id'] for row in rows} == {row['video_id'] for row in manifest}
print(f"verified {len(manifest)} token videos and {len(rows)} decisions")
PY

mkdir -p "$OUT/token_score"
"$PY" scripts/score_timing_paucf.py --result-dir "$OUT/token_score" \
  --token-dir "$OUT/token" --validation-manifest "$OUT/validation/final.jsonl" \
  --gold "$OUT/validation/final.gold.json" > "$OUT/token-score.log" 2>&1
touch "$OUT/results/complete.done"
