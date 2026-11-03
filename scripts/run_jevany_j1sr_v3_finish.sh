#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1sr-v3"
CHECKPOINT="$OUT/full-seed42/step-008243"
VALID="$OUT/evaluation/validation/full"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/tmp/pace-jevany-j1sr-v3-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs"

"$PY" - <<'PY'
import json
from pathlib import Path
p=Path('outputs/jevany-j1sr-v3/validation/canary64_gate_result.json')
if not json.loads(p.read_text())['passed']:
    raise RuntimeError('early response/coverage gate failed')
PY

if [[ ! -f "$OUT/full-seed42/summary.rank0.json" ]]; then
  for attempt in $(seq 1 10); do
    echo "Full training attempt $attempt at $(date -Is)" >> "$OUT/logs/full-train.log"
    if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_streaming.py --mode J1 --stage full --seed 42 \
      --manifest "$OUT/manifests/train.jsonl" --output "$OUT/full-seed42" \
      --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 100 \
      --resume >> "$OUT/logs/full-train.log" 2>&1; then
      break
    fi
    sleep 30
  done
fi
test -f "$OUT/full-seed42/summary.rank0.json"
test -f "$CHECKPOINT/training_state.pt"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
  --validation-manifest "$ROOT/outputs/jevany-j1sr-v2/validation/frozen.jsonl" \
  --output-dir "$VALID" --response-margin 0.0 --answer-model promoted \
  >> "$OUT/logs/validation.log" 2>&1
"$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$VALID" \
  --manifest "$ROOT/outputs/jevany-j1sr-v2/validation/frozen.jsonl" \
  --gold "$ROOT/outputs/jevany-j1sr-v2/validation/frozen.gold.json" \
  --world-size 8 --skip-judge >> "$OUT/logs/validation.log" 2>&1
if "$PY" - <<'PY'
import json
from pathlib import Path
p=Path('outputs/jevany-j1sr-v3/evaluation/validation/full/official/results/timing_prejudge.json')
x=json.loads(p.read_text())
raise SystemExit(0 if x['total_physical_fires'] >= 20 and x['covered_gold_windows'] >= 1 else 1)
PY
then
  "$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$VALID" \
    --manifest "$ROOT/outputs/jevany-j1sr-v2/validation/frozen.jsonl" \
    --gold "$ROOT/outputs/jevany-j1sr-v2/validation/frozen.gold.json" \
    --world-size 8 >> "$OUT/logs/validation.log" 2>&1
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$VALID" \
    >> "$OUT/logs/validation.log" 2>&1
else
  "$PY" - <<'PY'
import json
from pathlib import Path
p=Path('outputs/jevany-j1sr-v3/evaluation/selection.json')
p.write_text(json.dumps({'selected':False,'reason':'full validation became effectively silent'},indent=2)+'\n')
PY
  exit 0
fi

if ! "$PY" - <<'PY'
import json
from pathlib import Path
base=json.loads(Path('outputs/jevany-j1sr-v2/evaluation/validation/j1s/official/results/exp10_judged_summary.json').read_text())
new=json.loads(Path('outputs/jevany-j1sr-v3/evaluation/validation/full/official/results/exp10_judged_summary.json').read_text())
selected=new['pauc_f']>base['pauc_f']
record={'selected_for_benchmarks':selected,'j1s_judged_pauc_f':base['pauc_f'],
        'v3_judged_pauc_f':new['pauc_f'],'v3_fires':new['total_physical_fires'],
        'v3_covered_gold_windows':new['covered_gold_windows']}
Path('outputs/jevany-j1sr-v3/evaluation/selection.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
raise SystemExit(0 if selected else 1)
PY
then
  exit 0
fi

for benchmark in web ego; do
  if [[ "$benchmark" == ego ]]; then
    if ! "$PY" - <<'PY'
import json
from pathlib import Path
x=json.loads(Path('outputs/jevany-j1sr-v3/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json').read_text())
raise SystemExit(0 if x['num_fires']/x['num_videos']>=0.1 else 1)
PY
    then
      echo "EGO paused: WEB response rate below 0.1/video" >> "$OUT/logs/benchmarks.log"
      break
    fi
  fi
  dir="$OUT/evaluation/benchmarks/$benchmark"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$CHECKPOINT" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted >> "$OUT/logs/benchmarks.log" 2>&1
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8 \
    >> "$OUT/logs/benchmarks.log" 2>&1
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir" \
    >> "$OUT/logs/benchmarks.log" 2>&1
done
date -Is > "$OUT/evaluation/completed"
