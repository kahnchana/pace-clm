#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
SG="$ROOT/../stream-gemma"
PY="$SG/.venv/bin/python"
OUT="$ROOT/outputs/jevany-j1sr"
J1S="$ROOT/outputs/jevany-j1s/full-seed42/step-008243"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$SG/src:$SG"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-j1sr-triton
mkdir -p "$TRITON_CACHE_DIR" "$OUT/logs" "$OUT/source"

"$PY" scripts/build_jevany_j1sr.py mine | tee "$OUT/logs/manifest.log"
"$PY" - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
root=Path('/fsx/home/kranasinghe/repo/pace-clm')
out=root/'outputs/jevany-j1sr'
files=[root/p for p in (
    'scripts/build_jevany_j1sr.py', 'scripts/eval_jevany_streaming.py',
    'scripts/train_jevany_streaming.py', 'scripts/run_jevany_j1sr_current_node.sh',
    'src/pace_clm/jevany_streaming.py', 'configs/actions_v1.yaml',
    'outputs/jevany-j1sr/manifests/train.jsonl',
    'outputs/jevany-j1sr/validation/frozen.jsonl',
    'outputs/jevany-j1sr/validation/frozen.gold.json',
    'outputs/jevany-j1s/full-seed42/step-008243/controller_adapter/adapter_model.safetensors',
    'outputs/jevany-j1s/full-seed42/step-008243/pointer_head.pt')]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()
hashes={str(p.relative_to(root)):digest(p) for p in files}
record={'files_sha256':hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'recipe':'J1 history, fresh promoted PACE initialization; 8 GPUs; seed 42; 263760 presentations; 41220 Response; 8243 updates; zero margin; promoted answer generator',
        'mining_audit':json.loads((out/'manifests/audit.json').read_text())}
def ids(path):
    with path.open() as f:
        return {json.loads(line)['video_id'] for line in f if line.strip()}
train=ids(out/'manifests/train.jsonl')
valid=ids(out/'validation/frozen.jsonl')
old_valid=ids(root/'outputs/pace-clm-mixtures/validation/selection.jsonl') | ids(root/'outputs/pace-clm-mixtures/validation/final.jsonl')
bench={split:{str(row['question_id']).rsplit('.',1)[0] for row in json.loads(
    (root.parent/f'stream-gemma/datasets/proactive-videoqa-{split}/anno.json').read_text())}
    for split in ('web','ego')}
overlaps={'frozen':len(train&valid),'old_validation':len(train&old_valid),
          **{split:len(train&names) for split,names in bench.items()}}
if any(overlaps.values()):
    raise ValueError(f'train/evaluation video overlap: {overlaps}')
record['video_disjointness']={'train_videos':len(train),'frozen_validation_videos':len(valid),
                              'intersections':overlaps}
(out/'source/preflight.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY

if [[ ! -f "$OUT/full-seed42/summary.rank0.json" ]]; then
  for attempt in $(seq 1 10); do
    resume_args=()
    if [[ -f "$OUT/full-seed42/latest.json" ]]; then resume_args=(--resume); fi
    echo "Training attempt $attempt at $(date -Is)" >> "$OUT/logs/train.log"
    if "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
      scripts/train_jevany_streaming.py --mode J1 --stage full --seed 42 \
      --manifest "$OUT/manifests/train.jsonl" --output "$OUT/full-seed42" \
      --head-lr 0.0001 --lora-lr 0.000002 --accumulation 4 --save-every 100 \
      "${resume_args[@]}" >> "$OUT/logs/train.log" 2>&1; then
      break
    fi
    echo "Training attempt $attempt failed at $(date -Is)" >> "$OUT/logs/train.log"
    sleep 30
  done
fi
test -f "$OUT/full-seed42/summary.rank0.json"
checkpoint=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/full-seed42/latest.json")
test -f "$checkpoint/controller_adapter/adapter_model.safetensors"

for model in j1s j1sr; do
  if [[ "$model" == j1s ]]; then controller="$J1S"; else controller="$checkpoint"; fi
  dir="$OUT/evaluation/validation/$model"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$controller" \
    --validation-manifest "$OUT/validation/frozen.jsonl" --output-dir "$dir" \
    --response-margin 0.0 --answer-model promoted >> "$OUT/logs/eval-$model.log" 2>&1
  "$PY" scripts/prepare_exp10_scoring.py --rollout-dir "$dir" \
    --manifest "$OUT/validation/frozen.jsonl" --gold "$OUT/validation/frozen.gold.json" \
    --world-size 8 >> "$OUT/logs/eval-$model.log" 2>&1
  "$PY" scripts/judge_exp10_rollout.py --rollout-dir "$dir" >> "$OUT/logs/eval-$model.log" 2>&1
done

for benchmark in web ego; do
  dir="$OUT/evaluation/benchmarks/$benchmark"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/eval_jevany_streaming.py --controller "$checkpoint" \
    --benchmark "$benchmark" --output-dir "$dir" --response-margin 0.0 \
    --answer-model promoted >> "$OUT/logs/eval-$benchmark.log" 2>&1
  "$PY" scripts/merge_jevany_benchmark.py --input-dir "$dir" --world-size 8 \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
  "$PY" scripts/score_exp10_web_ego.py "$benchmark" --pred-dir "$dir" \
    >> "$OUT/logs/eval-$benchmark.log" 2>&1
done
"$PY" scripts/report_jevany_j1sr.py >> "$OUT/logs/report.log" 2>&1
date -Is > "$OUT/evaluation/completed"
