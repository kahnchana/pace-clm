#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$ROOT/outputs/jevany/deps032:$ROOT/src:$ROOT/scripts:$ROOT/../stream-gemma/src:$ROOT/../stream-gemma"
RUN="$ROOT/outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2"
PY="$ROOT/../stream-gemma/.venv/bin/python"
case "${1:-}" in
  prepare)
    "$PY" -u scripts/prepare_jevany_episode_balanced_mixed.py --output "$RUN"
    exec "$PY" -u scripts/run_jevany_mixed_workflow.py --run "$RUN" --prepare-only
    ;;
  train)
    [[ "$(hostname)" == ip-10-3-120-167 && "${SLURM_JOB_ID:-}" == 38177 ]]
    exec "$PY" -u scripts/run_jevany_mixed_workflow.py --run "$RUN"
    ;;
  eval)
    [[ "$(hostname)" == ip-10-3-70-177 && "${SLURM_JOB_ID:-}" == 38226 ]]
    exec "$PY" -u scripts/eval_jevany_mixed_queue.py --run "$RUN"
    ;;
  *) echo 'usage: run_jevany_episode_balanced_mixed.sh prepare|train|eval' >&2; exit 2 ;;
esac
