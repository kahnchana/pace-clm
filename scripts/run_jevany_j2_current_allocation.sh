#!/usr/bin/env bash
# Run as one step inside the existing eight-GPU allocation after smoke passes.
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
cd "$ROOT"
test -s outputs/jevany-j2/smoke/report.json
mkdir -p outputs/jevany-j2/logs
date -Is > outputs/jevany-j2/run.started
bash scripts/sbatch_jevany_j2_train.sh \
  > outputs/jevany-j2/logs/train-current-allocation.log 2>&1
date -Is > outputs/jevany-j2/train.completed
bash scripts/sbatch_jevany_j2_eval.sh \
  > outputs/jevany-j2/logs/eval-current-allocation.log 2>&1
date -Is > outputs/jevany-j2/eval.completed
