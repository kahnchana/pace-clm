#!/usr/bin/env bash
#SBATCH --job-name=pace-qwen35-2b-j1s-web
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-3-70-177
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-qwen35-2b-j1s/logs/slurm-%j.out
set -euo pipefail
trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
cd /fsx/home/kranasinghe/repo/pace-clm
direct_pid_file=outputs/jevany-qwen35-2b-j1s/direct_pipeline.pid
if [[ -f "$direct_pid_file" ]]; then
  direct_pid=$(cat "$direct_pid_file")
  while ps -p "$direct_pid" -o args= 2>/dev/null | grep -q '^bash scripts/run_qwen35_2b_j1s_web.sh$'; do
    sleep 60
  done
fi
bash scripts/run_qwen35_2b_j1s_web.sh
