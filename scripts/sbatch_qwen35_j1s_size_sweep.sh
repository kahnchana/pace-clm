#!/usr/bin/env bash
#SBATCH --job-name=pace-qwen35-j1s-sizes
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-3-70-177
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-qwen35-size-sweep-j1s/logs/slurm-%j.out
set -euo pipefail
trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
cd /fsx/home/kranasinghe/repo/pace-clm
bash scripts/run_qwen35_j1s_size_sweep.sh
