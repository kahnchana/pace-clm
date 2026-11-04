#!/usr/bin/env bash
#SBATCH --job-name=pace-qwen35-2b-util
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-3-70-177
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-qwen35-2b-util/logs/slurm-%j.out
set -euo pipefail
cd /fsx/home/kranasinghe/repo/pace-clm
bash scripts/run_qwen35_2b_util_benchmarks.sh
