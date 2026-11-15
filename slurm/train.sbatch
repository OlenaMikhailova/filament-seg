#!/bin/bash
# Usage: sbatch slurm/train.sbatch configs/baseline_r34.yaml fold=0 work_dir=runs/r34_f0
# Adjust partition / gres / time to your lab cluster.
#SBATCH --job-name=filament
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs
source ~/.bashrc
conda activate filament   # or: source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py "$@"
