#!/bin/bash
#SBATCH --job-name=ckmine
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ckmine_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
echo "=== MINE_ALL ==="; python -u scripts/mine_all.py
echo "=== SEED_ALL ==="; python -u scripts/seed_all.py
echo "=== ECON ==="; python -u scripts/econ.py
echo "=== DONE ==="
