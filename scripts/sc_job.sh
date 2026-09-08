#!/bin/bash
#SBATCH --job-name=sc_mech4
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/nissimb/DeepPEF/results/sidechain_mech4.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
python scripts/sidechain_mech4.py
