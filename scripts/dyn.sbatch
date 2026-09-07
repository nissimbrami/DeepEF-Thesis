#!/bin/bash
#SBATCH --job-name=ckptdyn
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/ckptdyn_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
python -u scripts/ckpt_dynamics.py
echo "DONE_DYN"
