#!/bin/bash
#SBATCH --job-name=emb_bp
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/nissimb/DeepPEF/logs/emb_bp_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
python -u embbp2.py
echo "EXIT=$?"
