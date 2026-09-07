#!/bin/bash
#SBATCH --job-name=w6emb
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/home/nissimb/DeepPEF/results/w6emb_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export PYTHONIOENCODING=utf-8
python scripts/w6_emb_probe.py
