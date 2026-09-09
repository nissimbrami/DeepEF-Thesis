#!/bin/bash
#SBATCH --job-name=taskloop
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/nissimb/DeepPEF/logs/taskloop_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
# restart-on-crash wrapper: rc=3 means "deliberate stop", anything else is a crash -> restart
while true; do
  python -u scripts/taskloop.py
  rc=$?
  echo "taskloop exited rc=$rc at $(date)"
  [ "$rc" = "3" ] && break
  sleep 60
done
