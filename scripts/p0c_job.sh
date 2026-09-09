#!/bin/bash
#SBATCH --job-name=p0c_v2
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/p0c_v2_%j.out
# P0c v2. The v1 job failed because run_calib_eval.sh takes THREE args:
#   run_calib_eval.sh <train_log> <model_dir> <run_tag>
# v1 passed <run_tag> <epoch>, so it tried to open the tag as a file and died.
# The DONE line printed anyway -- which is exactly why an ARTIFACT check, not a DONE
# message, is the only acceptable evidence in this project.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_

run_one () {  # log, tag
  echo "=== $2 ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$1" "${M}$2" "$2"
  F=$(ls eval_results/abl_${2}_e*.csv 2>/dev/null | tail -1)
  if [ -n "$F" ]; then echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)"; else echo "ARTIFACT MISSING for $2"; fi
}
run_one logs/gld_loroW_onehot_21084179.out           loroW_onehot_s42
run_one logs/p3_a1_d1_s0_D1_uemb_seed42_21050057.out p3_a1_d1_s0_D1_uemb_seed42
echo "P0C_V2_DONE"
