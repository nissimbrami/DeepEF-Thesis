#!/bin/bash
#SBATCH --job-name=score_done
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/score_done_%j.out
# Score every FINISHED golden arm that has no CSV. run_calib_eval.sh takes THREE args:
#   <train_log> <model_dir> <run_tag>   -- passing the tag alone silently writes nothing.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
for TAG in gld_loroWdesc2_s42 gld_w5_dg_s1 gld_w5_dg_s2 gld_slope_anchor_s42; do
  ls eval_results/abl_${TAG}_e*.csv >/dev/null 2>&1 && { echo "skip $TAG (scored)"; continue; }
  L=$(ls -t logs/${TAG}_*.out 2>/dev/null | head -1)
  D="${M}${TAG}"
  [ -z "$L" ] && { echo "no log for $TAG"; continue; }
  [ -d "$D" ] || { echo "no model dir for $TAG"; continue; }
  echo "=== $TAG (log $L) ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "$D" "$TAG"
  F=$(ls eval_results/abl_${TAG}_e*.csv 2>/dev/null | tail -1)
  [ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $TAG"
done
echo "SCORE_DONE"
