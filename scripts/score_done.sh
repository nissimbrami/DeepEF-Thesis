#!/bin/bash
#SBATCH --job-name=score_done2
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/score_done2_%j.out
# v2. v1 failed on every arm with "no model dir": the sbatch --job-name carries the gld_ prefix
# but --run_tag strips it, so the model directory is ..._w5_dg_s1, not ..._gld_w5_dg_s1.
# The eval CSV is likewise named after the run_tag. Verified against the training log.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
for TAG in loroWdesc2_s42 w5_dg_s1 w5_dg_s2 slope_anchor_s42; do
  ls eval_results/abl_${TAG}_e*.csv >/dev/null 2>&1 && { echo "skip $TAG (scored)"; continue; }
  L=$(ls -t logs/gld_${TAG}_*.out 2>/dev/null | head -1)
  D="${M}${TAG}"
  [ -z "$L" ] && { echo "no log for $TAG"; continue; }
  [ -d "$D" ] || { echo "no model dir $D"; continue; }
  echo "=== $TAG  log=$L ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "$D" "$TAG"
  F=$(ls eval_results/abl_${TAG}_e*.csv 2>/dev/null | tail -1)
  [ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $TAG"
done
echo "SCORE_DONE2"
