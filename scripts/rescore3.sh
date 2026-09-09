#!/bin/bash
#SBATCH --job-name=rescore3
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/rescore3_%j.out
# Rescore the three width-changing arms, now passing their training flags through.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
run () {  # tag, flags
  T="$1"; shift
  L=$(ls -t logs/gld_${T}_*.out 2>/dev/null | head -1)
  echo "=== $T  flags: $* ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T" "$*"
  F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
  [ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $T"
}
run loroWdesc2_s42 --aa_descriptors mordred_pca16_only
run w5_dg_s1 --burial_features
run w5_dg_s2 --burial_features
echo "RESCORE3_DONE"
