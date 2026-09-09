#!/bin/bash
#SBATCH --job-name=score_all6
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/score_all6_%j.out
# Score every trained-but-unscored arm, passing each one's TRAINING flags as the 4th argument.
# Without them the model is built at baseline width and load_state_dict fails on every layer
# while the script still prints DONE.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
run () {
  T="$1"; shift
  ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1 && { echo "skip $T (scored)"; return; }
  L=$(ls -t logs/gld_${T}_*.out logs/${T}_*.out 2>/dev/null | head -1)
  [ -z "$L" ] && { echo "no log for $T"; return; }
  [ -d "${M}${T}" ] || { echo "no model dir for $T"; return; }
  echo "=== $T   flags: $*   log: $L ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T" "$*"
  F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
  [ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $T"
}
run slope1.0_s3
run slope_w5_s42 --burial_features
run w5_dg_s1 --burial_features
run w5_dg_s2 --burial_features
run loroWdesc2_s42 --aa_descriptors mordred_pca16_only
run loroW_desc_s42 --aa_descriptors mordred_pca16
echo "SCORE_ALL6_DONE"
