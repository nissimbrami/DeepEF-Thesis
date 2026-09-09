#!/bin/bash
#SBATCH --job-name=score_w12
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/score_w12_%j.out
# Score the W12 side-chain arms as they finish. W12 is the lever the r/s decomposition says we
# need: it adds INFORMATION rather than rescaling, and sd(r)=0.0223 vs sd(s)=0.2441 shows every
# lever so far has only moved s. Requires --sidechain_features or the model is built narrow and
# load_state_dict fails on every layer while still printing DONE.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
run () {
  T="$1"; shift
  ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1 && { echo "skip $T (scored)"; return; }
  [ -d "${M}${T}" ] || { echo "no model dir for $T"; return; }
  # only score arms that reached epoch 14
  [ -f "${M}${T}/kf_all_epoch_14.pt" ] || { echo "skip $T (not finished)"; return; }
  L=$(ls -t logs/gld_${T}_*.out 2>/dev/null | head -1)
  [ -z "$L" ] && { echo "no log for $T"; return; }
  echo "=== $T   flags: $* ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T" "$*"
  F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
  [ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $T"
}
run w12_s2  --sidechain_features --burial_features
run w12_s1  --sidechain_features --burial_features
run w12_s42 --sidechain_features --burial_features
run w12_dg_s42 --sidechain_features --burial_features
run w12_slope_s42 --sidechain_features --burial_features
run w5_dg_s3 --burial_features
echo "SCORE_W12_DONE"
