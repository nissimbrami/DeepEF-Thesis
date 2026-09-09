#!/bin/bash
#SBATCH --job-name=score10
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/score10_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
run () {
  T="$1"; shift
  ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1 && { echo "SKIP $T (already scored)"; return; }
  [ -f "${M}${T}/kf_all_epoch_14.pt" ] || { echo "SKIP $T (no e14 checkpoint)"; return; }
  L=$(ls -t logs/*${T}_*.out 2>/dev/null | head -1)
  [ -z "$L" ] && { echo "SKIP $T (no log)"; return; }
  echo "=== SCORING $T   eval-flags: [$*] ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T" "$*"
  F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
  if [ -n "$F" ]; then echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)"; else echo "ARTIFACT MISSING: $T"; fi
}
# only WIDTH-CHANGING flags go to evaluate.py; loss/graph flags do not alter the model shape
run w5_dg_s3         --burial_features
run slope_w5_s1      --burial_features
run slope0.5_s42
run slope0.5_s1
run slope0.7_s42
run w7edge_s1
run w7span4_s1
run coil_nu588_s42
run coiledge_dg_s42
run loroW_desc_s42   --aa_descriptors mordred_pca16
echo "SCORE10_DONE"
