#!/bin/bash
# WAVE 3 - queued behind waves 1 and 2 so the golden lane never idles.
# Every arm here is justified by a MEASURED fact, not a hunch:
#  - slope OVERSHOOTS: s crosses 1.0 between e4 and e8 (P0a). Perfect calibration is s=1.0, so
#    weights BELOW 1.0 are untested territory and 0.5 may land on s=1 at the val-selected epoch.
#  - W12 on the dG channel with W5 stacked: the burial-weighted columns are zero unfolded, so
#    folded-minus-unfolded IS the hydrophobic driving force; only dG can see it (metric rule).
#  - The ensemble (K20) needs a trained arm, but first the coil must be given the right nu.
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"
sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "$N -> $J"
}
# --- slope BELOW 1.0: s overshoots at 1.0, so the optimum is lower. Never tested. ---
sub gld_slope0.5_s42 --loss_mode ddg --slope_weight 0.5 --seed 42
sub gld_slope0.5_s1  --loss_mode ddg --slope_weight 0.5 --seed 1
sub gld_slope0.7_s42 --loss_mode ddg --slope_weight 0.7 --seed 42
# --- W12 + W5 on dG: both act through burial; dG is the only metric that can see them ---
sub gld_w12_w5_dg_s42 --loss_mode dg --sidechain_features --burial_features --seed 42
# --- the coil with the CORRECT exponent, trained (the sweep was frozen-checkpoint only) ---
sub gld_coil_nu588_s42 --loss_mode dg --flory_unfolded --flory_nu 0.588 --seed 42
echo "--- golden queue now ---"
squeue -u nissimb -h -o "%.10i %.24j %.2t" | grep gld_ | wc -l
