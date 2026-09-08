#!/bin/bash
# WAVE 7 - K20 unfolded ensemble. Verified before submission, and a units bug fixed first.
#
# THE BUG THAT WOULD HAVE MADE THIS MEANINGLESS: the delivered script defaulted to
# coil_b_fixed = 3.8. Ours is _COIL_B_FIXED = 5.82A * 0.1 = 0.582 MODEL UNITS
# (train_utils.py:472-474). 3.8 is wrong in both unit systems and ~6.5x too large; the
# Gaussian kernel exp(coef*d^2) would saturate and flatten the unfolded reference to
# something featureless -- the same units bug that killed w5_dg.py. Fixed to 0.582.
#
# SELF-VERIFY AFTER THE FIX, all PASS:
#   triangle inequality 0.00% violations  <- proves coordinates are sampled, not distances
#   Flory scaling worst relative error 14.8% (< 25%)
#   cache determinism
#   ensemble diversity: mean coordinate spread 2.138 A (> 0.1)
#   gate_g4_cpu: ALL PASS, width 1092
#   flags proven parsed: --unfolded_ensemble_k 8 -> CFG = 8, reduce -> logsumexp
#
# METRIC: std(b_p) on dG. This is a REFERENCE-STATE lever, so ddG would cancel most of it.
# FALSIFIER, pre-registered: if k=8 does not push std(b_p) below the no-coil baseline of
# 0.9972, the reference-state programme is CLOSED. Not "try k=16".
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
  echo "SUBMIT $N -> $J"
}
# mean-field vs true free energy: their DIFFERENCE is the conformational entropy
sub gld_ens8mean_dg_s42  --loss_mode dg --flory_unfolded --flory_nu 0.588 --unfolded_ensemble_k 8 --unfolded_ensemble_reduce mean     --seed 42
sub gld_ens8lse_dg_s42   --loss_mode dg --flory_unfolded --flory_nu 0.588 --unfolded_ensemble_k 8 --unfolded_ensemble_reduce logsumexp --seed 42
sub gld_ens3mean_dg_s42  --loss_mode dg --flory_unfolded --flory_nu 0.588 --unfolded_ensemble_k 3 --unfolded_ensemble_reduce mean     --seed 42
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
