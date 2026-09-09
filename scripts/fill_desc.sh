#!/bin/bash
# The descriptor arm that can actually train -- Ofir's hypothesis, third attempt.
#
# TWO PRIOR COLLAPSES, SAME CAUSE. Both earlier arms froze at RMSE 2.541 for all 15 epochs with
# val PCC noise around zero. The block carried 13.8x then 3.65x the one-hot energy and swamped
# every other input. The correctly-scaled table was built to fix exactly this, documents that
# failure in its own header, and had NEVER been wired to a run.
#
# VERIFIED BEFORE SUBMITTING, by pushing a real one-hot through the loader:
#   mordred_pca16_only   block (20,16)  median L2 3.653  -> 3.65x one-hot
#   mordred_pca16_unit   block (20,16)  median L2 1.045  -> 1.04x one-hot   <- the new mode
#   descriptor_path -> data_fixed/aa_descriptors_mordred_pca16_unit.csv, dim 16, keeps_onehot=False
#   gate_g4_cpu: ALL PASS at width 1092 (the new mode is additive; existing modes untouched,
#   so every previous run stays reproducible byte-for-byte)
#
# PREDICTION, ON RECORD BEFORE THE RESULT: the arm will now TRAIN (RMSE will move, val PCC will
# climb like the one-hot arm's 0.548 -> 0.712). But it will NOT beat one-hot by more than the
# 0.0344 seed band, because ProtT5 already predicts held-out-residue hydropathy at R^2 = 0.704,
# so the descriptors are largely redundant. The value here is getting a MEASUREMENT instead of a
# third technical collapse.
# FALSIFIER: if RMSE freezes again at a single value, the scale hypothesis is wrong and the
# blocker is elsewhere -- do not run a fourth arm without a different diagnosis.
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
sub gld_descunit_s42 --loss_mode ddg --aa_descriptors mordred_pca16_unit --seed 42
sub gld_descunit_s1  --loss_mode ddg --aa_descriptors mordred_pca16_unit --seed 1
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
