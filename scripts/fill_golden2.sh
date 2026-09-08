#!/bin/bash
# WAVE 2 - queued BEHIND the 8 running arms so the golden lane never idles.
# SLURM holds these as PENDING under the keasar QOS and starts each as a card frees.
#
# PRIORITY ARGUMENT (measured, not guessed):
#  1. W12 side-chain: targets the LARGEST measured deficit. The model compresses mutations to
#     hydrophobic destinations 2x harder (slope 0.27-0.31 vs 0.53-0.57) AND Spearman degrades in
#     lockstep (corr KD -0.734/-0.740) => LOST INFORMATION, which no affine calibration recovers.
#     Mechanism verified as transfer free energy: corr(dG_transfer, slope) = -0.922 vs volume -0.408,
#     CV R^2 0.753 vs 0.031. 3 seeds because the seed band is +/-0.060 and n=1 cannot clear it.
#  2. W12 x slope combo: slope fixes a_p (0.496->0.857) but pooled FELL, proving b_p dominates.
#     W12 adds information rather than rescaling it, so the two should be complementary, not rival.
#  3. W5 seed 3: completes 3-seed replication of the best single-flag a_p (0.664).
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"

sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (already queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (already scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "$N -> $J"
}

# --- W12 side-chain, 3 seeds: the largest untested lever, scored on ddG AND dG ---
sub gld_w12_s42 --loss_mode ddg --sidechain_features --burial_features --seed 42
sub gld_w12_s1  --loss_mode ddg --sidechain_features --burial_features --seed 1
sub gld_w12_s2  --loss_mode ddg --sidechain_features --burial_features --seed 2
# --- W12 on the dG channel: burial cols are zero unfolded, so folded-minus-unfolded IS the force ---
sub gld_w12_dg_s42 --loss_mode dg --sidechain_features --burial_features --seed 42
# --- combination: information (W12) + rescaling (slope). Different mechanisms, should compose ---
sub gld_w12_slope_s42 --loss_mode ddg --sidechain_features --burial_features --slope_weight 1.0 --seed 42
# --- completes 3-seed replication of the best single-flag a_p ---
sub gld_w5_dg_s3 --loss_mode dg --burial_features --seed 3
# --- slope+W5 second seed: the current best combo arm at n=1 ---
sub gld_slope_w5_s1 --loss_mode ddg --slope_weight 1.0 --burial_features --seed 1
echo "--- golden queue after submit ---"
squeue -u nissimb -h -o "%.10i %.24j %.2t" | grep gld_
