#!/bin/bash
# Keep the 8 golden cards saturated, in priority order.
#
# WHY THIS ORDER. Measured facts driving it:
#   - --slope_weight 1.0 is the ONLY lever that beat the seed band (+0.058 pooled, a_p 0.496->0.740)
#     and it is n=1. Seed replication turns the project's one real result into a defensible one.
#   - The LORO descriptor arm COLLAPSED for a technical reason (descriptors centred but never
#     scaled, ~13.8x the one-hot energy), so the hypothesis was never tested. mordred_pca16_only
#     REPLACES one-hot instead of appending, which removes the scale clash.
#   - W5 burial on dG scored a_p 0.664 / s 0.870, the best a_p of any single-flag arm, at n=1.
#   - D0 seeds are what the A/B/C main effects need; D1 is settled at 14.4 sigma and is not queued.
#
# Golden lane = --partition=rtx6000 --qos keasar (a separate MaxTRESPA=8 pool from the public 5).
# Every job carries --resume so a preemption requeue continues instead of restarting.
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"

sub () { # name, flags
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "$N -> $J"
}

# P1 (3 cards) - replicate the ONLY proven lever across seeds
sub gld_slope1.0_s1 --loss_mode ddg --slope_weight 1.0 --seed 1
sub gld_slope1.0_s2 --loss_mode ddg --slope_weight 1.0 --seed 2
sub gld_slope1.0_s3 --loss_mode ddg --slope_weight 1.0 --seed 3

# P2 (1 card) - the LORO test that never actually ran, with the scale clash removed
sub gld_loroWdesc2_s42 --loss_mode ddg --holdout_residues W --aa_descriptors mordred_pca16_only --seed 42

# P3 (2 cards) - W5-on-dG had the best a_p of any single-flag arm; replicate it
sub gld_w5_dg_s1 --loss_mode dg --burial_features --flory_unfolded --coil_b fixed --seed 1
sub gld_w5_dg_s2 --loss_mode dg --burial_features --flory_unfolded --coil_b fixed --seed 2

# P4 (2 cards) - combine the two levers that each moved a_p, to test whether they add
sub gld_slope_w5_s42 --loss_mode ddg --slope_weight 1.0 --burial_features --seed 42
sub gld_slope_anchor_s42 --loss_mode ddg --slope_weight 1.0 --wt_anchor_weight 0.3 --seed 42

echo "--- golden now: $(squeue -u nissimb -h -o '%P'|grep -c rtx6000)/8"
