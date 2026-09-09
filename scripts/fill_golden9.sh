#!/bin/bash
# WAVE 9 - W15 side-chain PACKING. Verified end-to-end before submission.
#
# THE PROBLEM I FOUND BEFORE WIRING. Coordinates are SHARED across variants: 2PTL has ONE
# [62,4,3] backbone and 3,949 variants differing only in one_hot. A feature computed from the
# packed WT structure alone is therefore IDENTICAL for every variant and cancels EXACTLY in
# ddG = dG_mut - dG_wt. Building it that way would have produced an inert lever that trains,
# prints numbers, and means nothing -- the W5 bug repeated.
#
# THE FIX. Columns 0/2/3 read one_hot, so they change at the MUTATED position:
#   col0 reach(aa)              varies with the mutant
#   col1 env_density            geometry only, shared
#   col2 reach * env_density    THE INTERACTION -- bulky residue in a crowded pocket
#   col3 clash proxy            nonzero only when the mutant does not fit
#
# VERIFIED (gate_w15.py 10/10 PASS):
#   block DIFFERS between WT and mutant, max|delta| = 0.6222
#   the difference is LOCAL to the mutated position (exactly 1 position changed)
#   env_density is UNCHANGED by mutation, as designed
#   TRP in a crowded pocket 1.1667 vs in open space 0.2800
#   unfolded is EXACTLY zero; folded sum 102.03 vs unfolded 0.0 on the real graph
#   graph width 1093 -> 1097 (+4); gate_g4_cpu ALL PASS at 1092 with the flag off
#   flag proven parsed: --w15_features -> CFG.w15_features = True
#
# NOT one_hot @ T: measured on the FASPR reconstructions, one-hot alone explains only
# 0.443 / 0.488 / 0.245 of sc_sasa / contacts / clash, so 55-76% is geometry.
#
# METRIC: ddG per-protein PCC and a_p (within-protein information lever), AND std(b_p) to
# confirm it does not damage the offset. FALSIFIER: if per-protein ddG PCC does not beat
# control by more than the +/-0.060 seed band across 2 seeds, W15 is rejected.
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
sub gld_w15_s42 --loss_mode ddg --w15_features --seed 42
sub gld_w15_s1  --loss_mode ddg --w15_features --seed 1
sub gld_w15_s2  --loss_mode ddg --w15_features --seed 2
# packing geometry x transfer-free-energy chemistry
sub gld_w15_w12_s42 --loss_mode ddg --w15_features --sidechain_features --burial_features --seed 42
# and on dG, where the folded-minus-unfolded packing term lives
sub gld_w15_dg_s42 --loss_mode dg --w15_features --seed 42
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
