#!/bin/bash
# CONTROL SEED REPLICATION -- the measurement every comparison in this project depends on.
#
# WHY. Every reported gain is measured against ONE control run (calib_ctrl_repro2, pooled 0.5635)
# drawn from a distribution whose seed sd is 0.0344. The sigma family -- five seeds of the same
# control-like configuration -- averages 0.5781, i.e. 0.0146 HIGHER. Re-scoring every family
# against that mean instead of the single run, NOT ONE clears the noise band:
#   p3_a0_d0_s0_D0_coil  +0.0457 -> +0.0311
#   slope1.0 (n=10)      +0.0163 -> +0.0017
# The four families that looked like they beat noise did so only because the comparator was a
# lucky single draw. That is the same error as quoting 0.6382 as the headline -- an extremum
# treated as an estimate -- except here it flatters everything above it.
#
# This submits three more control seeds so the comparator becomes a mean with a real sd.
# BASE is identical to every other arm: no lever flags at all.
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
sub gld_ctrl_s1 --loss_mode ddg --seed 1
sub gld_ctrl_s2 --loss_mode ddg --seed 2
sub gld_ctrl_s3 --loss_mode ddg --seed 3
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
