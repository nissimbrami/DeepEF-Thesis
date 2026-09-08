#!/bin/bash
# WAVE 8 - HSE burial. The external plan proposed building "W14 direction burial" from scratch
# over two weeks. IT ALREADY EXISTS: train_utils.compute_hse is half-sphere exposure, which is
# exactly the direction-aware burial the plan describes ("a residue can be surrounded yet point
# outward"). It is exposed as --burial_mode hse, it needs only CA and CB, and a scontrol scan
# plus an eval_results scan confirm it has NEVER been run and is NOT queued.
#
# WHY IT SATISFIES THE PLAN'S OWN CONSTRAINT. The plan's key arithmetic is that any per-residue
# descriptor table is one_hot @ T, a rank-1 projection of a block fc1 already receives, so it
# cannot add information. HSE is NOT a function of residue type: two tryptophans at different
# positions get different values because the geometry differs. And burial is ZERO in the unfolded
# state, so it is state-dependent and survives in E_folded - E_unfolded rather than cancelling.
#
# METRIC: dG and std(b_p) for the reference-state channel, ddG/a_p for the within-protein channel.
# The plan's own prediction to test: the hydrophobic-vs-polar slope gap should shrink at BURIED
# positions (measured deficit 0.095) and not at exposed ones (0.024). If it shrinks everywhere
# equally the feature is not doing what it was designed to do.
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
# direction-aware burial on the dG channel, where the reference-state effect lives
sub gld_hse_dg_s42  --loss_mode dg  --burial_features --burial_mode hse --seed 42
sub gld_hse_dg_s1   --loss_mode dg  --burial_features --burial_mode hse --seed 1
# and on ddG, where the within-protein slope deficit lives
sub gld_hse_ddg_s42 --loss_mode ddg --burial_features --burial_mode hse --seed 42
# HSE x W12: geometry (state-dependent) x chemistry (transfer free energy)
sub gld_hse_w12_s42 --loss_mode ddg --burial_features --burial_mode hse --sidechain_features --seed 42
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
