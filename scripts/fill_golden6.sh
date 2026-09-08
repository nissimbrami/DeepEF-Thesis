#!/bin/bash
# WAVE 6 - K21 distogram, verified end-to-end before submission.
#
# VERIFICATION PASSED BEFORE THIS SUBMIT:
#   distogram_head.py self-test: 6/6 PASS -- shape (2,40,40,32), symmetric in (i,j),
#     loss finite 3.4973, untrained loss near ln(32)=3.466, gradients flow (sum|grad|=40.91),
#     and it OVERFITS one example 3.488 -> 0.009 (proves the gradient path is real).
#   gate_g4_cpu after wiring: ALL PASS, width 1092 (the flag adds a LOSS, not columns).
#   flag actually parsed: --distogram_weight 0.25 -> CFG.distogram_weight = 0.25 (proven by
#     importing train.py with that argv, not by grepping the source).
#
# WEIGHT CHOICE. IFUM uses 100, but their auxiliary is on a comparable scale to their primary.
# Ours is a cross-entropy of order ln(32)~3.5 against an MSE of order 1, so 100 would let the
# auxiliary dominate completely. Sweeping {0.01, 0.1, 1.0} instead.
#
# METRIC. This is a WITHIN-protein information lever, so ddG can see it: score on per-protein
# ddG PCC and a_p. Also report std(b_p) to confirm it does not damage the offset.
# FALSIFIER: if per-protein ddG PCC does not beat control by more than the +/-0.060 seed band,
# the head is rejected. An improving auxiliary loss with a flat metric is a FAILURE, not a partial win.
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
sub gld_disto0.01_s42 --loss_mode ddg --distogram_weight 0.01 --seed 42
sub gld_disto0.1_s42  --loss_mode ddg --distogram_weight 0.1  --seed 42
sub gld_disto1.0_s42  --loss_mode ddg --distogram_weight 1.0  --seed 42
# combined with the side-chain block: information + information, different channels
sub gld_disto_w12_s42 --loss_mode ddg --distogram_weight 0.1 --sidechain_features --burial_features --seed 42
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
