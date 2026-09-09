#!/bin/bash
# WAVE 10 -- deepen the queue so cards never idle. 34 arms are already pending behind 8 running
# ones; SLURM starts each new arm the instant a card frees, with no intervention. Adding more is
# free automation, not waste: the only cost of a deeper queue is that low-priority arms wait.
#
# EVERY ARM HERE IS JUSTIFIED BY A MEASUREMENT MADE TONIGHT:
#
# 1. SEED REPLICATION OF W12 AND HSE. The slope lever looked proven at n=1 (+0.058) and died at
#    n=4 (+0.0179 against a seed sd of 0.0344), because seed 42 was an outlier: s=1.053 while
#    seeds 1,2,3 gave 0.515-0.532. Any arm read at n=1 is therefore untrustworthy by default.
#    W12 and HSE are the two information levers; give them the seeds up front rather than
#    discovering the problem after the fact.
#
# 2. ANCHOR BELOW 0.3. The anchor sweep is monotone decreasing -- corr(weight, pooled) = -0.999
#    across w = 0.3, 1.0, 3.0 -- so its best tested weight is the SMALLEST tried and the optimum
#    was never bracketed. w = 0.1 and 0.03 are the untested side of the curve.
#
# 3. SLOPE BELOW 0.5. Same logic from the other direction: s must reach 1.0 for perfect
#    calibration, and three of four slope-1.0 seeds land at s ~ 0.52, i.e. far UNDER. If the
#    lever has any interior optimum it is not at 1.0.
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
# --- W12 seeds 3 and 4: the information lever, replicated before it is read ---
sub gld_w12_s3 --loss_mode ddg --sidechain_features --burial_features --seed 3
sub gld_w12_s4 --loss_mode ddg --sidechain_features --burial_features --seed 4
# --- HSE seeds 2 and 3: direction-aware burial, same treatment ---
sub gld_hse_ddg_s2 --loss_mode ddg --burial_features --burial_mode hse --seed 2
sub gld_hse_ddg_s3 --loss_mode ddg --burial_features --burial_mode hse --seed 3
# --- anchor BELOW 0.3: the untested side of a monotone sweep ---
sub gld_anchor0.1_s42  --loss_mode ddg --wt_anchor_weight 0.1  --seed 42
sub gld_anchor0.03_s42 --loss_mode ddg --wt_anchor_weight 0.03 --seed 42
sub gld_anchor0.1_s1   --loss_mode ddg --wt_anchor_weight 0.1  --seed 1
# --- slope below 0.5, since three of four seeds sit at s ~ 0.52 ---
sub gld_slope0.2_s42 --loss_mode ddg --slope_weight 0.2 --seed 42
echo "--- lane now: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
