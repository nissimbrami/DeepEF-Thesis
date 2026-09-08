#!/bin/bash
# GOLDEN-LANE REFILLER. The autopilot cannot do this: it has NO sbatch call anywhere in it
# (verified: grep -c sbatch scripts/autopilot.py == 0) and it is stuck in S5_BUILD_INFO with
# cells_done 18 > cells_total 16, re-deciding D1 every 10 minutes forever.
#
# This script is idempotent and SAFE:
#   - it NEVER cancels or kills anything (no scancel in this file)
#   - it skips any arm already queued or already scored
#   - it only ADDS work when the lane has free capacity
# Run it from cron or from the loop. Priority order is justified by measured facts.
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"

QUEUED=$(squeue -u nissimb -h -o "%P" | grep -c rtx6000)
echo "golden lane currently holds $QUEUED jobs (running+pending)"
if [ "$QUEUED" -ge 16 ]; then echo "lane is deep enough; adding nothing"; exit 0; fi

sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "SUBMIT $N -> $J"
}

# --- seed replication of W12, the largest untested lever (3 seeds already queued; add 2) ---
sub gld_w12_s3 --loss_mode ddg --sidechain_features --burial_features --seed 3
sub gld_w12_s4 --loss_mode ddg --sidechain_features --burial_features --seed 4
# --- slope below 1.0: s overshoots 1.0 between e4 and e8, so the optimum is BELOW 1.0 ---
sub gld_slope0.7_s1 --loss_mode ddg --slope_weight 0.7 --seed 1
sub gld_slope0.5_s2 --loss_mode ddg --slope_weight 0.5 --seed 2
# --- the best combination arm, replicated ---
sub gld_slope_w5_s2 --loss_mode ddg --slope_weight 1.0 --burial_features --seed 2
echo "--- lane after refill: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
