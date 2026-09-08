#!/bin/bash
# Score the four information levers once each has epoch 14.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval
for T in gld_w7span4_s42 gld_u10bidir_s42 gld_w7edge_s42 gld_w5_dg_s42; do
  D=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_$T
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "SKIP $T (csv exists)"; continue; fi
  [ -f "$D/kf_all_epoch_14.pt" ] || { echo "NOTREADY $T"; continue; }
  LOG=$(ls -t logs/${T%_s42}_*.out 2>/dev/null | head -1)
  [ -z "$LOG" ] && { echo "SKIP $T (no log)"; continue; }
  echo "=== SCORING $T (log=$LOG) ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$LOG" "$D" "$T" > logs/cpueval/${T}.log 2>&1
  # Verify the ARTIFACT, not the success line.
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "OK   $T -> $(ls eval_results/abl_${T}_e*.csv)"; else echo "FAIL $T (see logs/cpueval/${T}.log)"; fi
done
