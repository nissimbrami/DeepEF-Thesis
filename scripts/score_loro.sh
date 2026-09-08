#!/bin/bash
# Score BOTH LORO arms on CPU, then run the tryptophan readout.
# Arms differ in exactly one flag: --aa_descriptors none vs mordred_pca16.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval
for T in loroW_onehot_s42 loroW_desc_s42; do
  d="Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${T}"
  # pick the highest epoch checkpoint actually on disk
  E=$(ls $d/kf_all_epoch_*.pt 2>/dev/null | sed 's/.*epoch_//;s/\.pt//' | sort -n | tail -1)
  [ -z "$E" ] && { echo "SKIP $T (no ckpt)"; continue; }
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "SKIP $T (csv exists)"; continue; fi
  LOG=$(ls -t logs/gld_${T%_s42}*.out logs/${T}_*.out 2>/dev/null | head -1)
  [ -z "$LOG" ] && { echo "SKIP $T (no train log)"; continue; }
  echo "=== SCORING $T (epoch $E) log=$LOG ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$LOG" "$d" "$T" > logs/cpueval/${T}.log 2>&1
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "OK   $T"; else echo "FAIL $T (see logs/cpueval/${T}.log)"; fi
done
ls -la eval_results/abl_loroW_*.csv 2>/dev/null
