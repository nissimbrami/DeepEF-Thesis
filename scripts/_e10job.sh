#!/bin/bash
#SBATCH --job-name=e10canon
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/nissimb/DeepPEF/logs/cpueval/e10canon_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval
# Canonical (val-selected) epoch for gld_slope1.0_s42 is e10; e12 ties it at 3dp, score both.
T=gld_slope1.0_s42
D="Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${T}"
for E in 10 12; do
  CKPT="$D/kf_all_epoch_${E}.pt"
  OUT="eval_results/abl_${T}_e${E}"
  if [ -f "${OUT}.csv" ]; then echo "SKIP $T e$E (csv exists)"; continue; fi
  echo "=== SCORING $T epoch $E ==="
  python Megascale-fineTuning/evaluate.py \
    --trained_model_path "$CKPT" \
    --model_name "$OUT" \
    --readout sum --dg_length_norm none > logs/cpueval/${T}_e${E}.log 2>&1
  if [ -f "${OUT}.csv" ]; then echo "OK   $T e$E -> ${OUT}.csv ($(wc -l < ${OUT}.csv) rows)";
  else echo "FAIL $T e$E"; tail -15 logs/cpueval/${T}_e${E}.log; fi
done
