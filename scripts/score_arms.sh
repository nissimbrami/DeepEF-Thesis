#!/bin/bash
# Score the dG arm and the slope arm on CPU. Scores epoch 14 EXPLICITLY (not "best val epoch"):
# these two arms are hypothesis tests at a fixed budget, and val-PCC selection would pick a
# different epoch per arm and confound the comparison.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval

for T in gld_dg_coil_s42 gld_slope1.0_s42; do
  D="Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${T}"
  E=${1:-14}
  CKPT="$D/kf_all_epoch_${E}.pt"
  OUT="eval_results/abl_${T}_e${E}"
  if [ ! -f "$CKPT" ]; then echo "MISSING $CKPT"; continue; fi
  if [ -f "${OUT}.csv" ]; then echo "SKIP $T (csv exists)"; continue; fi
  # No affine.json in either dir -> DEEPEF_AFFINE deliberately unset, raw predictions.
  echo "=== SCORING $T epoch $E ==="
  python Megascale-fineTuning/evaluate.py \
    --trained_model_path "$CKPT" \
    --model_name "$OUT" \
    --readout sum --dg_length_norm none > logs/cpueval/${T}_e${E}.log 2>&1
  # Verify the ARTIFACT, never the success line.
  if [ -f "${OUT}.csv" ]; then echo "OK   $T -> ${OUT}.csv ($(wc -l < ${OUT}.csv) rows)";
  else echo "FAIL $T (see logs/cpueval/${T}_e${E}.log)"; tail -15 logs/cpueval/${T}_e${E}.log; fi
done
