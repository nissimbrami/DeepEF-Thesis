#!/bin/bash
# Score completed factorial cells on the LOGIN NODE (CPU).
#
# Scoring is a forward pass over 28 proteins and never needed a GPU. Three things had to be
# fixed before this could work at all, and each one alone produced ZERO CSVs from 12
# fully-trained cells:
#   1. evaluate.py hard-coded DEVICE='cuda' with the CPU fallback commented out.
#   2. train_utils.load_checkpoint assumed a wrapper dict, so the per-epoch BARE state_dict
#      saves raised KeyError 'model_state_dict'.
#   3. run_calib_eval.sh invokes `python3`, which resolves to the SYSTEM python (no pandas)
#      unless the conda env's bin is FIRST on PATH. `conda activate` cannot be used here
#      because a detached shell has no conda init.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval
echo "python3 resolves to: $(which python3)"
python3 -c "import pandas, torch; print('deps OK: pandas', pandas.__version__, 'torch', torch.__version__)" || { echo "ENV BROKEN - abort"; exit 1; }
for d in Megascale-fineTuning/models/*kf_p3_*; do
  [ -f "$d/kf_all_epoch_14.pt" ] || continue
  T=$(basename "$d" | sed 's/.*light_attentionkf_//')
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "SKIP $T (csv exists)"; continue; fi
  LOG=$(ls -t logs/${T}_*.out 2>/dev/null | head -1)
  [ -z "$LOG" ] && { echo "SKIP $T (no train log)"; continue; }
  echo "=== SCORING $T ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$LOG" "$d" "$T" > logs/cpueval/${T}.log 2>&1
  # Verify the ARTIFACT, never the success message: run_calib_eval.sh prints
  # "DONE ... -> abl_<tag>.csv" even after a crash, with no CSV on disk.
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "OK   $T"; else echo "FAIL $T (see logs/cpueval/${T}.log)"; fi
done
echo "DONE: $(ls eval_results/abl_p3_a*.csv 2>/dev/null | wc -l) factorial CSVs on disk"
