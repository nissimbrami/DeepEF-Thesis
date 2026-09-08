#!/bin/bash
# Score EVERY trained-to-epoch-14 run that has no CSV yet, on the CPU partition.
# Scoring is a forward pass over 28 proteins and never needed a GPU; DEVICE was hard-coded
# to 'cuda' with the fallback commented out, which is why 12 finished cells once showed 0 results.
# Verifies the ARTIFACT with ls after each cell -- run_calib_eval.sh prints "DONE ... -> file.csv"
# even after a crash, which is this project's signature failure mode.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=""
mkdir -p logs/cpueval
python3 -c "import pandas,torch" || { echo "ENV BROKEN"; exit 1; }
ok=0; fail=0; skip=0
for d in Megascale-fineTuning/models/*kf_*; do
  [ -f "$d/kf_all_epoch_14.pt" ] || continue
  T=$(basename "$d" | sed 's/.*light_attentionkf_//')
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then skip=$((skip+1)); continue; fi
  LOG=$(ls -t logs/${T}_*.out 2>/dev/null | head -1)
  [ -z "$LOG" ] && LOG=$(ls -t logs/gld_${T}_*.out 2>/dev/null | head -1)
  [ -z "$LOG" ] && { echo "SKIP $T (no train log)"; skip=$((skip+1)); continue; }
  echo "=== SCORING $T ==="
  bash Megascale-fineTuning/run_calib_eval.sh "$LOG" "$d" "$T" > logs/cpueval/${T}.log 2>&1
  if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "OK   $T"; ok=$((ok+1))
  else echo "FAIL $T (see logs/cpueval/${T}.log)"; fail=$((fail+1)); fi
done
echo "DONE: ok=$ok fail=$fail skip=$skip ; total factorial CSVs now $(ls eval_results/abl_p3_a*.csv 2>/dev/null|wc -l)"
