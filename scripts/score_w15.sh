#!/bin/bash
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
L=$(ls -t logs/pub_w15_s42_*.out | head -1)
echo "=== SCORING w15_s42 (best available epoch; run TIMEOUT at e11) log=$L ==="
bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}w15_s42" "w15_s42" "--w15_features"
F=$(ls eval_results/abl_w15_s42_e*.csv 2>/dev/null | tail -1)
[ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING: w15_s42"
