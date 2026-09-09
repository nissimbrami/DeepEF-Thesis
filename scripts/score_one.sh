#!/bin/bash
# usage: sbatch scripts/score_one.sh <run_tag> [eval flags...]
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
T="$1"; shift
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1 && { echo "SKIP $T already scored"; exit 0; }
[ -f "${M}${T}/kf_all_epoch_14.pt" ] || { echo "SKIP $T no e14 ckpt"; exit 0; }
L=$(ls -t logs/*${T}_*.out 2>/dev/null | head -1)
[ -z "$L" ] && { echo "SKIP $T no log"; exit 0; }
echo "=== SCORING $T  flags:[$*] ==="
bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T" "$*"
F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
[ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING: $T"
