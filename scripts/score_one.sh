#!/bin/bash
#SBATCH --job-name=score1
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/score1_%j.out
# Score ONE run tag, passed as $1. Verifies the ARTIFACT, not the success line.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
mkdir -p logs/cpueval
T=${1:?run tag}
D=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_$T
if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "SKIP $T (csv exists)"; exit 0; fi
[ -f "$D/kf_all_epoch_14.pt" ] || { echo "NOTREADY $T"; exit 1; }
LOG=$(ls -t logs/${T%_s42}_*.out 2>/dev/null | head -1)
echo "=== SCORING $T (log=$LOG) at $(date +%H:%M) ==="
bash Megascale-fineTuning/run_calib_eval.sh "$LOG" "$D" "$T" > logs/cpueval/${T}.log 2>&1
if ls eval_results/abl_${T}_e*.csv >/dev/null 2>&1; then echo "OK   $T -> $(ls eval_results/abl_${T}_e*.csv)"; else echo "FAIL $T"; fi
echo "DONE $(date +%H:%M)"
