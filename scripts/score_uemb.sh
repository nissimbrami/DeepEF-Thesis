#!/bin/bash
#SBATCH --job-name=score_uemb
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/nissimb/DeepPEF/logs/score_uemb_%j.out
# sa_uembmean_seed42 trained with --unfolded_emb mean. That is NOT the D1 arm: D1 is
# --unfolded_emb zero, which collapses the model (t=11.80, a_p ~0.015). 'mean' replaces the
# unfolded embedding with the per-protein mean instead of zeroing it -- a third, milder setting
# that has never been scored. It tests whether the unfolded embedding must carry per-RESIDUE
# information or only per-PROTEIN information, which is a real question the D0/D1 contrast
# cannot answer.
# --unfolded_emb does not change feature width, so no EVAL_FLAGS are needed.
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
M=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_
T=sa_uembmean_seed42
L=$(ls -t logs/${T}_*.out 2>/dev/null | grep -v ev_ | head -1)
echo "=== $T   log=$L ==="
bash Megascale-fineTuning/run_calib_eval.sh "$L" "${M}${T}" "$T"
F=$(ls eval_results/abl_${T}_e*.csv 2>/dev/null | tail -1)
[ -n "$F" ] && echo "ARTIFACT OK: $F ($(wc -l < "$F") rows)" || echo "ARTIFACT MISSING for $T"
echo "SCORE_UEMB_DONE"
