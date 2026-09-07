#!/bin/bash
#SBATCH --job-name=seqabl
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/seqabl_%j.out
set -uo pipefail
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
CK=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/kf_all_epoch_14.pt
mkdir -p results/seqabl
for M in none perm collapse collapse_oh perm_emb perm_oh; do
  echo "=== MODE $M $(date) ==="
  python scripts/seqabl.py --mode $M --seed 42 --ckpt "$CK" --out "results/seqabl/abl_${M}_s42" 2>&1 | tail -5
  if [ -f "./results/seqabl/abl_${M}_s42.csv" ]; then
     echo "ARTIFACT OK $M $(wc -l < ./results/seqabl/abl_${M}_s42.csv) lines"
  else
     echo "ARTIFACT MISSING $M"; ls -la results/seqabl/ ; fi
done
# second permutation seed to separate perm effect from seed noise
python scripts/seqabl.py --mode perm --seed 7 --ckpt "$CK" --out "results/seqabl/abl_perm_s7" 2>&1 | tail -3
echo "ALLDONE $(date)"; ls -la results/seqabl/
