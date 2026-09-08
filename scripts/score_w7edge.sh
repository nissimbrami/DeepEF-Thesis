#!/bin/bash
#SBATCH --job-name=scorew7e
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/scorew7e_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
D=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_gld_w7edge_s42
export DEEPEF_AFFINE=$D/affine.json
python scripts/eval_w7edge.py \
  --trained_model_path $D/kf_all_epoch_14.pt \
  --model_name "eval_results/abl_gld_w7edge_s42_e14" \
  --readout sum --dg_length_norm none
echo "--- artifact check ---"
ls -l eval_results/abl_gld_w7edge_s42_e14.csv 2>/dev/null && echo "OK csv exists" || echo "FAIL no csv"
