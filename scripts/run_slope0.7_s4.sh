#!/bin/bash
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
python Megascale-fineTuning/train.py --full_data --no_pretrain --no_freeze   --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1   --epochs 15 --loss_mode ddg --slope_weight 0.7 --seed 4 --run_tag slope0.7_s4
