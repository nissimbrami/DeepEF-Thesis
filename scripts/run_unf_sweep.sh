#!/bin/bash
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
CK=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/kf_all_epoch_14.pt
echo "=== SCALE 1.0 (raw Angstrom, what w0_dg.py used) ==="
python -u scripts/unfolded_sweep.py --ckpt "$CK" --coord_scale 1.0 --out results/unfolded_sweep_scale1.json --part all
echo "=== SCALE 0.1 (train.normalize_batch, what the model was TRAINED on) ==="
python -u scripts/unfolded_sweep.py --ckpt "$CK" --coord_scale 0.1 --out results/unfolded_sweep_scale01.json --part all
echo "ALL DONE"
