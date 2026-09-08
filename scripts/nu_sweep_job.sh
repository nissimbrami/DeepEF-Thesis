#!/bin/bash
#SBATCH --job-name=nu_sweep
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/nissimb/DeepPEF/logs/nu_sweep_%j.out
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
CK=Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/kf_all_epoch_14.pt
mkdir -p results/08_data
echo "=== nu x b sweep, frozen checkpoint, 2K5H reference CORRECTED ==="
python -u scripts/w0_dg.py --ckpt "$CK" \
  --nu 0.5 0.55 0.588 0.62 0.65 \
  --b 4.97 5.82 \
  --ref_fix 2K5H=4.805470 \
  --out results/08_data/nu_sweep.json
echo "SWEEP_EXIT=$?"
echo "=== same grid, 2K5H reference RAW (sensitivity check) ==="
python -u scripts/w0_dg.py --ckpt "$CK" \
  --nu 0.5 0.55 0.588 0.62 0.65 \
  --b 4.97 5.82 \
  --out results/08_data/nu_sweep_raw2k5h.json
echo "RAW_EXIT=$?"
echo ALL_DONE
