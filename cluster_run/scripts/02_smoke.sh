#!/bin/bash
# UNTESTED: requires cluster -- needs train.py + the K50 training data, neither of which exists on
# the build machine. Phase 1 pre-run gate: run the 8 preflight assertions, then a 1-epoch
# --full_data smoke with the EXACT calib_ctrl flags. ABORT on any preflight failure -- a stopped
# run is cheaper than a finished wrong one (monitor Rule 6, docs/COMPUTE_PLAN.md).
#
# SELF-CONTAINED NOTE: set REPO_ROOT to the shaharec/DeepPEF checkout (see README.md version marker).
# This script invokes cluster_run/code/preflight.py and cluster_run/code/monitor.py by path.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
PKG=${PKG:-"$(cd "$(dirname "$0")/.." && pwd)"}   # -> cluster_run/
cd "${REPO_ROOT}"

module load anaconda 2>/dev/null || true
source activate esm2_env_py38 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
mkdir -p logs

echo "==== preflight (8 checks; 1,2,5,7 are the cluster-only ones) ===="
python "${PKG}/code/preflight.py" || { echo "PREFLIGHT FAILED -- aborting."; exit 1; }

echo "==== 1-epoch smoke with the EXACT calib_ctrl flags (+ --epochs 1) ===="
# NOTE: --full_data bypasses k-fold (do NOT use --max_folds 1). Mini-batch is left at the hardcoded
# 64 (train.py:40) -- this is the reference regime; measure peak VRAM here before the long run.
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight 0 --designed_weight 1 \
  --val_frac 0.1 --epochs 1 --run_tag smoke_1ep 2>&1 | tee logs/smoke_1ep.log

echo "==== monitor the smoke log (should be clean at 1 epoch) ===="
python "${PKG}/code/monitor.py" logs/smoke_1ep.log --tag smoke_1ep || true
echo "==== DONE: smoke. If preflight+monitor are clean, proceed to scripts/03_calib_ctrl.sh ===="
