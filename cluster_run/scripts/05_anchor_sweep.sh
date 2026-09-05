#!/bin/bash
# UNTESTED: requires cluster. Phase 3/4 (docs/BUILD.md B8, docs/COMPUTE_PLAN.md): the WT-anchor
# sweep -- the same calib_ctrl recipe with --wt_anchor_weight in {0.3, 1.0, 3.0}, everything else at
# the reference. Factor A attacks the per-protein offset b; the known cost is slope collapse, so the
# deliverable is a TRADE-OFF CURVE (std(b) and the slope distribution vs anchor weight), NOT a single
# maximum. Do NOT read pooled alone: a cell that improves pooled while destroying slope is not a win.
# --wt_anchor_weight and --designed_weight already exist in train.py; no patch needed for this sweep.
# SLURM header matches sbatch_wtanchor.sh; afterok eval chain via run_calib_eval.sh.
#
# SELF-CONTAINED NOTE: set REPO_ROOT to the shaharec/DeepPEF checkout (see README.md version marker).
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
PKG=${PKG:-"$(cd "$(dirname "$0")/.." && pwd)"}
cd "${REPO_ROOT}"

SEED=${SEED:-42}
PARTITION=${PARTITION:-rtx6000}; QOS=${QOS:-keasar}; GPU_SPEC=${GPU_SPEC:---gpus=1}
EXCLUDE=${EXCLUDE:---exclude cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,cs-cpu256-01}
QOS_FLAG=""; [ -n "$QOS" ] && QOS_FLAG="--qos $QOS"
mkdir -p logs

for W in 0.3 1.0 3.0; do
  RUN_TAG="anchor_w${W}_s${SEED}"
  MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"
  JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" \
         --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 3-00:00:00 $EXCLUDE \
         --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
         --wrap "module load anaconda; source activate esm2_env_py38; \
                 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
                 python Megascale-fineTuning/train.py \
                   --full_data --no_pretrain --no_freeze --loss_mode ddg \
                   --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
                   --wt_anchor_weight ${W} --designed_weight 1 \
                   --val_frac 0.1 --epochs 15 --seed ${SEED} --run_tag ${RUN_TAG}")
  TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"
  sbatch --job-name "eval_${RUN_TAG}" \
         --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE \
         --dependency "afterok:${JID}" \
         --output "logs/eval_${RUN_TAG}_%j.out" --error "logs/eval_${RUN_TAG}_%j.out" \
         --wrap "module load anaconda; source activate esm2_env_py38; \
                 bash Megascale-fineTuning/run_calib_eval.sh '${TRAIN_LOG}' '${MODEL_DIR}' '${RUN_TAG}'"
  echo "submitted anchor w=${W} seed=${SEED}: train=${JID}  (supervise: ${PKG}/code/monitor.py ${TRAIN_LOG})"
done
echo "Every cell reports (calib_diag.py): pooled, PP, abs-WT-dG PCC, std(b), slope dist"
echo "(min/med/max, count<0.3), and the same on the designed subset. Build the trade-off curve."
