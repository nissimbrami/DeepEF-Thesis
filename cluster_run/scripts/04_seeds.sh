#!/bin/bash
# UNTESTED: requires cluster. Phase 2 (docs/COMPUTE_PLAN.md): five seeds of the reference to MEASURE
# sigma. On five GPUs = one run's wall-clock. Report mean +/- sigma for pooled and PP. sigma decides
# the Phase-3 seed count: sigma <= 0.01 -> 3 seeds per cell; sigma >= 0.02 -> 5. Free bonus: average
# the five predictions for a seed ensemble (report separately -- a deployment result, not a finding).
# Requires the --seed flag in train.py (apply code/add_seed.patch if not already present).
# SLURM header matches sbatch_wtanchor.sh; afterok eval chain via run_calib_eval.sh.
#
# SELF-CONTAINED NOTE: set REPO_ROOT to the shaharec/DeepPEF checkout (see README.md version marker).
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
PKG=${PKG:-"$(cd "$(dirname "$0")/.." && pwd)"}
cd "${REPO_ROOT}"

PARTITION=${PARTITION:-rtx6000}; QOS=${QOS:-keasar}; GPU_SPEC=${GPU_SPEC:---gpus=1}
EXCLUDE=${EXCLUDE:---exclude cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,cs-cpu256-01}
QOS_FLAG=""; [ -n "$QOS" ] && QOS_FLAG="--qos $QOS"
mkdir -p logs

for SEED in 42 1 2 3 4; do
  RUN_TAG="sigma_seed${SEED}"
  MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"
  JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" \
         --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 3-00:00:00 $EXCLUDE \
         --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
         --wrap "module load anaconda; source activate esm2_env_py38; \
                 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
                 python Megascale-fineTuning/train.py \
                   --full_data --no_pretrain --no_freeze --loss_mode ddg \
                   --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
                   --wt_anchor_weight 0 --designed_weight 1 \
                   --val_frac 0.1 --epochs 15 --seed ${SEED} --run_tag ${RUN_TAG}")
  TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"
  sbatch --job-name "eval_${RUN_TAG}" \
         --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE \
         --dependency "afterok:${JID}" \
         --output "logs/eval_${RUN_TAG}_%j.out" --error "logs/eval_${RUN_TAG}_%j.out" \
         --wrap "module load anaconda; source activate esm2_env_py38; \
                 bash Megascale-fineTuning/run_calib_eval.sh '${TRAIN_LOG}' '${MODEL_DIR}' '${RUN_TAG}'"
  echo "submitted seed ${SEED}: train=${JID}  (supervise: ${PKG}/code/monitor.py ${TRAIN_LOG})"
done
echo "After all 5 eval: compute mean +/- sigma of pooled and PP across seeds. Record sigma in docs/STATUS.md."
echo "QUEUE DISCIPLINE: keep all seeds of one config on the SAME GPU type; record the type with every result."
