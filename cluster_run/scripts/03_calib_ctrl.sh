#!/bin/bash
# UNTESTED: requires cluster -- this is the 3-4h reference training run; train.py + K50 data are
# cluster-only. Phase 1 (docs/COMPUTE_PLAN.md): reproduce calib_ctrl EXACTLY. Change nothing --
# not the mini-batch (leave the hardcoded 64), not the split, not the epochs. A reproduction with a
# modification is not a reproduction. GATE: pooled ~= 0.606, PP ~= 0.711 (from run_calib_eval.sh ->
# score_runs.py, NOT in-training validate()). Epoch-2 reproduction check >= 0.55 or stop early.
# SLURM header matches Megascale-fineTuning/sbatch_wtanchor.sh (afterok eval chain).
#
# SELF-CONTAINED NOTE: set REPO_ROOT to the shaharec/DeepPEF checkout (see README.md version marker).
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
PKG=${PKG:-"$(cd "$(dirname "$0")/.." && pwd)"}
cd "${REPO_ROOT}"

RUN_TAG=${RUN_TAG:-calib_ctrl_repro}
EPOCHS=${EPOCHS:-15}
VAL_FRAC=${VAL_FRAC:-0.1}
# SLURM (matches sbatch_wtanchor.sh). TODO(operator): confirm partition/QOS for this account.
PARTITION=${PARTITION:-rtx6000}
QOS=${QOS:-keasar}
GPU_SPEC=${GPU_SPEC:---gpus=1}
EXCLUDE=${EXCLUDE:---exclude cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,cs-cpu256-01}
QOS_FLAG=""; [ -n "$QOS" ] && QOS_FLAG="--qos $QOS"
mkdir -p logs
# MODEL_NAME is fixed at train.py:46 BEFORE the --no_freeze override, stays 'PEM_fine_tuned-<base>';
# train.py:47 appends 'kf'; :104 appends '_<tag>'. Derive, do not hardcode elsewhere.
MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"

JID=$(sbatch --parsable \
       --job-name "DeepEF_${RUN_TAG}" \
       --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 \
       --time 3-00:00:00 $EXCLUDE \
       --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
       --wrap "module load anaconda; source activate esm2_env_py38; \
               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
               python Megascale-fineTuning/train.py \
                 --full_data --no_pretrain --no_freeze --loss_mode ddg \
                 --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
                 --wt_anchor_weight 0 --designed_weight 1 \
                 --val_frac ${VAL_FRAC} --epochs ${EPOCHS} --seed 42 --run_tag ${RUN_TAG}")
echo "submitted TRAIN ${RUN_TAG} job=${JID}"
echo "SUPERVISE: python ${PKG}/code/monitor.py logs/${RUN_TAG}_${JID}.out --tag ${RUN_TAG}  (act on STOP:)"

TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"
EJID=$(sbatch --parsable \
       --job-name "eval_${RUN_TAG}" \
       --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 \
       --time 06:00:00 $EXCLUDE --dependency "afterok:${JID}" \
       --output "logs/eval_${RUN_TAG}_%j.out" --error "logs/eval_${RUN_TAG}_%j.out" \
       --wrap "module load anaconda; source activate esm2_env_py38; \
               bash Megascale-fineTuning/run_calib_eval.sh '${TRAIN_LOG}' '${MODEL_DIR}' '${RUN_TAG}'")
echo "submitted EVAL ${RUN_TAG} job=${EJID} (afterok:${JID})"
echo "GATE after eval: pooled ~= 0.606, PP ~= 0.711. Then feed the eval CSV to calib_diag.py."
