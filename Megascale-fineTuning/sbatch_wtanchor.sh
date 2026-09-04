#!/bin/bash
# WS-1 (paper plan): WT/absolute-dG ANCHOR calibration arm.
# Regime = the honest calib_ctrl recipe (random-init, full-data, no-freeze, ddg, val-split, affine)
# with pooled-corr OFF, plus the new --wt_anchor_weight (and optional designed-fold reweight).
# Isolates whether anchoring pred_dG(WT) to the experimental WT abs-dG calibrates the per-protein
# baseline/offset -> lifts pooled ddG PCC toward the ~0.80 oracle WITHOUT hurting per-protein PCC.
# Baseline to beat: calib_ctrl (anchor=0) PCC 0.606 / PP 0.711.
# Chains the trusted 28-test eval + offset diagnostic via run_calib_eval.sh (afterok dependency).
#
# Usage:  WT_ANCHOR_WEIGHT=0.3 DESIGNED_WEIGHT=1 RUN_TAG=wtanchor_w03 ./Megascale-fineTuning/sbatch_wtanchor.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

RUN_TAG=${RUN_TAG:-wtanchor}
EPOCHS=${EPOCHS:-15}
WT_ANCHOR_WEIGHT=${WT_ANCHOR_WEIGHT:-0.5}
DESIGNED_WEIGHT=${DESIGNED_WEIGHT:-1.0}
VAL_FRAC=${VAL_FRAC:-0.1}
# Partition/GPU overrides. Default = rtx6000 partition (keasar QOS, 4-GPU account cap).
# For extra capacity use the shared 'gpu' partition pinned to a cu117-SAFE GPU type (sm<=86):
#   PARTITION=gpu QOS= GPU_SPEC="--gres=gpu:rtx_3090:1" EXCLUDE="" ...
PARTITION=${PARTITION:-rtx6000}
QOS=${QOS:-keasar}
GPU_SPEC=${GPU_SPEC:---gpus=1}
EXCLUDE=${EXCLUDE:---exclude cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,cs-cpu256-01}
QOS_FLAG=""; [ -n "$QOS" ] && QOS_FLAG="--qos $QOS"

mkdir -p logs
# model dir for the eval step. NOTE: MODEL_NAME is fixed at train.py:46 BEFORE the --no_freeze
# override, so it stays 'PEM_fine_tuned-<base>'; train.py:47 appends 'kf'; :104 appends '_<tag>'.
MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"

JID=$(sbatch --parsable \
       --job-name "DeePEF_${RUN_TAG}" \
       --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 \
       --time 3-00:00:00 \
       $EXCLUDE \
       --output "logs/${RUN_TAG}_%j.out" \
       --error  "logs/${RUN_TAG}_%j.out" \
       --wrap "module load anaconda; source activate esm2_env_py38; \
               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
               python Megascale-fineTuning/train.py \
                 --full_data --no_pretrain --no_freeze --loss_mode ddg \
                 --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
                 --wt_anchor_weight ${WT_ANCHOR_WEIGHT} --designed_weight ${DESIGNED_WEIGHT} \
                 --val_frac ${VAL_FRAC} --epochs ${EPOCHS} --run_tag ${RUN_TAG}")
echo "submitted TRAIN ${RUN_TAG} job=${JID}: wt_anchor=${WT_ANCHOR_WEIGHT} designed=${DESIGNED_WEIGHT} val_frac=${VAL_FRAC} epochs=${EPOCHS}"

# auto-eval after training succeeds (interactive-class GPU not required; runs on rtx6000 briefly)
TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"
EJID=$(sbatch --parsable \
       --job-name "eval_${RUN_TAG}" \
       --partition $PARTITION $QOS_FLAG $GPU_SPEC --cpus-per-task=8 \
       --time 06:00:00 \
       $EXCLUDE \
       --dependency "afterok:${JID}" \
       --output "logs/eval_${RUN_TAG}_%j.out" \
       --error  "logs/eval_${RUN_TAG}_%j.out" \
       --wrap "module load anaconda; source activate esm2_env_py38; \
               bash Megascale-fineTuning/run_calib_eval.sh '${TRAIN_LOG}' '${MODEL_DIR}' '${RUN_TAG}'")
echo "submitted EVAL ${RUN_TAG} job=${EJID} (afterok:${JID}) -> eval_results/abl_${RUN_TAG}_e*.csv"
