#!/usr/bin/env bash
# Calibration factorial, 2^4 = 16 cells x 3 seeds = 48 runs.
#   A --wt_anchor_weight {0, 1.0}   attacks b_p
#   B --designed_weight  {1, 3}     attacks a_p
#   C --slope_weight     {0, w*}    attacks a_p directly  (Nissim's contribution)
#   D factor D: coil or unfolded-emb, DECIDED AT D0 BY THE W0 ABLATION
#
# Usage:
#   bash cluster_run/scripts/submit_factorial.sh --wave 1 --dflag "--flory_unfolded" --wstar 1.0
#   ... add --go to actually submit. DRY RUN IS THE DEFAULT.
set -euo pipefail
cd "$(dirname "$0")/../.."

WAVE=1; DFLAG="--flory_unfolded"; WSTAR="1.0"; GO=0; SEEDS="42"
while [ $# -gt 0 ]; do
  case "$1" in
    --wave) WAVE="$2"; shift 2;;
    --dflag) DFLAG="$2"; shift 2;;
    --wstar) WSTAR="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --go) GO=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# --- site rules, paid for in wasted runs ---
export WANDB_MODE=disabled          # 03/04/05 omit this and the run dies at wandb login
QOS_FLAG="--qos normal"             # normal has no per-account cap; keasar caps the lab at 8
GPU_SPEC="--gres=gpu:rtx_6000:1"    # NEVER by partition name: submit filter reroutes onto a 1080 -> OOM
EXCLUDE="--exclude=ise-pheno-01,ise-pheno-02,ise-pheno-03,ise-pheno-04,ise-pheno-05,ise-pheno-06,ise-pheno-07,ise-pheno-08,ise-pheno-09,ise-pheno-10,ise-pheno-11,ise-pheno-12"
PKG="cluster_run"
mkdir -p logs results eval_results

# D short label for the run tag, derived from the flag
case "$DFLAG" in
  *flory*) DLAB="coil";;
  *unfolded_emb*) DLAB="uemb";;
  "") DLAB="off";;
  *) DLAB="D";;
esac

# 16 cells: A B C D as 0/1.  Wave 1 = the 8 cells with D=0, wave 2 = D=1.
# Rationale in WAVE_PLAN.md: D is the only factor whose IDENTITY is still undecided,
# so every D=0 cell is safe to run before the W0 ablation lands.
submitted=0
for A in 0 1; do for B in 0 1; do for C in 0 1; do for D in 0 1; do
  CELLWAVE=$(( D + 1 ))
  [ "$CELLWAVE" -eq "$WAVE" ] || continue

  AW=$([ "$A" = 1 ] && echo "1.0" || echo "0")
  BW=$([ "$B" = 1 ] && echo "3"   || echo "1")
  CW=$([ "$C" = 1 ] && echo "$WSTAR" || echo "0")
  DD=$([ "$D" = 1 ] && echo "$DFLAG" || echo "")

  for SEED in $SEEDS; do
    RUN_TAG="p3_a${A}_d${B}_s${C}_D${D}_${DLAB}_seed${SEED}"
    MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"

    # idempotent: never double-submit a cell that already exists
    if [ -d "$MODEL_DIR" ]; then echo "SKIP (exists): ${RUN_TAG}"; continue; fi
    if squeue -u "$USER" -h -o "%j" | grep -qx "DeepEF_${RUN_TAG}"; then
      echo "SKIP (queued): ${RUN_TAG}"; continue; fi

    TRAIN_CMD="module load anaconda; source activate esm2_env_py38; \
export WANDB_MODE=disabled; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight ${AW} --designed_weight ${BW} --slope_weight ${CW} ${DD} \
  --val_frac 0.1 --epochs 15 --seed ${SEED} --run_tag ${RUN_TAG}"

    if [ "$GO" -eq 0 ]; then
      echo "[dry-run] ${RUN_TAG}  A=${AW} B=${BW} C=${CW} D='${DD}'"
      submitted=$((submitted+1)); continue
    fi

    JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" \
           $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 1-00:00:00 $EXCLUDE \
           --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
           --wrap "$TRAIN_CMD")
    TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"

    # A FINISHED TRAINING RUN IS NOT A RESULT: three 7h+ runs reached COMPLETED with no CSV.
    # The eval verifies its own output and fails loudly.
    EVAL_CMD="module load anaconda; source activate esm2_env_py38; \
export WANDB_MODE=disabled; mkdir -p eval_results; \
bash Megascale-fineTuning/run_calib_eval.sh '${TRAIN_LOG}' '${MODEL_DIR}' '${RUN_TAG}'; \
ls -s eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || { echo 'EVAL PRODUCED NO CSV'; exit 1; }"

    sbatch --job-name "ev_${RUN_TAG}" \
           $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE \
           --dependency "afterok:${JID}" \
           --output "logs/ev_${RUN_TAG}_%j.out" --error "logs/ev_${RUN_TAG}_%j.out" \
           --wrap "$EVAL_CMD" >/dev/null
    echo "submitted ${RUN_TAG}: train=${JID}  A=${AW} B=${BW} C=${CW} D='${DD}'"
    submitted=$((submitted+1))
  done
done; done; done; done

echo "---"
echo "wave ${WAVE}: ${submitted} run(s) $([ "$GO" -eq 0 ] && echo '(DRY RUN -- add --go to submit)' || echo submitted)"
echo "w* = ${WSTAR}; factor D = '${DFLAG}' (${DLAB})"
