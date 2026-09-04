#!/bin/bash
# Auto-eval for the calibration run: pick best epoch by VAL ddG-PCC (from the training log),
# score that checkpoint on the 28-protein test via the trusted evaluate.py path (with the affine
# sidecar applied), then print the per-epoch ddG curve via validation/score_runs.py.
# Usage: run_calib_eval.sh <train_log> <model_dir> <run_tag>
# Designed to be launched as a SLURM job with --dependency=afterok:<train_jobid>.
set -uo pipefail
cd "$(dirname "$0")/.."   # project root

TRAIN_LOG=${1:?train log path}
MODEL_DIR=${2:?model dir (contains kf_all_epoch_*.pt + affine.json)}
RUN_TAG=${3:-calib_combo}

# 1) best epoch by VAL overall ddG-PCC (validate() prints one line per epoch, in epoch order)
BEST_E=$(python3 - "$TRAIN_LOG" <<'PY'
import re, sys
t = open(sys.argv[1], errors='ignore').read()
m = re.findall(r'ddG PCC=(-?\d+\.\d+)\s+ddG PCC-PP=(-?\d+\.\d+)\s+ddG RMSE=(-?\d+\.\d+)', t)
if not m:
    print(-1)
else:
    best = max(range(len(m)), key=lambda i: float(m[i][0]))
    sys.stderr.write("VAL per-epoch (PCC/PP/RMSE):\n" +
                     "\n".join(f"  e{i}: {a}/{b}/{c}" for i,(a,b,c) in enumerate(m)) + "\n")
    sys.stderr.write(f"BEST val epoch = {best} (PCC={m[best][0]})\n")
    print(best)
PY
)
echo "best epoch (by val PCC) = ${BEST_E}"

# fallback: if the log had no val lines, use the highest saved epoch checkpoint
if [ "${BEST_E}" = "-1" ] || [ -z "${BEST_E}" ]; then
  BEST_E=$(ls "${MODEL_DIR}"/kf_all_epoch_*.pt 2>/dev/null | sed -E 's/.*epoch_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  echo "no val lines parsed; falling back to last checkpoint epoch = ${BEST_E}"
fi

CKPT="${MODEL_DIR}/kf_all_epoch_${BEST_E}.pt"
AFFINE="${MODEL_DIR}/affine.json"
if [ ! -f "${CKPT}" ]; then echo "ERROR: checkpoint ${CKPT} not found"; ls -1 "${MODEL_DIR}" || true; exit 1; fi

# 2) trusted 28-test eval (same defaults as the existing abl_*.csv: ds_type=pnas, no unstable/one
#    flags -> identical 28314-variant set). Affine applied via DEEPEF_AFFINE; length-norm OFF.
export WANDB_MODE=offline
if [ -f "${AFFINE}" ]; then export DEEPEF_AFFINE="${AFFINE}"; echo "applying affine ${AFFINE}"; fi
python Megascale-fineTuning/evaluate.py \
  --trained_model_path "${CKPT}" \
  --model_name "eval_results/abl_${RUN_TAG}_e${BEST_E}" \
  --readout sum --dg_length_norm none

# 3) score (per-epoch ddG curve for this run tag)
python validation/score_runs.py "${RUN_TAG}"

# 4) calibration diagnostic (Plot B): did lever 1 shrink the per-protein offset spread std(b)?
#    Run on the new run, save a run-specific PNG, then on the rand_scratch baseline for comparison.
echo "=== Plot B offset diagnostic — NEW RUN (${RUN_TAG} e${BEST_E}) ==="
python validation/diag_unfolded_offset.py "eval_results/abl_${RUN_TAG}_e${BEST_E}.csv" || true
cp -f eval_results/figures/plotB_unfolded_offset.png "eval_results/figures/plotB_${RUN_TAG}.png" 2>/dev/null || true
echo "=== Plot B offset diagnostic — BASELINE (rand_scratch e2) for comparison ==="
python validation/diag_unfolded_offset.py eval_results/abl_randscratch_e2.csv || true
echo "Compare the 'between-protein offset b: ... std=' line above: smaller std(b) on the new run = lever 1 worked."

echo "DONE: ${RUN_TAG} best epoch ${BEST_E} scored on 28-test -> eval_results/abl_${RUN_TAG}_e${BEST_E}.csv ; plot -> eval_results/figures/plotB_${RUN_TAG}.png"
