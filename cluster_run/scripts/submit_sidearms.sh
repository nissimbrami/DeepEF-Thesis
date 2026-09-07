#!/usr/bin/env bash
# Side arms, decided by the W0 channel ablation (results/w0.json, 28 test proteins,
# calib_ctrl_repro2 e14):
#   r_noemb = 0.331  -> ProtT5 IS the offset channel; factor D := --unfolded_emb
#   r_coil  = 1.175  -> the coil RAISES var(E_u); it was demoted from the factorial
#
# Two arms, one seed each, on top of the reference config:
#   coil      the demoted lever, run once so the negative result is measured, not assumed
#   uemb_mean the interesting middle case: if zero hurts but mean helps, the embedding
#             was carrying global scale rather than per-residue identity
set -euo pipefail
cd "$(dirname "$0")/../.."
export WANDB_MODE=disabled
QOS_FLAG="--qos normal"; GPU_SPEC="--gres=gpu:rtx_6000:1"
EXCLUDE="--exclude=ise-pheno-01,ise-pheno-02,ise-pheno-03,ise-pheno-04,ise-pheno-05,ise-pheno-06,ise-pheno-07,ise-pheno-08,ise-pheno-09,ise-pheno-10,ise-pheno-11,ise-pheno-12"
GO=${1:-dry}
mkdir -p logs results eval_results

# --- ITEM 28: supervision (same contract as submit_factorial.sh)
SUPERVISE="${SUPERVISE:-1}"
SUP_SH="cluster_run/scripts/run_supervised.sh"
E2_SH="cluster_run/scripts/epoch2_check.sh"
DIAG_SH="cluster_run/scripts/eval_diag.sh"
STATE_JSON="results/state.json"
export MONITOR_RESULTS_DIR="results"
if [ "$SUPERVISE" = "1" ] && [ ! -f "$SUP_SH" ]; then
  echo "FATAL: SUPERVISE=1 but ${SUP_SH} is missing (AUTOPILOT S3)."; exit 3
fi
record_job () {   # $1 = job id, $2 = run tag, $3 = kind
  python3 - "$STATE_JSON" "$1" "$2" "$3" <<'PYSTATE'
import json, os, sys, time
path, jid, tag, kind = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
st = {}
if os.path.exists(path):
    try:
        st = json.load(open(path))
    except ValueError:
        st = {}
st.setdefault("submitted", [])
if not any(isinstance(e, dict) and str(e.get("job_id")) == str(jid) for e in st["submitted"]):
    st["submitted"].append({"job_id": str(jid), "run_tag": tag, "kind": kind,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
tmp = path + ".tmp"
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
json.dump(st, open(tmp, "w"), indent=2, sort_keys=True)
os.replace(tmp, path)
PYSTATE
}

submit () {
  RUN_TAG="$1"; EXTRA="$2"
  MODEL_DIR="./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}"
  if [ -d "$MODEL_DIR" ]; then echo "SKIP (exists): ${RUN_TAG}"; return; fi
  if squeue -u "$USER" -h -o "%j" | grep -qx "DeepEF_${RUN_TAG}"; then echo "SKIP (queued): ${RUN_TAG}"; return; fi
  if [ "$GO" != "--go" ]; then echo "[dry-run] ${RUN_TAG}  ${EXTRA}"; return; fi
  # ITEM 28: side-arm training runs UNDER run_supervised.sh (monitor.py attached
  # live), then the epoch-2 check. \$SLURM_JOB_ID must expand on the node, not here.
  SA_TRAIN="python Megascale-fineTuning/train.py --full_data --no_pretrain --no_freeze \
--loss_mode ddg --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
--wt_anchor_weight 0 --designed_weight 1 ${EXTRA} \
--val_frac 0.1 --epochs 15 --seed 42 --run_tag ${RUN_TAG}"
  if [ "$SUPERVISE" = "1" ]; then
    SA_BODY="SUPLOG=logs/${RUN_TAG}_\$SLURM_JOB_ID.out; \
export MONITOR_RESULTS_DIR=results; \
bash ${SUP_SH} '${RUN_TAG}' \"\$SUPLOG\" -- ${SA_TRAIN}; TRC=\$?; \
bash ${E2_SH} '${RUN_TAG}' \"\$SUPLOG\" results || true; exit \$TRC"
  else
    SA_BODY="${SA_TRAIN}"
  fi
  JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" \
     $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 1-00:00:00 $EXCLUDE \
     --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
     --wrap "module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled; \
             export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
             ${SA_BODY}")
  record_job "$JID" "$RUN_TAG" train
  EJID=$(sbatch --parsable --job-name "ev_${RUN_TAG}" $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE \
     --dependency "afterok:${JID}" --output "logs/ev_${RUN_TAG}_%j.out" --error "logs/ev_${RUN_TAG}_%j.out" \
     --wrap "module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled; mkdir -p eval_results; \
             bash Megascale-fineTuning/run_calib_eval.sh 'logs/${RUN_TAG}_${JID}.out' '${MODEL_DIR}' '${RUN_TAG}'; \
             ls eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || { echo 'EVAL PRODUCED NO CSV'; exit 1; }; \
             export MONITOR_RESULTS_DIR=results; bash ${DIAG_SH} '${RUN_TAG}' '' results || true")
  record_job "$EJID" "$RUN_TAG" eval
  echo "submitted ${RUN_TAG}: train=${JID}  ${EXTRA}"
}

submit "sa_coil_seed42"      "--flory_unfolded"
submit "sa_uembmean_seed42"  "--unfolded_emb mean"
echo "--- side arms $([ "$GO" = "--go" ] && echo submitted || echo '(dry run; pass --go)')"
