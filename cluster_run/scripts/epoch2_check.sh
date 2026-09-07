#!/usr/bin/env bash
# epoch2_check.sh -- the AUTOPILOT S3 "epoch-2 reproduction check".
#
# WHAT IT CHECKS: at the end of epoch 2 a run should be reproducing the reference
# trajectory, not diverging. Cheap: it reads the training log only, no GPU, no data.
# It stops a doomed 6-hour job at hour one (UNFOLDED_AUDIT_AND_BUDGET B4).
#
# It NEVER cancels. It writes a verdict file; autodebug/the operator decides.
#
# Usage: bash cluster_run/scripts/epoch2_check.sh <RUN_TAG> <TRAIN_LOG> [RESULTS_DIR]
#
# Verdict -> results/<tag>_epoch2.json  {"verdict": PASS|FAIL|UNKNOWN, ...}
#   PASS     epoch-2 val ddG PCC-PP >= E2_PP_MIN  (default 0.30, same as monitor M4)
#   FAIL     below it -> the cell is bad. THIS IS DATA, NOT A FAULT: class is
#            "model-data-not-fault" so autodebug records it and never retries.
#   UNKNOWN  epoch 2 not reached yet, or the log has no parseable val line.
set -uo pipefail
RUN_TAG=${1:?run tag}
TRAIN_LOG=${2:?train log}
RESULTS_DIR=${3:-${MONITOR_RESULTS_DIR:-results}}
E2_PP_MIN=${E2_PP_MIN:-0.30}
E2_EPOCH=${E2_EPOCH:-2}
mkdir -p "$RESULTS_DIR"

python3 - "$RUN_TAG" "$TRAIN_LOG" "$RESULTS_DIR" "$E2_PP_MIN" "$E2_EPOCH" <<'PY'
import json, os, re, sys
tag, log, rdir, pp_min, e2 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
out = os.path.join(rdir, tag + "_epoch2.json")

txt = ""
if os.path.exists(log):
    txt = open(log, errors="ignore").read()

# Same regex family run_calib_eval.sh uses to pick the best epoch: validate() emits
# one line per epoch IN EPOCH ORDER, so index i == epoch i.
rows = re.findall(r'ddG PCC=(-?\d+\.\d+)\s+ddG PCC-PP=(-?\d+\.\d+)\s+ddG RMSE=(-?\d+\.\d+)', txt)

rec = {"run_tag": tag, "epoch": e2, "pp_min": pp_min,
       "epochs_parsed": len(rows), "verdict": "UNKNOWN",
       "class": "unknown", "pcc": None, "pp": None, "rmse": None}

if len(rows) > e2:
    pcc, pp, rmse = (float(x) for x in rows[e2])
    rec.update(pcc=pcc, pp=pp, rmse=rmse)
    if pp >= pp_min:
        rec["verdict"], rec["class"] = "PASS", "ok"
        rec["note"] = "epoch-%d val ddG PCC-PP %.4f >= %.2f" % (e2, pp, pp_min)
    else:
        # AUTOPILOT section 2: monitor M4 row. Not a fault -- the cell is bad.
        rec["verdict"], rec["class"] = "FAIL", "model-data-not-fault"
        rec["note"] = ("epoch-%d val ddG PCC-PP %.4f < %.2f -- BAD CELL, record it, "
                       "DO NOT RETRY (AUTOPILOT s2 M4 row, max retries 0)" % (e2, pp, pp_min))
else:
    rec["note"] = "epoch %d not reached (%d epoch lines parsed)" % (e2, len(rows))

with open(out, "w") as fh:
    json.dump(rec, fh, indent=2)
print("[epoch2] %s: %s (%s) -> %s" % (tag, rec["verdict"], rec.get("note", ""), out))
PY
# Always exit 0: a FAIL here is a measured result, not a job failure.
exit 0
