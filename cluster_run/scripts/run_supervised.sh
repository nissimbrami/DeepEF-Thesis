#!/usr/bin/env bash
# run_supervised.sh -- attach monitor.py to a live training run (AUTOPILOT S3 "Per run").
#
# WHY THIS EXISTS: monitor.py reads a log to EOF and exits. A training log is not at EOF
# until the job ends, so `python monitor.py logs/x.out` on a live job either blocks on a
# fifo or parses an empty file. This wrapper runs training and follows its log in parallel,
# so the stop rules are armed DURING the 6-hour run, not after it.
#
# Usage (inside an sbatch --wrap, project root as cwd):
#   bash cluster_run/scripts/run_supervised.sh <RUN_TAG> <TRAIN_LOG> -- <train command...>
#
# Contract:
#   - stdout/stderr of the train command go to the caller's SLURM log AND to $TRAIN_LOG.
#   - monitor writes results/<RUN_TAG>_monitor.jsonl
#   - a STOP writes results/<RUN_TAG>_monitor.stop  (rule + message + classification)
#   - EXIT CODE IS ALWAYS THE TRAINING EXIT CODE. A monitor STOP never fails the job:
#     the eval is chained --dependency afterok, and an M4/M5 stop is DATA (a bad cell we
#     still want scored), not a fault. autodebug.py reads the .stop file, not the exit code.
set -uo pipefail

RUN_TAG=${1:?run tag}; shift
TRAIN_LOG=${1:?train log path}; shift
if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -eq 0 ]; then echo "[supervise] no train command given"; exit 2; fi

MON_PY="${MONITOR_PY:-cluster_run/scripts/monitor.py}"
RESULTS_DIR="${MONITOR_RESULTS_DIR:-results}"
export MONITOR_RESULTS_DIR="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR" "$(dirname "$TRAIN_LOG")"

STOP_FILE="${RESULTS_DIR}/${RUN_TAG}_monitor.stop"
MON_OUT="${RESULTS_DIR}/${RUN_TAG}_monitor.out"
rm -f "$STOP_FILE"

# ---- KILL-SAFETY -----------------------------------------------------------
# This script NEVER cancels anything. It has no scancel path by construction.
# A monitor STOP is recorded and the run is allowed to finish; cancellation is a
# human/autodebug decision, and autodebug gates every scancel on the state.json
# allowlist. 48 calibration runs and 3 pilots are live.
#
# The one `pkill` below is `pkill -P "$MON_PID"`: it reaps THIS supervisor's own
# log-follower children (tail | python) by parent PID, inside this job's own
# process tree. It cannot reach another job, and there is no bare `pkill name`
# anywhere in this file.
# ----------------------------------------------------------------------------

if [ ! -f "$MON_PY" ]; then
  echo "[supervise] WARNING: monitor.py not found at ${MON_PY}; running UNSUPERVISED"
  "$@" 2>&1 | tee -a "$TRAIN_LOG"
  exit "${PIPESTATUS[0]}"
fi

# Write the .stop file from whatever monitor.py printed. Kept SEPARATE from the
# follower so it still runs when the follower is killed: an M5 that is detected but
# never recorded is the same as an M5 that was missed, and M4/M5 are the results
# this whole build exists to measure.
write_stop_file () {
  grep -q '^STOP \[' "$MON_OUT" 2>/dev/null || return 0
  [ -f "$STOP_FILE" ] && return 0                      # already written
  line=$(grep -m1 '^STOP \[' "$MON_OUT")
  rule=$(printf '%s' "$line" | sed -E 's/^STOP \[([A-Z0-9]+)\].*/\1/')
  case "$rule" in
    # AUTOPILOT section 2: these two are the failure mode UNDER STUDY.
    M4|M5) klass="model-data-not-fault";;
    *)     klass="fault-candidate";;
  esac
  python3 - "$STOP_FILE" "$RUN_TAG" "$rule" "$klass" "$line" <<'PYSTOP'
import json, sys
path, tag, rule, klass, line = sys.argv[1:6]
json.dump({"run_tag": tag, "rule": rule, "class": klass, "line": line},
          open(path, "w"))
PYSTOP
  echo "[supervise] monitor STOP [$rule] class=$klass -> $STOP_FILE"
}

# Follow the log from byte 0. The follower is stopped explicitly once training
# ends (see below); `--pid` is NOT used to bound it, because that PID would be
# this supervisor, which outlives training.
: > "$TRAIN_LOG"
( tail -F -n +1 "$TRAIN_LOG" 2>/dev/null | python3 "$MON_PY" - "$RUN_TAG" ) \
    > "$MON_OUT" 2>&1 &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true; write_stop_file' EXIT

"$@" 2>&1 | tee -a "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}

sleep 5                      # let the follower drain the tail of the log
# Kill the whole follower process group (tail AND python); killing only $! leaves
# a stray `tail -F` behind on the compute node.
pkill -P "$MON_PID" 2>/dev/null || true
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
write_stop_file

echo "[supervise] ${RUN_TAG}: train rc=${TRAIN_RC}; monitor jsonl=${RESULTS_DIR}/${RUN_TAG}_monitor.jsonl"
[ -f "$STOP_FILE" ] && cat "$STOP_FILE"
exit "$TRAIN_RC"
