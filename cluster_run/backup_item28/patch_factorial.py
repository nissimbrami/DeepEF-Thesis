#!/usr/bin/env python3
"""ITEM 28: patch submit_factorial.sh.

Line-anchored: each edit locates a UNIQUE line by exact content, then replaces a
known span. Refuses loudly on any mismatch and writes nothing.
Edits are applied BOTTOM-UP so earlier line indices stay valid.
"""
import sys

P = "cluster_run/scripts/submit_factorial.sh"
lines = open(P).read().split("\n")
if any("ITEM 28" in l for l in lines):
    print("ALREADY PATCHED -- refusing to double-apply")
    sys.exit(3)

BS = "\\"          # a single backslash, built without escaping ambiguity
DQ = '"'


def find_one(text):
    hits = [i for i, l in enumerate(lines) if l == text]
    if len(hits) != 1:
        raise SystemExit("ANCHOR NOT UNIQUE (%d hits): %r" % (len(hits), text))
    return hits[0]


# ---------------------------------------------------------------- A4 (bottom)
i = find_one('    sbatch --job-name "ev_${RUN_TAG}" ' + BS)
assert lines[i + 4] == '           --wrap "$EVAL_CMD" >/dev/null', repr(lines[i + 4])
a4 = [
    '    EJID=$(sbatch --parsable --job-name "ev_${RUN_TAG}" ' + BS,
    '           $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE ' + BS,
    '           --dependency "afterok:${JID}" ' + BS,
    '           --output "logs/ev_${RUN_TAG}_%j.out" --error "logs/ev_${RUN_TAG}_%j.out" ' + BS,
    '           --wrap "$EVAL_CMD")',
    '''    python3 - "$STATE_JSON" "$EJID" "$RUN_TAG" <<'PYSTATE2' '''.rstrip(),
    'import json, os, sys, time',
    'path, jid, tag = sys.argv[1], sys.argv[2], sys.argv[3]',
    'st = json.load(open(path)) if os.path.exists(path) else {}',
    'st.setdefault("submitted", [])',
    'if not any(isinstance(e, dict) and str(e.get("job_id")) == str(jid) for e in st["submitted"]):',
    '    st["submitted"].append({"job_id": str(jid), "run_tag": tag, "kind": "eval",',
    '                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})',
    'tmp = path + ".tmp"',
    'json.dump(st, open(tmp, "w"), indent=2, sort_keys=True)',
    'os.replace(tmp, path)',
    'PYSTATE2',
]
lines[i:i + 5] = a4

# ---------------------------------------------------------------- A3
a3_anchor = ("ls -s eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || "
             "{ echo 'EVAL PRODUCED NO CSV'; exit 1; }" + DQ)
i = find_one(a3_anchor)
lines[i:i + 1] = [
    ("ls -s eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || "
     "{ echo 'EVAL PRODUCED NO CSV'; exit 1; }; " + BS),
    ("export MONITOR_RESULTS_DIR=results; bash ${DIAG_SH} '${RUN_TAG}' '' results || true" + DQ),
]

# ---------------------------------------------------------------- A2
i = find_one('    JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" ' + BS)
assert lines[i + 3] == '           --wrap "$TRAIN_CMD")', repr(lines[i + 3])
assert lines[i + 4] == '    TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"', repr(lines[i + 4])
a2 = [
    '    # ITEM 28: the training command runs UNDER run_supervised.sh, which follows the',
    '    # live log with monitor.py (stop rules armed DURING the run, not after it), then',
    '    # runs the epoch-2 reproduction check. ' + BS + '$SLURM_JOB_ID is escaped on purpose: it',
    '    # must expand on the compute node, not here -- JID does not exist yet.',
    '    if [ "$SUPERVISE" = "1" ]; then',
    '      WRAPPED="module load anaconda; source activate esm2_env_py38; ' + BS,
    'export MONITOR_RESULTS_DIR=results; ' + BS,
    'SUPLOG=logs/${RUN_TAG}_' + BS + '$SLURM_JOB_ID.out; ' + BS,
    ('bash ${SUP_SH} ' + "'${RUN_TAG}' " + BS + DQ + BS + '$SUPLOG' + BS + DQ +
     ' -- ${TRAIN_CMD}; TRC=' + BS + '$?; ' + BS),
    ('bash ${E2_SH} ' + "'${RUN_TAG}' " + BS + DQ + BS + '$SUPLOG' + BS + DQ +
     ' results || true; ' + BS),
    'exit ' + BS + '$TRC"',
    '    else',
    '      WRAPPED="$TRAIN_CMD"',
    '    fi',
    '',
    '    JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" ' + BS,
    '           $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 1-00:00:00 $EXCLUDE ' + BS,
    '           --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" ' + BS,
    '           --wrap "$WRAPPED")',
    '    TRAIN_LOG="logs/${RUN_TAG}_${JID}.out"',
    '',
    '    # ITEM 28: the cancellation allowlist. autodebug.py may cancel ONLY job ids',
    '    # recorded here. Written BEFORE the eval submit so a crash between the two',
    '    # still leaves the train job cancellable.',
    '''    python3 - "$STATE_JSON" "$JID" "$RUN_TAG" <<'PYSTATE' '''.rstrip(),
    'import json, os, sys, time',
    'path, jid, tag = sys.argv[1], sys.argv[2], sys.argv[3]',
    'st = {}',
    'if os.path.exists(path):',
    '    try:',
    '        st = json.load(open(path))',
    '    except ValueError:',
    '        st = {}',
    'st.setdefault("submitted", [])',
    'if not any(isinstance(e, dict) and str(e.get("job_id")) == str(jid) for e in st["submitted"]):',
    '    st["submitted"].append({"job_id": str(jid), "run_tag": tag, "kind": "train",',
    '                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})',
    'tmp = path + ".tmp"',
    'os.makedirs(os.path.dirname(path) or ".", exist_ok=True)',
    'json.dump(st, open(tmp, "w"), indent=2, sort_keys=True)',
    'os.replace(tmp, path)',
    'PYSTATE',
]
lines[i:i + 5] = a2

# ---------------------------------------------------------------- A1 (top)
i = find_one("mkdir -p logs results eval_results")
assert lines[i - 1] == 'PKG="cluster_run"', repr(lines[i - 1])
a1 = [
    '',
    '# --- ITEM 28: supervision. AUTOPILOT S3 requires monitor.py on every job, an',
    '# epoch-2 reproduction check, and calib_diag on every evaluation.',
    'SUPERVISE="${SUPERVISE:-1}"                       # SUPERVISE=0 restores the old behaviour exactly',
    'SUP_SH="cluster_run/scripts/run_supervised.sh"',
    'E2_SH="cluster_run/scripts/epoch2_check.sh"',
    'DIAG_SH="cluster_run/scripts/eval_diag.sh"',
    'export MONITOR_RESULTS_DIR="results"',
    'if [ "$SUPERVISE" = "1" ] && [ ! -f "$SUP_SH" ]; then',
    '  echo "FATAL: SUPERVISE=1 but ${SUP_SH} is missing. AUTOPILOT S3 requires monitor.py"',
    '  echo "       on every job. Copy the ITEM 28 files in, or run with SUPERVISE=0 and"',
    '  echo "       record that the wave ran unmonitored."',
    '  exit 3',
    'fi',
    'STATE_JSON="results/state.json"',
]
lines[i + 1:i + 1] = a1

open(P, "w").write("\n".join(lines))
print("submit_factorial.sh PATCHED (A1 A2 A3 A4)")
