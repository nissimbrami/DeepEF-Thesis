#!/usr/bin/env python3
"""ITEM 28: patch submit_sidearms.sh. Line-anchored, bottom-up, refuses on mismatch."""
import sys

P = "cluster_run/scripts/submit_sidearms.sh"
lines = open(P).read().split("\n")
if any("ITEM 28" in l for l in lines):
    print("ALREADY PATCHED -- refusing to double-apply")
    sys.exit(3)

BS = "\\"
DQ = '"'


def find_one(text):
    hits = [i for i, l in enumerate(lines) if l == text]
    if len(hits) != 1:
        raise SystemExit("ANCHOR NOT UNIQUE (%d hits): %r" % (len(hits), text))
    return hits[0]


# ---------------------------------------------------------------- B3 (bottom)
i = find_one('  sbatch --job-name "ev_${RUN_TAG}" $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE ' + BS)
last = ("             ls eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || "
        "{ echo 'EVAL PRODUCED NO CSV'; exit 1; }" + DQ + " >/dev/null")
assert lines[i + 4] == last, repr(lines[i + 4])
b3 = [
    '  EJID=$(sbatch --parsable --job-name "ev_${RUN_TAG}" $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 06:00:00 $EXCLUDE ' + BS,
    lines[i + 1],
    lines[i + 2],
    lines[i + 3],
    ("             ls eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || "
     "{ echo 'EVAL PRODUCED NO CSV'; exit 1; }; " + BS),
    ("             export MONITOR_RESULTS_DIR=results; bash ${DIAG_SH} '${RUN_TAG}' '' results || true" + DQ + ")"),
    '  record_job "$EJID" "$RUN_TAG" eval',
]
lines[i:i + 5] = b3

# ---------------------------------------------------------------- B2
i = find_one('  JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" ' + BS)
assert lines[i + 8].endswith('--run_tag ${RUN_TAG}")'), repr(lines[i + 8])
b2 = [
    '  # ITEM 28: side-arm training runs UNDER run_supervised.sh (monitor.py attached',
    '  # live), then the epoch-2 check. ' + BS + '$SLURM_JOB_ID must expand on the node, not here.',
    '  SA_TRAIN="python Megascale-fineTuning/train.py --full_data --no_pretrain --no_freeze ' + BS,
    '--loss_mode ddg --pooled_corr_weight 0 --dg_length_norm none --affine_calib ' + BS,
    '--wt_anchor_weight 0 --designed_weight 1 ${EXTRA} ' + BS,
    '--val_frac 0.1 --epochs 15 --seed 42 --run_tag ${RUN_TAG}"',
    '  if [ "$SUPERVISE" = "1" ]; then',
    '    SA_BODY="SUPLOG=logs/${RUN_TAG}_' + BS + '$SLURM_JOB_ID.out; ' + BS,
    'export MONITOR_RESULTS_DIR=results; ' + BS,
    ('bash ${SUP_SH} ' + "'${RUN_TAG}' " + BS + DQ + BS + '$SUPLOG' + BS + DQ +
     ' -- ${SA_TRAIN}; TRC=' + BS + '$?; ' + BS),
    ('bash ${E2_SH} ' + "'${RUN_TAG}' " + BS + DQ + BS + '$SUPLOG' + BS + DQ +
     ' results || true; exit ' + BS + '$TRC"'),
    '  else',
    '    SA_BODY="${SA_TRAIN}"',
    '  fi',
    '  JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" ' + BS,
    '     $QOS_FLAG $GPU_SPEC --cpus-per-task=8 --time 1-00:00:00 $EXCLUDE ' + BS,
    '     --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" ' + BS,
    '     --wrap "module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled; ' + BS,
    '             export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ' + BS,
    '             ${SA_BODY}")',
    '  record_job "$JID" "$RUN_TAG" train',
]
lines[i:i + 9] = b2

# ---------------------------------------------------------------- B1 (top)
i = find_one("mkdir -p logs results eval_results")
assert lines[i - 1] == 'GO=${1:-dry}', repr(lines[i - 1])
b1 = [
    '',
    '# --- ITEM 28: supervision (same contract as submit_factorial.sh)',
    'SUPERVISE="${SUPERVISE:-1}"',
    'SUP_SH="cluster_run/scripts/run_supervised.sh"',
    'E2_SH="cluster_run/scripts/epoch2_check.sh"',
    'DIAG_SH="cluster_run/scripts/eval_diag.sh"',
    'STATE_JSON="results/state.json"',
    'export MONITOR_RESULTS_DIR="results"',
    'if [ "$SUPERVISE" = "1" ] && [ ! -f "$SUP_SH" ]; then',
    '  echo "FATAL: SUPERVISE=1 but ${SUP_SH} is missing (AUTOPILOT S3)."; exit 3',
    'fi',
    'record_job () {   # $1 = job id, $2 = run tag, $3 = kind',
    '''  python3 - "$STATE_JSON" "$1" "$2" "$3" <<'PYSTATE' '''.rstrip(),
    'import json, os, sys, time',
    'path, jid, tag, kind = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]',
    'st = {}',
    'if os.path.exists(path):',
    '    try:',
    '        st = json.load(open(path))',
    '    except ValueError:',
    '        st = {}',
    'st.setdefault("submitted", [])',
    'if not any(isinstance(e, dict) and str(e.get("job_id")) == str(jid) for e in st["submitted"]):',
    '    st["submitted"].append({"job_id": str(jid), "run_tag": tag, "kind": kind,',
    '                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})',
    'tmp = path + ".tmp"',
    'os.makedirs(os.path.dirname(path) or ".", exist_ok=True)',
    'json.dump(st, open(tmp, "w"), indent=2, sort_keys=True)',
    'os.replace(tmp, path)',
    'PYSTATE',
    '}',
]
lines[i + 1:i + 1] = b1

open(P, "w").write("\n".join(lines))
print("submit_sidearms.sh PATCHED (B1 B2 B3)")
