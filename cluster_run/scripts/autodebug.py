#!/usr/bin/env python3
# TESTED: in isolation only -- classifier exercised on synthetic log fixtures (see
#   --selftest), allowlist gate unit-tested including the empty-state.json case. NOT
#   exercised: a live scancel, a live sbatch, a real SLURM queue (cluster-only).
"""
autodebug.py -- classify a failed DeepEF job from its log and apply the AUTOPILOT
section-2 failure-protocol table.

It is a PLANNER by default. `--apply` is required before it runs any scancel, and
even then every cancellation passes the allowlist gate below.

=========================== THE TWO SAFETY INVARIANTS ===========================
1. CANCELLATION ALLOWLIST. A job id may be cancelled ONLY if it appears in
   results/state.json -> submitted[]. 48 calibration runs and 3 pilots are live and
   irreplaceable. Any id not on the list is refused, loudly, with no fallback path.
   There is exactly one function that shells out to scancel (`_scancel`) and it
   calls `assert_cancellable()` on its first line.

2. A monitor.py STOP on M4 (val PP < 0.30 @ epoch 2) or M5 (slope collapse) is
   DATA, NOT A FAULT. It is the failure mode under study. Class MODEL_DATA is
   recorded with retries=0 and is NEVER retried, resubmitted or cancelled.
================================================================================

Retry discipline (AUTOPILOT s2): never retry an identical command more than twice.
A third attempt MUST change something and record what changed -- enforced in
`plan_action` by escalating to a mutated command, and refused outright if nothing
further can be changed.

Usage:
  python autodebug.py --log logs/p3_a1_d0_s1_D0_uemb_seed42_18021044.out
  python autodebug.py --run-tag p3_... --job-id 18021044          # locates the log
  python autodebug.py --scan logs/                                # classify many
  python autodebug.py --log ... --apply                           # actually act
  python autodebug.py --selftest                                  # no cluster needed
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

STATE_PATH_DEFAULT = "results/state.json"
RESULTS_DIR_DEFAULT = "results"

# ------------------------------------------------------------------ classes
CUDA_OOM      = "cuda_oom"
PENDING_LONG  = "pending_long"
SLOW_NODE     = "slow_node"
HANG          = "hang"
NAN_INF       = "nan_inf"
EVAL_FAILED   = "eval_failed"
MODEL_DATA    = "model_data_not_fault"   # M4 / M5 -- the failure mode under study
DISK_FULL     = "disk_full"
UNKNOWN       = "unknown"
OK            = "ok"

# AUTOPILOT section 2 max-retry column.
MAX_RETRIES = {
    CUDA_OOM: 2, PENDING_LONG: 2, SLOW_NODE: 3, HANG: 2,
    NAN_INF: 1, EVAL_FAILED: 2, MODEL_DATA: 0, DISK_FULL: 0, UNKNOWN: 1, OK: 0,
}

# Node classes with >=48GB, for the OOM escalation. AUTOPILOT s3 preferred list.
BIG_NODES = "cs-6000,ee-l40s"
BAD_NODES = ["ise-pheno", "cs-2080", "cs-1080"]   # never schedule here
HANG_MINUTES = 60
PENDING_HOURS = 12
SLOW_FACTOR = 2.0

# ------------------------------------------------------------------ log patterns
RE_OOM       = re.compile(r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|"
                          r"CUBLAS_STATUS_ALLOC_FAILED", re.I)
RE_NAN_LOSS  = re.compile(r"loss[^\n]{0,40}\b(nan|inf)\b|\b(nan|inf)\b[^\n]{0,20}loss", re.I)
RE_DISK      = re.compile(r"No space left on device|Disk quota exceeded", re.I)
RE_EVALFAIL  = re.compile(r"EVAL PRODUCED NO CSV|ERROR: checkpoint .* not found", re.I)
RE_STOPLINE  = re.compile(r"^STOP \[([A-Z0-9]+)\]\s*(.*)$", re.M)
RE_SIT       = re.compile(r"([0-9.]+)s/it")
RE_ITS       = re.compile(r"([0-9.]+)it/s")
RE_CKPT      = re.compile(r"kf_all_epoch_(\d+)\.pt")


# ==================================================================== state
def load_state(path=STATE_PATH_DEFAULT):
    """state.json is the ONLY authority on what this run set owns."""
    if not os.path.exists(path):
        return {"state": None, "submitted": [], "retries": {}, "decisions": [],
                "blocked": [], "regime_changed": []}
    with open(path) as fh:
        st = json.load(fh)
    st.setdefault("submitted", [])
    st.setdefault("retries", {})
    st.setdefault("decisions", [])
    st.setdefault("blocked", [])
    st.setdefault("regime_changed", [])
    return st


def save_state(st, path=STATE_PATH_DEFAULT):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)   # atomic: a torn state.json loses the allowlist


def submitted_ids(st):
    """Normalise submitted[] to a set of string job ids.

    Accepts either ["18021044", ...] or [{"job_id": "18021044", ...}, ...] so it
    survives whichever shape the submitter writes.
    """
    out = set()
    for e in st.get("submitted", []):
        if isinstance(e, dict):
            for k in ("job_id", "jobid", "jid", "id", "eval_job_id", "eval_jid"):
                if e.get(k) is not None:
                    out.add(str(e[k]).strip())
        else:
            out.add(str(e).strip())
    return {x for x in out if x}


# ============================================================ ALLOWLIST GATE
class NotOurs(Exception):
    """Raised when a job id cannot be proven to belong to this run set."""


def assert_cancellable(job_id, st, path=STATE_PATH_DEFAULT):
    """THE GATE. Prove `job_id` belongs to this run set or refuse.

    Refuses on: missing/empty state.json, an id not in submitted[], a non-numeric
    id, or an id with a step suffix (18021044.batch) that would otherwise slip
    past a naive membership test.

    An empty allowlist refuses EVERYTHING. That is deliberate: 'we do not know
    what we own' must never mean 'cancel freely'.
    """
    jid = str(job_id).strip()
    if not jid:
        raise NotOurs("empty job id")
    if not re.fullmatch(r"\d+", jid):
        raise NotOurs("job id %r is not a bare numeric id (array/step ids are "
                      "refused; cancel the parent explicitly)" % jid)
    allow = submitted_ids(st)
    if not allow:
        raise NotOurs(
            "%s lists no submitted jobs. The allowlist is empty, so NOTHING may be "
            "cancelled. 48 calibration runs and 3 pilots are live." % path)
    if jid not in allow:
        raise NotOurs(
            "job %s is NOT in %s submitted[] (%d known ids). REFUSING to cancel: it "
            "may be one of the 48 calibration runs or 3 pilots." % (jid, path, len(allow)))
    return True


def _scancel(job_id, st, apply=False, path=STATE_PATH_DEFAULT):
    """The ONLY scancel path in this file. Gated on the first line."""
    assert_cancellable(job_id, st, path)          # <-- raises NotOurs, no fallback
    cmd = ["scancel", str(job_id)]
    if not apply:
        return {"would_run": " ".join(cmd), "ran": False}
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"ran": True, "cmd": " ".join(cmd), "rc": r.returncode,
            "stderr": r.stderr.strip()[:200]}


# ==================================================================== helpers
def _read(path, limit_bytes=4_000_000):
    """Read a log tail-first; training logs are large and context is scarce."""
    if not path or not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, errors="ignore") as fh:
        if size > limit_bytes:
            fh.seek(size - limit_bytes)
        return fh.read()


def log_age_minutes(path):
    if not path or not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 60.0


def parse_sit(text):
    """Median seconds-per-iteration from tqdm output (handles both s/it and it/s)."""
    vals = [float(x) for x in RE_SIT.findall(text)]
    vals += [1.0 / float(x) for x in RE_ITS.findall(text) if float(x) > 0]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def newest_checkpoint(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return None, None
    best, bestn = None, -1
    for f in os.listdir(model_dir):
        m = RE_CKPT.match(f)
        if m and int(m.group(1)) > bestn:
            bestn, best = int(m.group(1)), os.path.join(model_dir, f)
    return best, (bestn if bestn >= 0 else None)


def monitor_stop(run_tag, results_dir=RESULTS_DIR_DEFAULT):
    """Read the .stop file run_supervised.sh writes, if any."""
    p = os.path.join(results_dir, run_tag + "_monitor.stop") if run_tag else None
    if p and os.path.exists(p):
        try:
            return json.load(open(p))
        except (ValueError, OSError):
            return None
    return None


def epoch2_verdict(run_tag, results_dir=RESULTS_DIR_DEFAULT):
    p = os.path.join(results_dir, run_tag + "_epoch2.json") if run_tag else None
    if p and os.path.exists(p):
        try:
            return json.load(open(p))
        except (ValueError, OSError):
            return None
    return None


# ==================================================================== classify
def classify(log_text, run_tag=None, log_path=None, job_state=None,
             pending_hours=None, sit=None, wave_median_sit=None,
             results_dir=RESULTS_DIR_DEFAULT, is_eval_log=False):
    """Return (klass, evidence).

    ORDER MATTERS: MODEL_DATA is tested FIRST so a run under study is never
    mistaken for a fault -- an M5 slope-collapse log can also contain
    scary-looking numbers, and retrying it would destroy the result.
    """
    # --- 1. monitor STOP that is data, not a fault -------------------------
    stop = monitor_stop(run_tag, results_dir)
    if stop and stop.get("rule") in ("M4", "M5"):
        return MODEL_DATA, ("monitor STOP [%s] -- %s. This is the failure mode under "
                            "study: RECORD, NEVER RETRY." % (stop["rule"], stop.get("line", "")))
    e2 = epoch2_verdict(run_tag, results_dir)
    if e2 and e2.get("verdict") == "FAIL":
        return MODEL_DATA, ("epoch-2 check FAIL (PP=%s < %s). Bad cell, not a fault: "
                            "RECORD, NEVER RETRY." % (e2.get("pp"), e2.get("pp_min")))
    m = RE_STOPLINE.search(log_text or "")
    if m and m.group(1) in ("M4", "M5"):
        return MODEL_DATA, ("monitor STOP [%s] in log -- %s. RECORD, NEVER RETRY."
                            % (m.group(1), m.group(2)[:120]))

    # --- 2. hard resource / numerical faults -------------------------------
    if RE_DISK.search(log_text or ""):
        return DISK_FULL, "no space left on device / quota exceeded"
    if RE_OOM.search(log_text or ""):
        return CUDA_OOM, "CUDA out of memory in log"

    if log_text:
        mm = RE_NAN_LOSS.search(log_text)
        if mm:
            return NAN_INF, "NaN/Inf in loss: %r" % mm.group(0)[:80]
        ms = RE_STOPLINE.search(log_text)
        if ms and ms.group(1) == "M2":
            return NAN_INF, "monitor STOP [M2] NaN/Inf: %s" % ms.group(2)[:100]

    # --- 3. queue / contention --------------------------------------------
    if job_state == "PENDING":
        if pending_hours is not None and pending_hours > PENDING_HOURS:
            return PENDING_LONG, "PENDING %.1f h > %d h" % (pending_hours, PENDING_HOURS)
        return OK, ("PENDING %.1f h (under the %d h threshold)"
                    % (pending_hours or 0.0, PENDING_HOURS))

    if sit is not None and wave_median_sit:
        if sit > SLOW_FACTOR * wave_median_sit:
            return SLOW_NODE, ("s/it %.3f > %.1fx wave median %.3f"
                               % (sit, SLOW_FACTOR, wave_median_sit))

    # --- 4. hang: no log line for 60 min while still RUNNING ---------------
    # Only a hang if the job is actually alive; a finished job's log is
    # legitimately old, and calling that a hang would cancel a COMPLETED run.
    age = log_age_minutes(log_path)
    if job_state == "RUNNING" and age is not None and age > HANG_MINUTES:
        return HANG, "no log line for %.0f min (> %d) while RUNNING" % (age, HANG_MINUTES)

    # --- 5. eval-only failure ---------------------------------------------
    if RE_EVALFAIL.search(log_text or ""):
        return EVAL_FAILED, "eval produced no CSV / checkpoint not found"

    if not log_text:
        return UNKNOWN, "log empty or unreadable"
    return UNKNOWN, "no known failure signature matched"


# ==================================================================== plan
def retry_key(run_tag, klass):
    return "%s::%s" % (run_tag or "?", klass)


def plan_action(klass, evidence, st, run_tag=None, job_id=None, model_dir=None,
                node=None, results_dir=RESULTS_DIR_DEFAULT,
                state_path=STATE_PATH_DEFAULT):
    """Return an action dict implementing the AUTOPILOT s2 table.

    Retry accounting: retries[key] counts PRIOR attempts. Attempt 3 (n>=2) MUST
    change something; `changed` records what. If nothing further can be changed
    the cell is marked failed rather than retried identically.
    """
    key = retry_key(run_tag, klass)
    n = int(st.get("retries", {}).get(key, 0))
    cap = MAX_RETRIES.get(klass, 1)

    act = {"run_tag": run_tag, "job_id": job_id, "class": klass,
           "evidence": evidence, "prior_retries": n, "max_retries": cap,
           "cancel": None, "resubmit": None, "changed": None,
           "regime_changed": False, "record_only": False, "note": ""}

    # ---- the failure mode under study: record, never retry ---------------
    if klass == MODEL_DATA:
        act.update(record_only=True, max_retries=0,
                   note=("DATA, NOT A FAULT. Record the cell's result in RESULTS.tsv "
                         "and move on. No retry, no cancel, no resubmit."))
        return act

    if klass == OK:
        act.update(record_only=True, note="nothing to do")
        return act

    if klass == DISK_FULL:
        act.update(record_only=True,
                   note=("AUTOPILOT s2: delete epoch_0..epoch_7 for runs whose best epoch "
                         "is known. Still full -> HALT H3. Not automated: deleting "
                         "checkpoints is an operator decision."))
        return act

    # Generic cap. NAN_INF is exempt here only so its own branch can emit the
    # specific "drop the cell, log F3" instruction; that branch is terminal too,
    # so the cap is still enforced -- see the NaN block below.
    if n >= cap and klass != NAN_INF:
        act.update(record_only=True,
                   note=("retry cap reached (%d/%d) -- mark the cell `failed`, continue with "
                         "the rest, report it in S9. A missing cell is a footnote; a "
                         "fabricated one is misconduct." % (n, cap)))
        return act

    # ---- CUDA OOM --------------------------------------------------------
    if klass == CUDA_OOM:
        if n == 0:
            act["resubmit"] = {"constraint_nodes": BIG_NODES,
                               "sbatch_extra": "--constraint=%s" % BIG_NODES}
            act["changed"] = "node class -> >=48GB (%s); mini_batch UNCHANGED" % BIG_NODES
            act["note"] = "attempt 2: bigger node only, so the 64-batch comparison stays valid"
        else:
            # Attempt 3 must change something ELSE, and it changes the regime.
            act["resubmit"] = {"constraint_nodes": BIG_NODES,
                               "sbatch_extra": "--constraint=%s" % BIG_NODES,
                               "train_arg_override": {"--mini_batch": "32"}}
            act["changed"] = "mini_batch 64 -> 32 (halved) AND node >=48GB"
            act["regime_changed"] = True
            act["note"] = ("REGIME-CHANGED: mini_batch != 64. This run is EXCLUDED from any "
                           "comparison against the 64-batch cells. Recorded in "
                           "state.json.regime_changed; it must carry the tag in RESULTS.tsv.")
        return act

    # ---- PENDING > 12 h --------------------------------------------------
    if klass == PENDING_LONG:
        parts = ["--partition=gpu-a", "--partition=gpu-b"]
        act["cancel"] = job_id     # gated at execution time
        act["resubmit"] = {"partition": parts[min(n, len(parts) - 1)]}
        act["changed"] = "partition -> %s (attempt %d)" % (parts[min(n, len(parts) - 1)], n + 2)
        act["note"] = ("log the node-class change with the result; a cell's seeds must not "
                       "span node classes (AUTOPILOT s3)")
        return act

    # ---- slow node -------------------------------------------------------
    if klass == SLOW_NODE:
        excl = list(BAD_NODES)
        if node and node not in excl:
            excl.append(node)
        act["cancel"] = job_id
        act["resubmit"] = {"exclude_nodes": ",".join(excl)}
        act["changed"] = ("exclude %s (ise-pheno is 3.5x slower and is excluded permanently)"
                          % (node or "n/a"))
        return act

    # ---- hang ------------------------------------------------------------
    if klass == HANG:
        ckpt, epoch = newest_checkpoint(model_dir)
        act["cancel"] = job_id
        if ckpt:
            act["resubmit"] = {"resume_from": ckpt, "resume_epoch": epoch}
            act["changed"] = "resume from newest checkpoint %s (epoch %s)" % (ckpt, epoch)
        else:
            act["resubmit"] = {"from_scratch": True}
            act["changed"] = "no epoch_*.pt found -> restart from scratch"
            act["note"] = "no checkpoint to resume from; this loses the partial run"
        return act

    # ---- NaN / Inf -------------------------------------------------------
    if klass == NAN_INF:
        if n == 0:
            act["resubmit"] = {"train_arg_override": {"--ranking_weight": "0"}}
            act["changed"] = "--ranking_weight 0 (AUTOPILOT s2: resubmit ONCE, this way only)"
            act["note"] = "do NOT retry as-is"
        else:
            act.update(record_only=True, max_retries=1,
                       note="still NaN with --ranking_weight 0 -> DROP THE CELL, log F3")
        return act

    # ---- eval failed, training fine --------------------------------------
    if klass == EVAL_FAILED:
        act["resubmit"] = {"eval_only": True}
        if n == 0:
            act["changed"] = "re-run eval only (attempt 2)"
        else:
            act["changed"] = ("re-run eval AND verify the checkpoint path is DERIVED, "
                              "not hardcoded (attempt 3 must change something)")
            act["note"] = "twice failed -> check the checkpoint path is derived, not hardcoded"
        return act

    # ---- unknown ---------------------------------------------------------
    act.update(record_only=True,
               note=("unclassified. Do NOT retry blindly (AUTOPILOT s2: classify first). "
                     "Inspect the log tail by hand and add a signature here."))
    return act


# ==================================================================== execute
def execute(act, st, apply=False, state_path=STATE_PATH_DEFAULT):
    """Apply an action. Cancellation is gated; resubmission is EMITTED, not run.

    Resubmission is deliberately not automated: the submit scripts are idempotent
    and carry the site rules (QOS, gres, exclude list) that a hand-built sbatch
    here would get wrong -- notably `--gres=gpu:rtx_6000:1`, which must never be
    replaced by a partition name because the submit filter reroutes onto a 1080
    and the run OOMs. autodebug prints the exact command and records it.
    """
    res = {"run_tag": act.get("run_tag"), "class": act["class"],
           "applied": bool(apply), "cancel": None, "resubmit_cmd": None,
           "refused": None}

    if act.get("record_only") and not act.get("cancel"):
        res["note"] = act.get("note", "")
        _record(st, act, res, state_path, count_retry=False)
        return res

    if act.get("cancel"):
        try:
            res["cancel"] = _scancel(act["cancel"], st, apply=apply, path=state_path)
        except NotOurs as e:
            # HARD STOP for this action. No fallback cancellation path exists.
            res["refused"] = str(e)
            res["cancel"] = {"ran": False, "refused": str(e)}
            res["note"] = ("cancellation REFUSED by the allowlist; the resubmit is "
                           "withheld too, because resubmitting beside a job we could "
                           "not cancel would double-run the cell.")
            _record(st, act, res, state_path, count_retry=False)
            return res

    if act.get("resubmit") is not None:
        res["resubmit_cmd"] = _resubmit_command(act)

    _record(st, act, res, state_path, count_retry=True)
    return res


def _resubmit_command(act):
    """Build the human-runnable resubmission line (never executed here)."""
    r = act["resubmit"]
    tag = act.get("run_tag") or "<tag>"
    if r.get("eval_only"):
        return ("bash Megascale-fineTuning/run_calib_eval.sh '<train_log>' "
                "'<model_dir>' '%s'   # eval only" % tag)
    bits = ["bash cluster_run/scripts/resubmit_one.sh --run-tag %s" % tag]
    if r.get("sbatch_extra"):
        bits.append("--sbatch-extra '%s'" % r["sbatch_extra"])
    if r.get("partition"):
        bits.append("--partition %s" % r["partition"])
    if r.get("exclude_nodes"):
        bits.append("--exclude %s" % r["exclude_nodes"])
    if r.get("resume_from"):
        bits.append("--resume-from %s" % r["resume_from"])
    for k, v in (r.get("train_arg_override") or {}).items():
        bits.append("--train-arg '%s %s'" % (k, v))
    if act.get("regime_changed"):
        bits.append("--regime-changed")
    bits.append("--go")
    return " ".join(bits)


def _record(st, act, res, state_path, count_retry):
    key = retry_key(act.get("run_tag"), act["class"])
    if count_retry and act["class"] not in (MODEL_DATA, OK):
        st.setdefault("retries", {})
        st["retries"][key] = int(st["retries"].get(key, 0)) + 1
    if act.get("regime_changed"):
        rc = st.setdefault("regime_changed", [])
        entry = {"run_tag": act.get("run_tag"),
                 "reason": act.get("changed"),
                 "excluded_from": "any comparison against mini_batch=64",
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if entry not in rc:
            rc.append(entry)
    st.setdefault("autodebug_log", []).append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_tag": act.get("run_tag"), "job_id": act.get("job_id"),
        "class": act["class"], "evidence": act.get("evidence"),
        "changed": act.get("changed"), "note": act.get("note"),
        "applied": res.get("applied"), "refused": res.get("refused"),
        "resubmit_cmd": res.get("resubmit_cmd"),
        "prior_retries": act.get("prior_retries"),
    })
    save_state(st, state_path)


# ==================================================================== cli
def _find_log(run_tag, job_id, logs_dir="logs"):
    if not os.path.isdir(logs_dir):
        return None
    cands = []
    for f in os.listdir(logs_dir):
        if run_tag and run_tag not in f:
            continue
        if job_id and str(job_id) not in f:
            continue
        cands.append(os.path.join(logs_dir, f))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _report(act, res):
    print("=" * 70)
    print("run_tag : %s   job %s" % (act.get("run_tag"), act.get("job_id")))
    print("class   : %s   (retries %s/%s)"
          % (act["class"], act.get("prior_retries"), act.get("max_retries")))
    print("evidence: %s" % act.get("evidence"))
    if act.get("changed"):
        print("changed : %s" % act["changed"])
    if act.get("regime_changed"):
        print("REGIME-CHANGED: excluded from any comparison against mini_batch=64")
    if act.get("note"):
        print("note    : %s" % act["note"])
    if res.get("refused"):
        print("REFUSED : %s" % res["refused"])
    if res.get("cancel"):
        print("cancel  : %s" % json.dumps(res["cancel"]))
    if res.get("resubmit_cmd"):
        print("resubmit: %s" % res["resubmit_cmd"])
    print("applied : %s" % res.get("applied"))


def survey_queue(st, state_path, results_dir, apply=False):
    """Classify every live job of this user against the allowlist. READ ONLY.

    This is the --dry-run verification path: it proves, on the real queue, that the
    48 calibration runs are NOT cancellable, because they are not on the allowlist.
    It never calls _scancel; it only asks assert_cancellable() what would happen.
    """
    fmt = "%i|%j|%T|%M|%R"
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", fmt],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        print("[autodebug] cannot read squeue: %s" % e)
        return 1
    rows = [l for l in r.stdout.splitlines() if l.strip()]
    allow = submitted_ids(st)
    print("=" * 78)
    print("AUTODEBUG QUEUE SURVEY  (DRY RUN -- nothing is cancelled)")
    print("state file : %s" % state_path)
    print("allowlist  : %d job id(s) in submitted[]" % len(allow))
    print("live jobs  : %d" % len(rows))
    print("=" * 78)
    print("%-10s %-34s %-9s %-8s %s" % ("JOBID", "NAME", "STATE", "TIME", "CANCELLABLE"))
    n_prot = 0
    for line in rows:
        parts = (line.split("|") + [""] * 5)[:5]
        jid, name, state, tm, reason = parts
        try:
            assert_cancellable(jid, st, state_path)
            verdict = "YES (on allowlist)"
        except NotOurs:
            verdict = "NO -- protected"
            n_prot += 1
        print("%-10s %-34s %-9s %-8s %s" % (jid[:10], name[:34], state[:9], tm[:8], verdict))
    print("-" * 78)
    print("%d of %d live job(s) are PROTECTED (not on the allowlist -> uncancellable)."
          % (n_prot, len(rows)))
    print("autodebug would cancel NOTHING in this invocation (dry run, apply=%s)." % apply)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--run-tag")
    ap.add_argument("--job-id")
    ap.add_argument("--model-dir")
    ap.add_argument("--node")
    ap.add_argument("--job-state", choices=["RUNNING", "PENDING", "FAILED",
                                            "COMPLETED", "TIMEOUT", "CANCELLED"])
    ap.add_argument("--pending-hours", type=float)
    ap.add_argument("--wave-median-sit", type=float,
                    help="median s/it across the wave, for the contention rule")
    ap.add_argument("--scan", help="classify every *.out in this directory")
    ap.add_argument("--state", default=STATE_PATH_DEFAULT)
    ap.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--apply", action="store_true",
                    help="actually run scancel (still allowlist-gated). Without it, plan only.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="explicit no-op default: plan only, never scancel. Overrides --apply.")
    ap.add_argument("--queue", action="store_true",
                    help="survey the live squeue: classify every job of this user against "
                         "the allowlist and print what WOULD be done. Never cancels.")
    ap.add_argument("--json", help="write the action(s) to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    # DRY RUN IS THE DEFAULT and --dry-run forces it even if --apply was passed.
    # Belt and braces: the allowlist gate is the real protection, this is the fuse.
    if args.dry_run and args.apply:
        print("[autodebug] --dry-run overrides --apply: NOTHING will be cancelled.")
        args.apply = False

    st = load_state(args.state)
    out = []

    if args.queue:
        return survey_queue(st, args.state, args.results_dir, apply=args.apply)

    targets = []
    if args.scan:
        for f in sorted(os.listdir(args.scan)):
            if f.endswith(".out"):
                targets.append((os.path.join(args.scan, f), None, None))
    else:
        log = args.log or _find_log(args.run_tag, args.job_id)
        targets.append((log, args.run_tag, args.job_id))

    for log, tag, jid in targets:
        if tag is None and log:
            base = os.path.basename(log)
            m = re.match(r"(?:ev_)?(.+?)_(\d+)\.out$", base)
            if m:
                tag, jid = m.group(1), (jid or m.group(2))
        text = _read(log)
        is_eval = bool(log and os.path.basename(log).startswith("ev_"))
        klass, ev = classify(text, run_tag=tag, log_path=log,
                             job_state=args.job_state,
                             pending_hours=args.pending_hours,
                             sit=parse_sit(text),
                             wave_median_sit=args.wave_median_sit,
                             results_dir=args.results_dir,
                             is_eval_log=is_eval)
        act = plan_action(klass, ev, st, run_tag=tag, job_id=jid,
                          model_dir=args.model_dir, node=args.node,
                          results_dir=args.results_dir, state_path=args.state)
        res = execute(act, st, apply=args.apply, state_path=args.state)
        _report(act, res)
        out.append({"action": act, "result": res, "log": log})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote %s" % args.json)
    return 0


# ==================================================================== selftest
def selftest():
    """No cluster, no GPU. Proves the classifier and, above all, the gate."""
    import tempfile
    state = {"ok": True}

    def check(name, cond):
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        state["ok"] = state["ok"] and bool(cond)

    tmp = tempfile.mkdtemp()
    rdir = os.path.join(tmp, "results")
    os.makedirs(rdir)
    spath = os.path.join(rdir, "state.json")

    print("---- allowlist gate ----")
    empty = load_state(spath)
    try:
        assert_cancellable("18021044", empty, spath)
        check("empty allowlist refuses every id", False)
    except NotOurs:
        check("empty allowlist refuses every id", True)

    st = {"submitted": ["18021044", {"job_id": "18021045", "eval_job_id": "18021046"}],
          "retries": {}, "regime_changed": []}
    check("listed bare id allowed", assert_cancellable("18021044", st, spath))
    check("listed dict id allowed", assert_cancellable("18021045", st, spath))
    check("listed eval id allowed", assert_cancellable("18021046", st, spath))
    for bad, why in [("99999999", "unlisted id refused"),
                     ("18021044.batch", "step id refused"),
                     ("", "empty id refused"),
                     ("all", "'all' refused"),
                     ("-u nissim", "'-u user' refused")]:
        try:
            assert_cancellable(bad, st, spath)
            check(why, False)
        except NotOurs:
            check(why, True)

    try:
        _scancel("99999999", st, apply=True, path=spath)
        check("_scancel refuses an unlisted id even with --apply", False)
    except NotOurs:
        check("_scancel refuses an unlisted id even with --apply", True)

    print("---- classifier ----")
    cases = [
        ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate", CUDA_OOM),
        ("Epoch 3 | loss: nan | ddG PCC: 0.1", NAN_INF),
        ("EVAL PRODUCED NO CSV", EVAL_FAILED),
        ("OSError: [Errno 28] No space left on device", DISK_FULL),
        ("Epoch 1 | PCC: 0.42 | ddG PCC: 0.41", UNKNOWN),
    ]
    for text, want in cases:
        got, _ = classify(text, run_tag="t_none", results_dir=rdir)
        check("classify -> %-12s (got %s)" % (want, got), got == want)

    got, _ = classify("Training Epoch: 4: 120/300 [00:41<02:00, 1.10s/it]",
                      run_tag="t_slow", results_dir=rdir,
                      sit=parse_sit("1.10s/it"), wave_median_sit=0.40)
    check("s/it 1.10 > 2x median 0.40 -> slow_node", got == SLOW_NODE)

    got, _ = classify("waiting", run_tag="t_pend", results_dir=rdir,
                      job_state="PENDING", pending_hours=18.0)
    check("PENDING 18h -> pending_long", got == PENDING_LONG)
    got, _ = classify("waiting", run_tag="t_pend", results_dir=rdir,
                      job_state="PENDING", pending_hours=3.0)
    check("PENDING 3h -> ok (no action)", got == OK)

    # M4/M5 must win over everything
    with open(os.path.join(rdir, "t_m5_monitor.stop"), "w") as fh:
        json.dump({"run_tag": "t_m5", "rule": "M5", "class": "model-data-not-fault",
                   "line": "STOP [M5] std(pred)/std(true)=0.08 < 0.15"}, fh)
    got, ev = classify("STOP [M5] slope collapse\nsome loss nan text here",
                       run_tag="t_m5", results_dir=rdir)
    check("M5 -> MODEL_DATA even with 'loss nan' in the same log", got == MODEL_DATA)
    a = plan_action(got, ev, {"retries": {}}, run_tag="t_m5", job_id="18021044")
    check("M5 record_only", a["record_only"] is True)
    check("M5 no cancel", a["cancel"] is None)
    check("M5 no resubmit", a["resubmit"] is None)
    check("M5 max_retries 0", a["max_retries"] == 0)

    with open(os.path.join(rdir, "t_m4_epoch2.json"), "w") as fh:
        json.dump({"verdict": "FAIL", "pp": 0.11, "pp_min": 0.30}, fh)
    got, _ = classify("nothing odd here", run_tag="t_m4", results_dir=rdir)
    check("epoch-2 FAIL -> MODEL_DATA", got == MODEL_DATA)

    # a MODEL_DATA action must never increment a retry counter
    st_md = {"retries": {}, "regime_changed": []}
    act_md = plan_action(MODEL_DATA, "m5", st_md, run_tag="t_m5", job_id="18021044")
    execute(act_md, st_md, apply=False, state_path=spath)
    check("MODEL_DATA never counted as a retry", st_md["retries"] == {})

    print("---- retry escalation ----")
    s2 = {"retries": {}, "regime_changed": []}
    a1 = plan_action(CUDA_OOM, "oom", s2, run_tag="c1", job_id="18021044")
    check("OOM attempt 2 = bigger node, no regime change",
          a1["regime_changed"] is False and "UNCHANGED" in (a1["changed"] or ""))
    s2["retries"][retry_key("c1", CUDA_OOM)] = 1
    a2 = plan_action(CUDA_OOM, "oom", s2, run_tag="c1", job_id="18021044")
    check("OOM attempt 3 changes something (mini_batch halved) + regime-changed",
          a2["regime_changed"] is True and "32" in json.dumps(a2["resubmit"]))
    s2["retries"][retry_key("c1", CUDA_OOM)] = 2
    a3 = plan_action(CUDA_OOM, "oom", s2, run_tag="c1", job_id="18021044")
    check("OOM at cap -> record_only, no 4th identical retry", a3["record_only"] is True)

    # regime_changed must be recorded in state
    s2b = {"retries": {retry_key("c1", CUDA_OOM): 1}, "regime_changed": []}
    a2b = plan_action(CUDA_OOM, "oom", s2b, run_tag="c1", job_id="18021044")
    execute(a2b, s2b, apply=False, state_path=spath)
    check("regime_changed recorded in state.json",
          any(e["run_tag"] == "c1" for e in s2b["regime_changed"]))

    s3 = {"retries": {}}
    n1 = plan_action(NAN_INF, "nan", s3, run_tag="n1", job_id="18021044")
    check("NaN attempt 2 sets --ranking_weight 0",
          n1["resubmit"]["train_arg_override"]["--ranking_weight"] == "0")
    s3["retries"][retry_key("n1", NAN_INF)] = 1
    n2 = plan_action(NAN_INF, "nan", s3, run_tag="n1", job_id="18021044")
    check("NaN twice -> drop the cell, log F3",
          n2["record_only"] is True and "F3" in n2["note"])
    # the NAN_INF cap exemption must not leak into an unbounded loop
    for k in (2, 3, 7):
        s3["retries"][retry_key("n1", NAN_INF)] = k
        nk = plan_action(NAN_INF, "nan", s3, run_tag="n1", job_id="18021044")
        check("NaN at n=%d still terminal (no resubmit)" % k,
              nk["record_only"] is True and nk["resubmit"] is None)

    print("---- refusal withholds the resubmit ----")
    st4 = load_state(spath)                     # empty allowlist
    act = plan_action(HANG, "hang", st4, run_tag="h1", job_id="18021044", model_dir=None)
    r = execute(act, st4, apply=True, state_path=spath)
    check("refused cancel -> no resubmit_cmd emitted", r["resubmit_cmd"] is None)
    check("refusal recorded in the result", bool(r["refused"]))
    st5 = load_state(spath)
    check("refusal persisted to state.json",
          any(e.get("refused") for e in st5.get("autodebug_log", [])))

    print("\n[autodebug] SELFTEST %s" % ("PASSED" if state["ok"] else "FAILED"))
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
