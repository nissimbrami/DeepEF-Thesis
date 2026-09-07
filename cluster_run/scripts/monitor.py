#!/usr/bin/env python3
# TESTED: in isolation only -- each stop rule fired on a synthetic fixture; parses real log without error. NOT exercised: live attachment to a running train.py (cluster-only).
"""
monitor.py -- DeepEF training run monitor.

Reads a training log from stdin OR from a file path (argv[1]) and, per epoch,
appends a JSON line to results/<tag>_monitor.jsonl. Watches early-failure signals
and prints a STOP line (and exits nonzero) if a stop rule fires.

Usage:
    python monitor.py                          # read stdin, tag=run
    python monitor.py path/to/train.log        # read file, tag=run
    python monitor.py path/to/train.log mytag  # read file, tag=mytag
    cat train.log | python monitor.py - mytag  # stdin with explicit tag

Stop rules (return code 2 = STOP, 0 = clean, 3 = warn-only issues seen):
  M1: train ddg_loss flat or RISING through epoch 2                -> STOP
  M2: any NaN or Inf anywhere                                      -> STOP (first hit)
  M3: grad-norm > 100, OR < 1e-6 for a full epoch                  -> STOP
  M4: val ddG PCC-PP < 0.30 at END of epoch 2                      -> STOP
  M5: median within-protein std(pred)/std(true) < 0.15            -> STOP (slope collapse)
  M6: log std(b) each epoch, NEVER stop on it                     -> record only
  M7: epoch wall-clock > 2x first epoch's wall-clock              -> STOP
  M8: cov() degrees-of-freedom warnings > 20% of minibatches      -> STOP (ranking dead)
  M9: val-train PP gap > 0.3 early                                -> WARN (record)
"""
import sys
import os
import re
import json
import math
import time

# ---------------------------------------------------------------- thresholds
M1_EPOCH        = 2       # evaluate ddg_loss trend through this epoch
M3_GRAD_HI      = 100.0
M3_GRAD_LO      = 1e-6
M4_PP_MIN       = 0.30
M4_EPOCH        = 2
M5_SLOPE_MIN    = 0.15
M7_WALL_FACTOR  = 2.0
M8_COV_FRAC     = 0.20
M9_GAP_MAX      = 0.30
M9_EARLY_EPOCH  = 2       # "early" = at or before this epoch

# ---------------------------------------------------------------- regexes
RE_NANINF = re.compile(r"\b(nan|inf|-inf|\+inf)\b", re.IGNORECASE)

# tqdm-style epoch marker: "Training Epoch: 3:" or "Epoch 3:"
RE_TQDM_EPOCH = re.compile(r"(?:Training\s+)?Epoch[:\s]+(\d+)\s*[:\|]", re.IGNORECASE)

# spec-style: [metrics] epoch=N dG_PCC=.. ddG_PCC=..
RE_METRICS_LINE = re.compile(r"\[metrics\]", re.IGNORECASE)

# real summary line: "Epoch 5 | PCC: 0.42 | Spearman: .. | RMSE: .. | ddG PCC: 0.42"
RE_EPOCH_SUMMARY = re.compile(r"\bEpoch\s+(\d+)\s*\|", re.IGNORECASE)

RE_VAL_LOSS   = re.compile(r"Validation\s+Loss[:\s=]+([-+0-9.eE]+)", re.IGNORECASE)
RE_COV_WARN   = re.compile(r"cov\(\):\s*degrees of freedom", re.IGNORECASE)

# generic numeric token (also nan/inf)
NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"


def _to_float(s):
    if s is None:
        return None
    sl = s.strip().lower()
    try:
        if sl == "nan":
            return float("nan")
        if sl in ("inf", "+inf"):
            return float("inf")
        if sl == "-inf":
            return float("-inf")
        return float(s)
    except (ValueError, TypeError):
        return None


def _find_kv(line, keys):
    """Return (float_value_or_None, raw_str_or_None) for first matched key.
    Matches 'key=val', 'key: val', 'key val'."""
    for k in keys:
        pat = re.compile(re.escape(k) + r"\s*[:=]?\s*" + NUM, re.IGNORECASE)
        m = pat.search(line)
        if m:
            raw = m.group(1)
            return _to_float(raw), raw
    return None, None


def _is_bad(x):
    return x is not None and isinstance(x, float) and (math.isnan(x) or math.isinf(x))


class EpochState:
    """Accumulates parsed signals for one epoch."""
    def __init__(self, epoch):
        self.epoch = epoch
        self.ddg_loss = None
        self.grad_norms = []
        self.val_pp = None
        self.val_ddg_pcc = None
        self.train_pp = None
        self.std_pred_over_true = None
        self.std_b = None
        self.minibatches = 0
        self.cov_warns = 0
        self.wall_start = None
        self.wall_clock = None
        self.saw_naninf = False
        self._finalized = False


class Monitor:
    def __init__(self, tag, results_dir):
        self.tag = tag
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.out_path = os.path.join(results_dir, tag + "_monitor.jsonl")
        open(self.out_path, "w").close()  # fresh file each run
        self.epochs = {}
        self.order = []
        self.cur = None
        self.first_wall = None
        self.stopped = False
        self.stop_reason = None
        self.warnings = []

    # ------------------------------------------------ epoch lifecycle
    def _get_epoch(self, n):
        if n not in self.epochs:
            st = EpochState(n)
            st.wall_start = time.monotonic()
            self.epochs[n] = st
            self.order.append(n)
        return self.epochs[n]

    def _switch_epoch(self, n):
        if self.cur is not None and self.cur.epoch != n:
            self._finalize_epoch(self.cur)
        self.cur = self._get_epoch(n)

    # ------------------------------------------------ line parsing
    def feed(self, line):
        if self.stopped:
            return
        raw = line.rstrip("\n")

        # M2: NaN / Inf anywhere (exact-token match)
        if RE_NANINF.search(raw):
            if self.cur is not None:
                self.cur.saw_naninf = True
            self._stop("M2", "NaN/Inf detected in log: " + raw.strip()[:120])
            return

        # cov() warning accounting
        if RE_COV_WARN.search(raw):
            if self.cur is not None:
                self.cur.cov_warns += 1
            return

        # tqdm epoch marker + minibatch index
        mt = RE_TQDM_EPOCH.search(raw)
        if mt:
            self._switch_epoch(int(mt.group(1)))
            mb = re.search(r"(\d+)/(\d+)\s*\[", raw)
            if mb and self.cur is not None:
                self.cur.minibatches = max(self.cur.minibatches, int(mb.group(1)))

        # spec metrics line
        if RE_METRICS_LINE.search(raw):
            ep, _ = _find_kv(raw, ["epoch"])
            if ep is not None:
                self._switch_epoch(int(ep))
            self._scrape_metrics(raw)
            return

        # real epoch summary line closes an epoch
        ms = RE_EPOCH_SUMMARY.search(raw)
        if ms:
            self._switch_epoch(int(ms.group(1)))
            self._scrape_metrics(raw)
            self._finalize_epoch(self.cur)
            return

        # rescore-style line
        if "[rescore]" in raw.lower() and "pcc" in raw.lower():
            self._scrape_rescore(raw)
            return

        # generic scrape (loss / grad / std_b etc.)
        self._scrape_metrics(raw)

    def _scrape_metrics(self, raw):
        if self.cur is None:
            self.cur = self._get_epoch(0)
        st = self.cur

        v, rawv = _find_kv(raw, ["ddg_loss", "ddG_loss", "train_ddg_loss"])
        if rawv is not None:
            if _is_bad(v):
                self._stop("M2", "NaN/Inf in ddg_loss: " + raw.strip()[:120])
                return
            st.ddg_loss = v

        v, rawv = _find_kv(raw, ["grad_norm", "gradnorm", "grad-norm", "total_norm"])
        if rawv is not None:
            if _is_bad(v):
                self._stop("M2", "NaN/Inf in grad_norm: " + raw.strip()[:120])
                return
            st.grad_norms.append(v)

        v, rawv = _find_kv(raw, ["std_pred_over_true", "slope_ratio", "std_ratio"])
        if rawv is not None and not _is_bad(v):
            st.std_pred_over_true = v

        v, rawv = _find_kv(raw, ["std_b", "std(b)", "b_std"])
        if rawv is not None:
            st.std_b = v

        v, rawv = _find_kv(raw, ["wall_clock", "epoch_time", "epoch_seconds", "elapsed"])
        if rawv is not None and not _is_bad(v):
            st.wall_clock = v

        # ddG PCC (space in token) -> handle explicitly, before generic PP
        m = re.search(r"ddG\s*PCC[:\s=]+" + NUM, raw, re.IGNORECASE)
        if m:
            fv = _to_float(m.group(1))
            if not _is_bad(fv):
                st.val_ddg_pcc = fv
        else:
            v, rawv = _find_kv(raw, ["ddG_PCC", "ddg_pcc"])
            if rawv is not None and not _is_bad(v):
                st.val_ddg_pcc = v

        # per-protein PCC
        m = re.search(r"\bPP\b[^0-9\-+]{0,12}" + NUM, raw)
        if m:
            fv = _to_float(m.group(1))
            if not _is_bad(fv):
                st.val_pp = fv
        else:
            v, rawv = _find_kv(raw, ["val_pp", "pcc_pp", "per_protein_pcc", "ddg_pcc_pp"])
            if rawv is not None and not _is_bad(v):
                st.val_pp = v

        v, rawv = _find_kv(raw, ["train_pp", "train_pcc_pp", "train_pc_corr"])
        if rawv is not None and not _is_bad(v):
            st.train_pp = v

    def _scrape_rescore(self, raw):
        me = re.search(r"epoch[=\s_]*(\d+)", raw, re.IGNORECASE)
        if me:
            self._switch_epoch(int(me.group(1)))
        if self.cur is None:
            self.cur = self._get_epoch(0)
        st = self.cur
        m = re.search(r"ddG\s*PCC\s*=\s*" + NUM, raw, re.IGNORECASE)
        if m:
            fv = _to_float(m.group(1))
            if not _is_bad(fv):
                st.val_ddg_pcc = fv
        m = re.search(r"PP\s*\(mean per-protein\)\s*=\s*" + NUM, raw, re.IGNORECASE)
        if m:
            fv = _to_float(m.group(1))
            if not _is_bad(fv):
                st.val_pp = fv

    def maybe_banner(self, line):
        m = re.search(r"=+\s*EPOCH\s+(\d+)\s*=+", line, re.IGNORECASE)
        if m:
            self._switch_epoch(int(m.group(1)))
            return True
        return False

    # ------------------------------------------------ finalize + rules
    def _finalize_epoch(self, st):
        if st is None or st._finalized:
            return
        st._finalized = True

        if st.wall_clock is None and st.wall_start is not None:
            st.wall_clock = round(time.monotonic() - st.wall_start, 4)
        if self.first_wall is None and st.wall_clock is not None:
            self.first_wall = st.wall_clock

        cov_frac = None
        if st.minibatches > 0:
            cov_frac = st.cov_warns / st.minibatches
        elif st.cov_warns > 0:
            cov_frac = 1.0

        rec = {
            "epoch": st.epoch,
            "ddg_loss": st.ddg_loss,
            "grad_norm": (max(st.grad_norms) if st.grad_norms else None),
            "grad_norm_min": (min(st.grad_norms) if st.grad_norms else None),
            "val_pp": st.val_pp,
            "val_ddg_pcc": st.val_ddg_pcc,
            "train_pp": st.train_pp,
            "std_pred_over_true": st.std_pred_over_true,
            "std_b": st.std_b,
            "wall_clock": st.wall_clock,
            "cov_warn_frac": cov_frac,
            "minibatches": st.minibatches,
            "cov_warns": st.cov_warns,
        }
        with open(self.out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        self._check_rules(st, cov_frac)

    def _check_rules(self, st, cov_frac):
        if self.stopped:
            return

        # M3: grad-norm bounds
        if st.grad_norms:
            gmax = max(st.grad_norms)
            if gmax > M3_GRAD_HI:
                self._stop("M3", "grad_norm %.4g > %s at epoch %d" % (gmax, M3_GRAD_HI, st.epoch))
                return
            if gmax < M3_GRAD_LO:
                self._stop("M3", "grad_norm max %.3g < %s for full epoch %d (vanished)" % (gmax, M3_GRAD_LO, st.epoch))
                return

        # M8: cov() df warnings > 20% of minibatches
        if cov_frac is not None and cov_frac > M8_COV_FRAC:
            self._stop("M8", "cov() df warnings %.1f%% of minibatches at epoch %d (ranking term dead)" % (cov_frac * 100, st.epoch))
            return

        # M5: slope collapse
        if st.std_pred_over_true is not None and st.std_pred_over_true < M5_SLOPE_MIN:
            self._stop("M5", "std(pred)/std(true)=%.3g < %s at epoch %d (slope collapse)" % (st.std_pred_over_true, M5_SLOPE_MIN, st.epoch))
            return

        # M7: epoch wall > 2x first epoch
        if (self.first_wall is not None and st.wall_clock is not None
                and self.order and st.epoch != self.order[0] and self.first_wall > 0):
            if st.wall_clock > M7_WALL_FACTOR * self.first_wall:
                self._stop("M7", "epoch %d wall %.4gs > %sx first-epoch %.4gs" % (st.epoch, st.wall_clock, M7_WALL_FACTOR, self.first_wall))
                return

        # M4: val PP < 0.30 at END of epoch 2
        if st.epoch == M4_EPOCH and st.val_pp is not None:
            if st.val_pp < M4_PP_MIN:
                self._stop("M4", "val ddG PCC-PP %.4g < %s at end of epoch %d" % (st.val_pp, M4_PP_MIN, M4_EPOCH))
                return

        # M1: ddg_loss flat or rising through epoch 2
        if st.epoch == M1_EPOCH:
            losses = [self.epochs[e].ddg_loss for e in self.order
                      if e <= M1_EPOCH and self.epochs[e].ddg_loss is not None]
            if len(losses) >= 2:
                first, last = losses[0], losses[-1]
                improved = (first - last) > abs(first) * 0.01
                if not improved:
                    self._stop("M1", "train ddg_loss flat/rising through epoch %d: %.4g -> %.4g" % (M1_EPOCH, first, last))
                    return

        # M9: val-train PP gap > 0.3 early -> WARN only
        if (st.epoch <= M9_EARLY_EPOCH and st.val_pp is not None and st.train_pp is not None):
            gap = st.train_pp - st.val_pp
            if gap > M9_GAP_MAX:
                self.warnings.append("M9 WARN: val-train PP gap %.3g > %s at epoch %d" % (gap, M9_GAP_MAX, st.epoch))

    def _stop(self, rule, msg):
        self.stopped = True
        self.stop_reason = (rule, msg)
        print("STOP [%s] %s" % (rule, msg), flush=True)

    def close(self):
        if self.cur is not None and not self.cur._finalized:
            self._finalize_epoch(self.cur)
        for w in self.warnings:
            print(w, flush=True)


def main(argv):
    path = None
    tag = "run"
    args = list(argv[1:])
    if len(args) >= 1 and args[0] not in ("-", ""):
        path = args[0]
    if len(args) >= 2 and args[1]:
        tag = args[1]

    results_dir = os.environ.get("MONITOR_RESULTS_DIR", "results")
    mon = Monitor(tag, results_dir)

    if path and path != "-":
        src = open(path, "r", errors="replace")
    else:
        src = sys.stdin

    try:
        for line in src:
            if mon.stopped:
                break
            mon.maybe_banner(line)
            mon.feed(line)
    finally:
        if src is not sys.stdin:
            src.close()

    mon.close()

    if mon.stopped:
        return 2
    if mon.warnings:
        return 3
    print("OK: parsed %d epoch(s); no stop rule fired. jsonl=%s" % (len(mon.order), mon.out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
