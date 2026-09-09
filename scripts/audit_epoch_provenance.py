#!/usr/bin/env python3
"""audit_epoch_provenance.py -- the DEEPER half of the P0 epoch audit.

scripts/gate_epoch_selection.py answers "does any run have MORE THAN ONE scored test
epoch?". That is the peeking check and it runs every round.

This script answers the complementary question, which is slower and needs the training
logs: "for each run with exactly ONE scored epoch, is that epoch the val argmax?"
It reproduces run_calib_eval.sh's selection exactly -- same regex, same first-max
tie-break -- and reports MATCH / MISMATCH / NO_LOG per run.

Run it after a batch of new evals land, not every round.

    python scripts/audit_epoch_provenance.py

IMPORTANT -- the substring trap. Matching a run tag to its log by 'tag in filename'
gives FALSE MISMATCHES: 'sigma_seed4' is a substring of 'sigma_seed42', so sigma_seed4
picks up seed42's log and looks mis-selected when it is not. This script requires the
tag to match a whole token of the log basename (split on '_' with the trailing SLURM
jobid removed), and reports AMBIGUOUS rather than guessing.
"""
import os
import re
import sys
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, 'eval_results')
LOG_DIR = os.path.join(ROOT, 'logs')
ALLOWLIST = os.path.join(ROOT, 'results', 'trajectory_runs.txt')

# The exact regex run_calib_eval.sh line 18 uses.
VAL_RE = re.compile(r'ddG PCC=(-?\d+\.\d+)\s+ddG PCC-PP=(-?\d+\.\d+)\s+ddG RMSE=(-?\d+\.\d+)')


def val_argmax(log_path):
    """(best_epoch, best_pcc, all_pccs) reproducing run_calib_eval.sh, or None."""
    with open(log_path, errors='ignore') as fh:
        matches = VAL_RE.findall(fh.read())
    if len(matches) < 2:
        return None
    pccs = [m[0] for m in matches]
    # max(range(n), key=...) -> FIRST maximum, exactly as the shell helper does.
    best = max(range(len(pccs)), key=lambda i: float(pccs[i]))
    return best, pccs[best], pccs


def strip_jobid(basename):
    """'p3_slope1.0_s42_21024049.out' -> 'p3_slope1.0_s42'"""
    stem = re.sub(r'\.(out|log)$', '', basename)
    return re.sub(r'_\d{7,}$', '', stem)


def variants(tag):
    """Naming conventions seen in logs/ for the same run.

    The submit scripts are not consistent: eval CSVs carry the seed suffix
    ('gld_w5_dg_s42') while several training logs drop it ('gld_w5_dg_21084598.out'),
    and the LORO arm's log carries an extra 'gld_' prefix that the CSV tag lacks
    ('gld_loroW_onehot_...' vs 'loroW_onehot_s42'). Enumerate explicitly rather than
    falling back to substring matching, which is the sigma_seed4/sigma_seed42 trap.
    """
    out = {tag}
    noseed = re.sub(r'_s\d+$', '', tag)          # gld_w5_dg_s42 -> gld_w5_dg
    out.add(noseed)
    out.add('gld_' + tag)                        # loroW_onehot_s42 -> gld_loroW_onehot_s42
    out.add('gld_' + noseed)                     # loroW_onehot_s42 -> gld_loroW_onehot
    return out


def candidate_logs(tag, logs):
    """Logs whose jobid-stripped stem EQUALS the tag or one of its naming variants.
    Never a bare substring -- that is the sigma_seed4/sigma_seed42 trap."""
    want = variants(tag)
    exact = [lg for lg in logs if strip_jobid(os.path.basename(lg)) == tag]
    if exact:
        return exact
    return [lg for lg in logs if strip_jobid(os.path.basename(lg)) in want]


def main():
    tags = defaultdict(list)
    for path in glob.glob(os.path.join(EVAL_DIR, 'abl_*.csv')):
        m = re.match(r'^abl_(.+)_e(\d+)\.csv$', os.path.basename(path))
        if m:
            tags[m.group(1)].append(int(m.group(2)))

    allow = set()
    if os.path.exists(ALLOWLIST):
        with open(ALLOWLIST, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('#'):
                    allow.add(line.split('|')[0].strip())

    logs = glob.glob(os.path.join(LOG_DIR, '*.out')) + glob.glob(os.path.join(LOG_DIR, '*.log'))
    verdicts = defaultdict(list)

    print(f"{'run tag':<42}{'scored':>7}{'val argmax':>11}   verdict     log")
    for tag in sorted(tags):
        eps = sorted(set(tags[tag]))
        if len(eps) != 1:
            note = 'allowlisted trajectory' if tag in allow else 'MULTI-EPOCH, NOT ALLOWLISTED'
            print(f"{tag:<42}{'multi':>7}{'--':>11}   SKIP        {note}")
            verdicts['SKIP'].append(tag)
            continue
        scored = eps[0]
        cands = candidate_logs(tag, logs)
        parsed = [(lg, val_argmax(lg)) for lg in cands]
        parsed = [(lg, v) for lg, v in parsed if v]
        if not parsed:
            print(f"{tag:<42}e{scored:<6}{'?':>11}   NO_LOG      -")
            verdicts['NO_LOG'].append(tag)
            continue
        if len({v[0] for _, v in parsed}) > 1:
            print(f"{tag:<42}e{scored:<6}{'?':>11}   AMBIGUOUS   "
                  f"{[os.path.basename(l) for l, _ in parsed]}")
            verdicts['AMBIGUOUS'].append(tag)
            continue
        lg, (best, pcc, _) = parsed[0]
        ok = (best == scored)
        verdicts['MATCH' if ok else 'MISMATCH'].append(tag)
        print(f"{tag:<42}e{scored:<6}{('e' + str(best)):>11}   "
              f"{'MATCH   ' if ok else 'MISMATCH'}    {os.path.basename(lg)} (val {pcc})")

    print()
    for k in ('MATCH', 'MISMATCH', 'NO_LOG', 'AMBIGUOUS', 'SKIP'):
        print(f"  {k:<10} {len(verdicts[k])}")
    if verdicts['MISMATCH']:
        print("\nMISMATCH runs (scored at an epoch that is NOT the val argmax):")
        for t in verdicts['MISMATCH']:
            print("  - " + t)
        print("These are not necessarily fraud -- check whether the better checkpoint even")
        print("existed when the CSV was written, and which direction the error runs.")
    # Advisory, not a gate: this is reported, never used to fail the loop.
    return 0


if __name__ == '__main__':
    sys.exit(main())
