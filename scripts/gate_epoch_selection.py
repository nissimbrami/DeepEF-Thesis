#!/usr/bin/env python3
"""gate_epoch_selection.py -- P0 epoch-selection audit gate.

WHY THIS EXISTS
---------------
Megascale-fineTuning/run_calib_eval.sh line 15 selects BEST_E as the argmax of the
*validation* ddG PCC, parsed out of the training log that validate() emits from
self.val_ds, and then scores the 28-protein test set ONCE at that epoch. That is
honest model selection.

The failure mode this gate blocks is a run that has MORE THAN ONE scored test
epoch. Once several test epochs exist, quoting the best of them is test-set
peeking -- exactly the practice this project criticises in the prior work.

Two runs legitimately have six scored epochs each, because they were scored to
measure a TRAJECTORY (does r stay flat while s rises). Those are allowlisted in
results/trajectory_runs.txt, which ALSO records the CANONICAL epoch -- the
val-selected one -- that any table must quote.

FAIL conditions:
  1. A run tag has >1 scored epoch and is not in the allowlist.
  2. An allowlisted tag's canonical epoch has no scored CSV.
  3. An allowlist entry names a tag that has no eval CSVs at all (stale entry).
  4. An allowlisted tag has exactly one scored epoch (allowlist no longer needed).

Exit 0 = PASS, 1 = FAIL.
"""
import os
import re
import sys
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, 'eval_results')
ALLOWLIST = os.path.join(ROOT, 'results', 'trajectory_runs.txt')

CSV_RE = re.compile(r'^abl_(.+)_e(\d+)\.csv$')


def scan_eval_dir(eval_dir=None):
    """tag -> sorted list of scored epochs."""
    eval_dir = eval_dir or EVAL_DIR
    tags = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(eval_dir, 'abl_*.csv'))):
        m = CSV_RE.match(os.path.basename(path))
        if not m:
            continue
        tags[m.group(1)].append(int(m.group(2)))
    return {t: sorted(set(e)) for t, e in tags.items()}


def parse_allowlist(path=None):
    """tag -> {'canonical': int, 'reason': str}.

    Format, one record per line, pipe-separated:
        <run_tag> | canonical=e<N> | <one-line reason>
    Blank lines and lines starting with '#' are ignored.
    """
    path = path or ALLOWLIST
    allow = {}
    if not os.path.exists(path):
        return allow
    with open(path, encoding='utf-8') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{lineno}: expected '<tag> | canonical=e<N> | <reason>', got: {line}")
            tag = parts[0]
            m = re.fullmatch(r'canonical=e(\d+)', parts[1])
            if not m:
                raise ValueError(
                    f"{path}:{lineno}: second field must be 'canonical=e<N>', got: {parts[1]}")
            allow[tag] = {'canonical': int(m.group(1)), 'reason': ' | '.join(parts[2:])}
    return allow


def main():
    if not os.path.isdir(EVAL_DIR):
        print(f"GATE epoch_selection: FAIL -- no eval dir {EVAL_DIR}")
        return 1

    tags = scan_eval_dir()
    allow = parse_allowlist()
    failures = []
    notes = []

    multi = {t: e for t, e in tags.items() if len(e) > 1}
    single = {t: e for t, e in tags.items() if len(e) == 1}

    # 1. unallowlisted multi-epoch runs
    for tag in sorted(multi):
        if tag not in allow:
            failures.append(
                f"UNALLOWLISTED MULTI-EPOCH RUN: {tag} has {len(multi[tag])} scored test epochs "
                f"{['e%d' % e for e in multi[tag]]}. Quoting the best of these is test-set "
                f"peeking. Either delete the extra CSVs or add the tag to "
                f"results/trajectory_runs.txt with its val-selected canonical epoch.")

    # 2/4. allowlist entries validated against reality
    for tag in sorted(allow):
        if tag not in tags:
            failures.append(
                f"STALE ALLOWLIST ENTRY: {tag} is in results/trajectory_runs.txt but has no "
                f"eval_results/abl_{tag}_e*.csv. Remove the entry.")
            continue
        eps = tags[tag]
        if len(eps) == 1:
            failures.append(
                f"UNNEEDED ALLOWLIST ENTRY: {tag} has only one scored epoch (e{eps[0]}); "
                f"the allowlist exemption is not needed. Remove the entry.")
            continue
        canon = allow[tag]['canonical']
        if canon not in eps:
            failures.append(
                f"CANONICAL EPOCH NOT SCORED: {tag} canonical=e{canon} (val-selected) but the "
                f"scored epochs are {['e%d' % e for e in eps]}. Any table quoting this run is "
                f"quoting a non-canonical epoch. Score e{canon} on test, or correct the "
                f"canonical field if the val log says otherwise.")
        else:
            notes.append(f"  allowlisted {tag}: scored {['e%d' % e for e in eps]}, "
                         f"CANONICAL=e{canon} -- tables MUST quote e{canon}")

    print(f"GATE epoch_selection: {len(tags)} run tags, "
          f"{len(single)} single-epoch, {len(multi)} multi-epoch, {len(allow)} allowlisted")
    for n in notes:
        print(n)

    if failures:
        print("GATE epoch_selection: FAIL")
        for f in failures:
            print("  - " + f)
        return 1

    print("GATE epoch_selection: PASS -- every run has exactly one scored test epoch, "
          "except allowlisted trajectory runs whose canonical epoch is on record.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
