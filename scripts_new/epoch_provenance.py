#!/usr/bin/env python3
"""P0a/b/c — find every number quoted on a non-canonical basis, and say what it should be.

WHY
---
The epoch audit found that a_p = 0.740 for gld_slope1.0_s42 was the MAXIMUM over six
test-scored epochs, while the validation argmax is e10. Quoting the maximum over
test-scored epochs is test-set peeking -- the practice this project faulted in the prior
work and rejected a headline number over.

Several documents also quote figures on the 28-protein basis rather than the canonical 27.

Mixing bases is not a presentation problem. It is how a project ends up unable to say
whether it improved on anything, and it has already happened here more than once.

WHAT THIS DOES
    1. For every run, determine the canonical epoch from the TRAINING LOG (validation
       argmax) and score that epoch alone.
    2. Flag every run where the best TEST epoch differs from the canonical one, and report
       how much peeking would have bought -- the size of the temptation.
    3. Scan the markdown for numbers that match a NON-canonical epoch of a known run, and
       report file, line, the wrong number and the right one.

It does not edit anything. Every replacement is proposed with its evidence, because a
script that silently rewrites recorded numbers is worse than the problem it fixes.

USAGE
    python epoch_provenance.py --eval-dir eval_results --log-dir logs --docs results
"""
import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd

EXCLUDE = {"2K5H"}
MIN_MUTS = 3
EXPECTED_N_PROTEINS = 27

EVAL_RE = re.compile(r"^abl_(?P<tag>.+)_e(?P<ep>\d+)\.csv$")
VAL_PATTERNS = [
    re.compile(r"epoch[=: ]+(\d+).*?val[^\n]*?ddg[_ ]?pcc[=: ]+(-?\d*\.?\d+)", re.I),
    re.compile(r"\[metrics\][^\n]*?epoch[=: ]+(\d+)[^\n]*?ddG[_ ]?PCC[=: ]+(-?\d*\.?\d+)", re.I),
]
NUM_RE = re.compile(r"(?<![\d.])0\.\d{3,4}(?![\d])")


def val_argmax(log_path):
    if not os.path.exists(log_path):
        return None, None, 0
    best_e, best_v, n = None, None, 0
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            for pat in VAL_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                n += 1
                e, v = int(m.group(1)), float(m.group(2))
                if best_v is None or v > best_v:
                    best_e, best_v = e, v
                break
    return best_e, best_v, n


def score(csv):
    df = pd.read_csv(csv)
    df = df[~df.protein.isin(EXCLUDE)].dropna(subset=["ddG", "pred_ddG"])
    pooled = float(np.corrcoef(df.pred_ddG, df.ddG)[0, 1])
    pp, slopes = [], []
    for _, g in df.groupby("protein"):
        if len(g) < MIN_MUTS or g.ddG.std(ddof=0) == 0:
            continue
        pp.append(np.corrcoef(g.pred_ddG, g.ddG)[0, 1])
        slopes.append(np.polyfit(g.ddG, g.pred_ddG, 1)[0])
    return dict(pooled=round(pooled, 4),
                pp=round(float(np.mean(pp)), 4),
                a_p=round(float(np.median(slopes)), 4),
                n_proteins=int(df.protein.nunique()),
                n_mut=int(len(df)))


def find_log(log_dir, tag):
    for cand in (f"{tag}.log", f"{tag}.out", f"train_{tag}.log", f"{tag}.txt"):
        p = os.path.join(log_dir, cand)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(log_dir, f"*{tag}*"))
    return hits[0] if hits else None


def build_table(eval_dir, log_dir):
    rows = []
    for csv in sorted(glob.glob(os.path.join(eval_dir, "abl_*_e*.csv"))):
        m = EVAL_RE.match(os.path.basename(csv))
        if not m:
            continue
        tag, ep = m.group("tag"), int(m.group("ep"))
        log = find_log(log_dir, tag)
        can_e, can_v, n_lines = val_argmax(log) if log else (None, None, 0)
        try:
            s = score(csv)
        except Exception as exc:
            print(f"ERROR {csv}: {exc}", file=sys.stderr)
            continue
        rows.append(dict(tag=tag, epoch=ep, canonical_epoch=can_e, val_pcc=can_v,
                         val_lines=n_lines, log=os.path.basename(log) if log else None,
                         is_canonical=(can_e is not None and can_e == ep), **s))
    return pd.DataFrame(rows).sort_values(["tag", "epoch"])


def report_peeking(df):
    print("=== 1. PEEKING EXPOSURE ===")
    print("For each run scored at more than one epoch: how much would reporting the best")
    print("TEST epoch have bought over the canonical one?\n")
    any_flag = False
    for tag, g in df.groupby("tag"):
        if len(g) < 2:
            continue
        can = g[g.is_canonical]
        best = g.loc[g.pooled.idxmax()]
        print(f"  {tag}")
        print(f"    scored at epochs {sorted(g.epoch.tolist())}")
        if len(can) == 1:
            c = can.iloc[0]
            delta = best.pooled - c.pooled
            print(f"    canonical  e{int(c.epoch):<3} pooled {c.pooled:.4f}  "
                  f"pp {c.pp:.4f}  a_p {c.a_p:.4f}   <-- REPORT THIS")
            print(f"    best test  e{int(best.epoch):<3} pooled {best.pooled:.4f}  "
                  f"pp {best.pp:.4f}  a_p {best.a_p:.4f}")
            if int(best.epoch) != int(c.epoch):
                any_flag = True
                print(f"    PEEKING would buy {delta:+.4f} pooled, "
                      f"{best.a_p - c.a_p:+.4f} a_p")
        else:
            any_flag = True
            print(f"    NO canonical epoch identified "
                  f"(log {'missing' if not g.log.iloc[0] else 'unparsed'}) -- "
                  f"cannot report any number from this run")
        print()
    if not any_flag:
        print("  No run differs between canonical and best-test epoch.\n")


def report_basis(df):
    print("=== 2. BASIS CHECK ===")
    bad = df[df.n_proteins != EXPECTED_N_PROTEINS]
    if len(bad):
        print(f"  {len(bad)} eval CSVs are not on the {EXPECTED_N_PROTEINS}-protein basis:")
        for _, r in bad.iterrows():
            print(f"    {r.tag} e{int(r.epoch)}: {r.n_proteins} proteins")
        print("  Numbers from these are not comparable to canonical ones.\n")
    else:
        print(f"  PASS  all on the {EXPECTED_N_PROTEINS}-protein basis\n")


def scan_docs(df, docs_dir):
    """Find quoted numbers that match a NON-canonical epoch of a known run."""
    print("=== 3. DOCUMENT SCAN ===")
    non_can = {}
    for _, r in df[~df.is_canonical].iterrows():
        for col in ("pooled", "pp", "a_p"):
            non_can.setdefault(f"{r[col]:.4f}", []).append(
                (r.tag, int(r.epoch), col, r[col]))

    canon = {}
    for _, r in df[df.is_canonical].iterrows():
        canon[r.tag] = r

    md = glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)
    hits = 0
    for path in md:
        try:
            lines = open(path, errors="ignore").read().splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            for num in NUM_RE.findall(line):
                if num not in non_can:
                    continue
                for tag, ep, col, val in non_can[num]:
                    if tag.split("_")[0] not in line and tag not in line:
                        continue                      # require the run to be named nearby
                    hits += 1
                    c = canon.get(tag)
                    right = f"{c[col]:.4f}" if c is not None else "unknown"
                    rel = os.path.relpath(path, docs_dir)
                    print(f"  {rel}:{i}")
                    print(f"    quotes {num} = {tag} {col} at NON-canonical e{ep}")
                    print(f"    canonical value is {right}"
                          + (f" (e{int(c.epoch)})" if c is not None else ""))
                    print(f"    > {line.strip()[:100]}")
    if hits == 0:
        print("  No non-canonical number found next to its run name.")
        print("  Note this is a conservative scan: it only flags a number when the run is")
        print("  named on the same line, so it under-reports rather than over-reports.\n")
    else:
        print(f"\n  {hits} occurrence(s). Each needs a manual correction -- this script")
        print("  deliberately does not rewrite recorded numbers.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--docs", default="results")
    ap.add_argument("--out", default="results/EPOCH_PROVENANCE.tsv")
    a = ap.parse_args()

    df = build_table(a.eval_dir, a.log_dir)
    if df.empty:
        print("FAIL no eval CSVs parsed.", file=sys.stderr)
        return 2

    print(f"{len(df)} eval CSVs across {df.tag.nunique()} runs\n")
    report_peeking(df)
    report_basis(df)
    if os.path.isdir(a.docs):
        scan_docs(df, a.docs)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"Written: {a.out}")

    print("\n=== CANONICAL TABLE -- the only numbers that may be quoted ===")
    can = df[df.is_canonical]
    print(can[["tag", "epoch", "pooled", "pp", "a_p", "n_proteins"]]
          .to_string(index=False) if len(can) else "(none identified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
