#!/usr/bin/env python3
"""Idea 23 / T-E5 — the 26,315 double mutants as the only genuinely held-out test.

WHY THIS MATTERS MORE THAN ANOTHER SPLIT
----------------------------------------
Every split in this project has been looked at. The test proteins have been scored dozens
of times across dozens of arms; the validation set selects epochs. Neither is untouched.

The double mutants are different: 26,315 measurements that appear in NO split, because
`--one_mut` filtering removed them at dataset construction. They have never been trained
on, never been validated on, never been scored. They are the closest thing to a genuinely
held-out set that exists here.

AND THEY TEST SOMETHING THE SINGLE MUTANTS CANNOT
-------------------------------------------------
A double mutant asks whether the model has learned an ENERGY FUNCTION or a lookup over
single substitutions. If the model is additive -- pred(AB) ~ pred(A) + pred(B) -- it has
learned per-site effects. If it captures epistasis, the deviation from additivity should
correlate with the measured deviation.

    additivity gap = ddG(AB) - [ddG(A) + ddG(B)]

An energy function that sums per-residue terms is additive BY CONSTRUCTION for mutations at
distant sites, but should show coupling for nearby ones, through the shared contact
environment. That is a testable prediction of the architecture, not a metric.

USAGE
    python double_mutant_test.py --pred doubles_pred.csv --singles singles_pred.csv
"""
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_mutation(s):
    """'A12G' -> ('A', 12, 'G'). Returns None if unparseable."""
    s = str(s).strip()
    if len(s) < 3:
        return None
    wt, mt = s[0], s[-1]
    try:
        pos = int(s[1:-1])
    except ValueError:
        return None
    return wt, pos, mt


def split_double(s):
    """'A12G,K30A' or 'A12G:K30A' -> ['A12G', 'K30A']."""
    for sep in (",", ":", ";", "+", "/"):
        if sep in str(s):
            return [p.strip() for p in str(s).split(sep)]
    return None


def build_additive_baseline(doubles, singles):
    """For each double, the sum of its two singles, where both are available."""
    key = {}
    for _, r in singles.iterrows():
        key[(r["protein"], str(r["mut"]).strip())] = (r["ddG"], r["pred_ddG"])

    rows = []
    for _, r in doubles.iterrows():
        parts = split_double(r["mut"])
        if not parts or len(parts) != 2:
            continue
        a = key.get((r["protein"], parts[0]))
        b = key.get((r["protein"], parts[1]))
        if a is None or b is None:
            continue
        pa, pb = parse_mutation(parts[0]), parse_mutation(parts[1])
        sep = abs(pa[1] - pb[1]) if (pa and pb) else np.nan
        rows.append(dict(
            protein=r["protein"], mut=r["mut"],
            true_double=r["ddG"], pred_double=r["pred_ddG"],
            true_additive=a[0] + b[0], pred_additive=a[1] + b[1],
            seq_separation=sep,
        ))
    return pd.DataFrame(rows)


def report(df):
    if df.empty:
        print("FAIL no double mutant could be matched to both of its singles.")
        return 1

    print(f"Matched {len(df)} doubles to both singles, "
          f"{df.protein.nunique()} proteins\n")

    print("=== 1. Raw predictive accuracy on unseen doubles ===")
    r = pearsonr(df.true_double, df.pred_double)[0]
    s = spearmanr(df.true_double, df.pred_double)[0]
    rmse = float(np.sqrt(((df.true_double - df.pred_double) ** 2).mean()))
    print(f"  pooled PCC {r:.4f}   SCC {s:.4f}   RMSE {rmse:.4f}")
    print("  This is the cleanest generalisation number available in the project:")
    print("  no epoch was selected on it and no arm was chosen because of it.\n")

    print("=== 2. Does the model beat its own additive baseline? ===")
    r_add = pearsonr(df.true_double, df.pred_additive)[0]
    print(f"  model prediction      PCC {r:.4f}")
    print(f"  sum of its 2 singles  PCC {r_add:.4f}")
    delta = r - r_add
    if delta > 0.01:
        print(f"  +{delta:.4f} -- the model captures something beyond additivity.")
    elif delta < -0.01:
        print(f"  {delta:.4f} -- the model is WORSE than summing its own singles.")
        print("  That means the joint prediction is degraded by processing both mutations")
        print("  together, which points at the representation rather than the energy.")
    else:
        print("  Indistinguishable. The model is effectively additive: it has learned")
        print("  per-site effects, not a coupled energy function.")
    print()

    print("=== 3. Epistasis: is the deviation from additivity predicted? ===")
    df = df.copy()
    df["true_gap"] = df.true_double - df.true_additive
    df["pred_gap"] = df.pred_double - df.pred_additive
    gr = pearsonr(df.true_gap, df.pred_gap)[0]
    print(f"  corr(true gap, predicted gap) {gr:.4f}")
    print(f"  measured epistasis  mean {df.true_gap.mean():+.3f}  "
          f"std {df.true_gap.std():.3f}")
    print(f"  predicted epistasis mean {df.pred_gap.mean():+.3f}  "
          f"std {df.pred_gap.std():.3f}")
    if df.pred_gap.std() < 0.1 * df.true_gap.std():
        print("  The model predicts almost NO epistasis. For a per-residue-sum energy")
        print("  function that is the expected behaviour, and it is a clean statement")
        print("  about the architecture's limit rather than a training failure.")
    print()

    print("=== 4. Does coupling depend on sequence separation? ===")
    print("  Physical prediction: nearby mutations share a contact environment and should")
    print("  couple; distant ones should be additive.")
    ok = df.dropna(subset=["seq_separation"])
    if len(ok) > 50:
        for lo, hi in [(0, 5), (5, 15), (15, 40), (40, 10**6)]:
            sub = ok[(ok.seq_separation >= lo) & (ok.seq_separation < hi)]
            if len(sub) < 20:
                continue
            print(f"  sep [{lo:>3},{hi if hi < 10**6 else 'inf'!s:>4}): n={len(sub):>5}  "
                  f"|true gap| {sub.true_gap.abs().mean():.3f}  "
                  f"|pred gap| {sub.pred_gap.abs().mean():.3f}")
        cs = pearsonr(ok.seq_separation, ok.true_gap.abs())[0]
        print(f"  corr(separation, |true epistasis|) {cs:+.4f}  "
              f"(expect negative: closer couples more)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="predictions on the double mutants")
    p.add_argument("--singles", required=True, help="predictions on the single mutants")
    p.add_argument("--out", default="results/DOUBLE_MUTANTS.tsv")
    a = p.parse_args()

    doubles, singles = pd.read_csv(a.pred), pd.read_csv(a.singles)
    for name, d in (("doubles", doubles), ("singles", singles)):
        for c in ("protein", "mut", "ddG", "pred_ddG"):
            if c not in d.columns:
                print(f"FAIL {name}: missing column '{c}'. Columns: "
                      f"{list(d.columns)}", file=sys.stderr)
                return 2

    df = build_additive_baseline(doubles, singles)
    rc = report(df)
    if not df.empty:
        df.to_csv(a.out, sep="\t", index=False)
        print(f"\nWritten: {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
