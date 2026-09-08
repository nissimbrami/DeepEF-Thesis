#!/usr/bin/env python3
"""P2 + P4 — two audits that decide whether existing results are interpretable.

P2  LORO WAS NEVER ACTUALLY TESTED
    Leave-One-Residue-Out asks: does the model predict the effect of substituting to a
    residue type it never saw during training? That is the sharpest generalisation test
    available, and it is the one that decides whether the chemistry channel is doing
    anything beyond memorising twenty symbols.

    It only tests that if the held-out residue really is absent from training. If it
    leaked, the arm measures nothing.

P4  FACTOR A IS NOT ESTIMABLE
    In the factorial, factor D turned out to have an effect roughly ten times larger than
    the others -- one D arm flattens the model to a_p of 0.005. Averaging a main effect for
    A across both D levels therefore averages across two different models. The A effect is
    not estimable from the full design; it is only estimable WITHIN a D level.

    This script checks the design matrix and says which effects survive.

USAGE
    python loro_and_factorA_audit.py loro --train-csv T --eval-csv E --held-out W
    python loro_and_factorA_audit.py factorial --results results/CANONICAL.tsv
"""
import re
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

EXCLUDE = {"2K5H"}


# ======================================================================================
# P2 -- LORO leakage audit
# ======================================================================================

def parse_target(mut):
    s = str(mut).strip()
    return s[-1] if len(s) >= 3 else None


def loro_audit(train_csv, eval_csv, held_out):
    """Confirm the held-out residue is genuinely absent from training, then score it."""
    print(f"=== LORO audit: held-out residue '{held_out}' ===\n")

    tr = pd.read_csv(train_csv)
    tr["target"] = tr["mut"].apply(parse_target)
    n_leak = int((tr.target == held_out).sum())
    print(f"1. LEAKAGE CHECK")
    print(f"   training mutations to '{held_out}': {n_leak}")
    if n_leak > 0:
        print(f"   FAIL -- the residue is present in training. This arm does not test")
        print(f"   generalisation to an unseen residue and its number must not be quoted")
        print(f"   as such.")
        print(f"   Proteins involved: "
              f"{sorted(tr[tr.target == held_out].protein.unique())[:10]}")
        return 1
    print(f"   PASS -- absent from training\n")

    ev = pd.read_csv(eval_csv)
    ev = ev[~ev.protein.isin(EXCLUDE)]
    ev["target"] = ev["mut"].apply(parse_target)
    held = ev[ev.target == held_out].dropna(subset=["ddG", "pred_ddG"])
    seen = ev[ev.target != held_out].dropna(subset=["ddG", "pred_ddG"])

    if len(held) < 20:
        print(f"   only {len(held)} held-out mutations in eval -- too few to score")
        return 1

    r_h = pearsonr(held.ddG, held.pred_ddG)[0]
    r_s = pearsonr(seen.ddG, seen.pred_ddG)[0]
    a_h = np.polyfit(held.ddG, held.pred_ddG, 1)[0]
    a_s = np.polyfit(seen.ddG, seen.pred_ddG, 1)[0]

    print(f"2. GENERALISATION")
    print(f"   held-out '{held_out}': n={len(held):>5}  PCC {r_h:.4f}  slope {a_h:.4f}")
    print(f"   seen residues       : n={len(seen):>5}  PCC {r_s:.4f}  slope {a_s:.4f}")
    print(f"   gap: PCC {r_h - r_s:+.4f}   slope {a_h - a_s:+.4f}\n")

    print(f"3. INTERPRETATION")
    if r_h > 0.8 * r_s:
        print(f"   The model transfers to an unseen residue type. Its representation is")
        print(f"   continuous in chemistry rather than a lookup over twenty symbols.")
        print(f"   Corollary: an explicit descriptor block (W6) is likely redundant, which")
        print(f"   matches the finding that ProtT5 predicts held-out-residue hydropathy at")
        print(f"   R^2 0.704.")
    elif r_h > 0.5 * r_s:
        print(f"   Partial transfer. Some chemistry generalises, some does not. Worth")
        print(f"   reporting the per-residue breakdown rather than one number.")
    else:
        print(f"   Transfer FAILS. The model has learned per-residue-type behaviour and")
        print(f"   cannot extrapolate. That is the strongest possible case for an explicit")
        print(f"   physicochemical descriptor block.")
    return 0


# ======================================================================================
# P4 -- factor estimability
# ======================================================================================

TAG_RE = re.compile(r"a(?P<A>[\d.]+)_d(?P<B>[\d.]+)_s(?P<C>[\d.]+)_D(?P<D>\w+)")


def parse_design(tags):
    rows = []
    for t in tags:
        m = TAG_RE.search(t)
        if m:
            d = m.groupdict()
            d["tag"] = t
            rows.append(d)
    return pd.DataFrame(rows)


def factorial_audit(results_tsv):
    """Which effects are estimable, given that D dominates?"""
    df = pd.read_csv(results_tsv, sep="\t")
    df = df[df.get("is_canonical", True)]
    design = parse_design(df.tag.tolist())
    if design.empty:
        print("FAIL could not parse any tag as a factorial cell.")
        print("     Expected the pattern a<A>_d<B>_s<C>_D<D>.")
        return 2

    df = df.merge(design, on="tag")
    print(f"=== Factorial estimability audit ({len(df)} canonical cells) ===\n")

    print("1. CELL COVERAGE")
    for f in "ABCD":
        print(f"   factor {f}: levels {sorted(df[f].unique())}")
    full = df.groupby(list("ABCD")).size()
    print(f"   distinct cells present: {len(full)} of "
          f"{np.prod([df[f].nunique() for f in 'ABCD'])}\n")

    print("2. FACTOR D DOMINANCE")
    for d_lvl, g in df.groupby("D"):
        print(f"   D={d_lvl:<12} n={len(g):>3}  pooled "
              f"{g.pooled.min():.4f}..{g.pooled.max():.4f}  "
              f"a_p median {g.get('a_p_median', pd.Series([np.nan])).median()}")
    if df.D.nunique() > 1:
        spread = df.groupby("D").pooled.mean()
        d_effect = float(spread.max() - spread.min())
        others = []
        for f in "ABC":
            if df[f].nunique() > 1:
                s = df.groupby(f).pooled.mean()
                others.append(abs(float(s.max() - s.min())))
        max_other = max(others) if others else 0.0
        print(f"\n   |D effect| {d_effect:.4f}   largest other main effect "
              f"{max_other:.4f}")
        if max_other > 0 and d_effect > 5 * max_other:
            print(f"   D dominates by {d_effect / max_other:.0f}x.")
            print(f"   CONSEQUENCE: main effects for A, B and C are NOT estimable across D.")
            print(f"   Averaging over D averages over two different models. Report A, B and")
            print(f"   C only WITHIN a D level, and say so.\n")
        else:
            print(f"   D does not dominate; main effects are estimable across it.\n")

    print("3. ESTIMABLE EFFECTS, WITHIN EACH D LEVEL")
    for d_lvl, g in df.groupby("D"):
        print(f"\n   --- D={d_lvl} (n={len(g)}) ---")
        if len(g) < 4:
            print(f"       too few cells for any effect")
            continue
        for f in "ABC":
            if g[f].nunique() < 2:
                print(f"       {f}: single level, not estimable")
                continue
            s = g.groupby(f).pooled.agg(["mean", "count"])
            eff = float(s["mean"].iloc[-1] - s["mean"].iloc[0])
            print(f"       {f}: effect {eff:+.4f}  "
                  f"(n per level {s['count'].tolist()})")
        for f1, f2 in itertools.combinations("ABC", 2):
            if g[f1].nunique() > 1 and g[f2].nunique() > 1:
                try:
                    piv = g.pivot_table(index=f1, columns=f2, values="pooled",
                                        aggfunc="mean")
                    if piv.shape == (2, 2):
                        inter = (piv.iloc[1, 1] - piv.iloc[1, 0]) - \
                                (piv.iloc[0, 1] - piv.iloc[0, 0])
                        print(f"       {f1}x{f2} interaction {inter:+.4f}")
                except Exception:
                    pass
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("loro")
    a.add_argument("--train-csv", required=True)
    a.add_argument("--eval-csv", required=True)
    a.add_argument("--held-out", required=True)
    b = sub.add_parser("factorial")
    b.add_argument("--results", default="results/CANONICAL.tsv")
    args = p.parse_args()

    if args.cmd == "loro":
        return loro_audit(args.train_csv, args.eval_csv, args.held_out)
    return factorial_audit(args.results)


if __name__ == "__main__":
    sys.exit(main())
