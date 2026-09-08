#!/usr/bin/env python3
"""W13-robust — turn the measured-offset finding into a defensible recommendation.

W13 showed that MEASURING the per-protein offset from k labelled mutations lifts pooled
ddG PCC from 0.6210 to 0.7330 at k=20, against an oracle of 0.7468 -- 98% of the
recoverable gain, at zero GPU cost, after nine attempts to PREDICT that offset had failed.

A finding is not yet a recommendation. Four things stand between them, and this script
answers all four.

    A  BUDGET       what is the marginal return per extra measurement, and where does it
                    stop paying? A lab needs "measure 6", not "measure some".

    B  GENERALITY   does it hold on every checkpoint, or only the one it was found on?
                    If it holds everywhere it is a METHOD; if not it is an observation
                    about one model.

    C  ROBUSTNESS   two proteins were measured to carry 78% of the offset-removal gain.
                    Does the result survive their removal, or is it those two proteins?

    D  HONEST NULL  Pearson is invariant to a constant shift, so subtracting the SAME
                    number from every protein changes pooled PCC by exactly zero. The null
                    a per-protein corrector must beat is therefore the RAW number, never a
                    "mean baseline". This script verifies that invariance numerically
                    rather than asserting it.

Everything here is CPU-only and reads evaluation CSVs.

USAGE
    python w13_robust.py eval1.csv [eval2.csv ...] [--k 1 3 5 8 12 20] [--reps 40]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

EXCLUDE = {"2K5H"}
MIN_EVAL_LEFT = 5


# --------------------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------------------

def load(path):
    d = pd.read_csv(path)
    need = {"protein", "ddG", "pred_ddG"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path}: needs {sorted(need)}, has {list(d.columns)}")
    d = d[["protein", "ddG", "pred_ddG"]].dropna()
    d = d[~d.protein.astype(str).str.upper().isin(EXCLUDE)]
    if len(d) < 100:
        raise ValueError(f"{path}: only {len(d)} usable rows")
    return d


def pooled(y, p):
    if len(y) < 3 or np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(p, y)[0, 1])


def measured_offset_score(d, k, reps, seed=0, drop=()):
    """Pooled PCC after subtracting an offset measured from k random labelled mutations.

    Calibration and evaluation indices are disjoint BY CONSTRUCTION. If that is violated
    the number is circular and worthless, so it is enforced with index sets rather than
    with care.
    """
    dd = d[~d.protein.isin(drop)] if drop else d
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(max(1, reps if k > 0 else 1)):
        ys, ps = [], []
        for _, g in dd.groupby("protein"):
            n = len(g)
            if n < k + MIN_EVAL_LEFT:
                continue
            gy, gp = g.ddG.values, g.pred_ddG.values
            if k == 0:
                ys.append(gy); ps.append(gp); continue
            perm = rng.permutation(n)
            calib, ev = perm[:k], perm[k:]
            off = (gp[calib] - gy[calib]).mean()
            ys.append(gy[ev]); ps.append(gp[ev] - off)
        if ys:
            scores.append(pooled(np.concatenate(ys), np.concatenate(ps)))
    s = np.array([v for v in scores if np.isfinite(v)])
    return (float(s.mean()), float(s.std()), len(s)) if len(s) else (np.nan, np.nan, 0)


def oracle_score(d, drop=()):
    dd = d[~d.protein.isin(drop)] if drop else d
    ys, ps = [], []
    for _, g in dd.groupby("protein"):
        off = (g.pred_ddG - g.ddG).mean()
        ys.append(g.ddG.values); ps.append(g.pred_ddG.values - off)
    return pooled(np.concatenate(ys), np.concatenate(ps))


def per_protein_offsets(d):
    return {n: float((g.pred_ddG - g.ddG).mean()) for n, g in d.groupby("protein")}


# --------------------------------------------------------------------------------------
# D -- the shift-invariance null
# --------------------------------------------------------------------------------------

def check_shift_invariance(d):
    """Pearson is invariant to adding a constant. Verify, do not assume."""
    base = pooled(d.ddG.values, d.pred_ddG.values)
    print("=== D. THE NULL: a global constant cannot change pooled PCC ===")
    ok = True
    for c in (-5.0, -1.0, 0.0, 1.0, 5.0):
        v = pooled(d.ddG.values, d.pred_ddG.values - c)
        same = abs(v - base) < 1e-9
        ok &= same
        print(f"  subtract {c:>+6.1f} -> pooled {v:.6f}  "
              f"{'unchanged' if same else 'CHANGED -- impossible, check the code'}")
    print(f"  {'PASS' if ok else 'FAIL'}  the null to beat is the RAW number "
          f"({base:.4f}), never a mean baseline.\n")
    return base


# --------------------------------------------------------------------------------------
# A -- budget
# --------------------------------------------------------------------------------------

def budget_curve(d, ks, reps, seed=0):
    orc = oracle_score(d)
    base, _, _ = measured_offset_score(d, 0, 1, seed)
    print("=== A. BUDGET: marginal return per extra measurement ===")
    print(f"  k=0 baseline {base:.4f}   oracle {orc:.4f}   "
          f"recoverable {orc - base:+.4f}\n")
    print(f"  {'k':>4} {'pooled':>9} {'sd':>7} {'gain':>9} {'%recov':>8} "
          f"{'d/measurement':>14}")
    rows, prev_k, prev_v = [], 0, base
    for k in ks:
        m, sd, n = measured_offset_score(d, k, reps, seed)
        gain = m - base
        pct = 100 * gain / (orc - base) if (orc - base) else np.nan
        marg = (m - prev_v) / (k - prev_k) if k > prev_k else np.nan
        print(f"  {k:>4} {m:>9.4f} {sd:>7.4f} {gain:>+9.4f} {pct:>7.1f}% "
              f"{marg:>+14.5f}")
        rows.append(dict(k=k, pooled=m, sd=sd, gain=gain, pct_recovered=pct,
                         marginal=marg))
        prev_k, prev_v = k, m

    df = pd.DataFrame(rows)
    print("\n  RECOMMENDATION")
    for tgt in (0.5, 0.8, 0.9, 0.95):
        hit = df[df.pct_recovered >= 100 * tgt]
        if len(hit):
            print(f"    {int(tgt*100)}% of the recoverable gain at k = "
                  f"{int(hit.k.min())}")
    knee = df[df.marginal < 0.2 * df.marginal.max()]
    if len(knee):
        print(f"    marginal return falls below 20% of its peak at k = "
              f"{int(knee.k.min())} -- past that, extra measurements buy little")
    neg = df[(df.k > 0) & (df.gain < 0)]
    if len(neg):
        print(f"    k = {sorted(neg.k.astype(int))} make things WORSE. The offset "
              f"estimate is noisier than the offset is informative;")
        print(f"    below that threshold, do not correct at all.")
    return df, base, orc


# --------------------------------------------------------------------------------------
# C -- robustness to protein removal
# --------------------------------------------------------------------------------------

def robustness(d, k, reps, seed=0):
    """Does the gain survive removing the proteins that carry it?"""
    offs = per_protein_offsets(d)
    ranked = sorted(offs, key=lambda p: -abs(offs[p]))
    base, _, _ = measured_offset_score(d, 0, 1, seed)
    full, _, _ = measured_offset_score(d, k, reps, seed)

    print(f"\n=== C. ROBUSTNESS at k={k}: drop the highest-|offset| proteins ===")
    print(f"  {'dropped':>8} {'n_prot':>7} {'k=0':>9} {f'k={k}':>9} {'gain':>9} "
          f"{'% of full':>10}")
    full_gain = full - base
    rows = []
    for n_drop in (0, 1, 2, 3, 5):
        drop = set(ranked[:n_drop])
        b, _, _ = measured_offset_score(d, 0, 1, seed, drop)
        m, _, _ = measured_offset_score(d, k, reps, seed, drop)
        g = m - b
        pct = 100 * g / full_gain if full_gain else np.nan
        print(f"  {n_drop:>8} {d.protein.nunique() - n_drop:>7} {b:>9.4f} "
              f"{m:>9.4f} {g:>+9.4f} {pct:>9.1f}%")
        rows.append(dict(n_dropped=n_drop, gain=g, pct_of_full=pct))

    two = [r for r in rows if r["n_dropped"] == 2]
    if two:
        p = two[0]["pct_of_full"]
        print()
        if p > 70:
            print(f"  ROBUST. Removing the two largest-offset proteins retains {p:.0f}% of")
            print(f"  the gain, so the effect is distributed across the set rather than")
            print(f"  carried by a couple of outliers.")
        elif p > 40:
            print(f"  PARTLY CARRIED. {p:.0f}% survives. State the dependence explicitly")
            print(f"  and report the distribution across proteins, not one number.")
        else:
            print(f"  FRAGILE. Only {p:.0f}% survives without those two proteins. The")
            print(f"  headline is about them, and it must be reported that way. Quote the")
            print(f"  median per-protein effect alongside the pooled figure.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# B -- generality across checkpoints
# --------------------------------------------------------------------------------------

def generality(results):
    print(f"\n{'=' * 78}\n=== B. GENERALITY across {len(results)} checkpoints ===")
    print(f"  {'checkpoint':<38} {'k=0':>8} {'best k':>7} {'best':>8} {'gain':>9}")
    for r in results:
        print(f"  {r['file'][:38]:<38} {r['base']:>8.4f} {r['best_k']:>7} "
              f"{r['best']:>8.4f} {r['gain']:>+9.4f}")
    g = np.array([r["gain"] for r in results])
    print(f"\n  gain: mean {g.mean():+.4f}  sd {g.std():.4f}  "
          f"min {g.min():+.4f}  max {g.max():+.4f}")
    pos = int((g > 0).sum())
    print(f"  positive on {pos}/{len(g)} checkpoints")
    if pos == len(g) and g.std() < 0.5 * abs(g.mean()):
        print("\n  A METHOD. The gain is positive on every checkpoint and its spread is")
        print("  small relative to its size, so it is a property of the correction rather")
        print("  than of any one model. Report it as a method.")
    elif pos >= 0.8 * len(g):
        print("\n  MOSTLY GENERAL. Positive nearly everywhere but variable. Report the")
        print("  distribution across checkpoints, not a single number.")
    else:
        print("\n  NOT GENERAL. The gain does not reproduce across checkpoints, so it is an")
        print("  observation about specific models and must not be called a method.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--robust-k", type=int, default=5)
    ap.add_argument("--out", default="results/W13_ROBUST.tsv")
    a = ap.parse_args()

    results, all_rows = [], []
    for csv in a.csvs:
        try:
            d = load(csv)
        except Exception as exc:
            print(f"SKIP {csv}: {exc}", file=sys.stderr)
            continue

        print(f"\n{'#' * 78}\n# {os.path.basename(csv)}   "
              f"{d.protein.nunique()} proteins, {len(d)} mutations\n{'#' * 78}\n")

        check_shift_invariance(d)
        df, base, orc = budget_curve(d, a.k, a.reps)
        rob = robustness(d, a.robust_k, a.reps)

        best = df.loc[df.pooled.idxmax()]
        results.append(dict(file=os.path.basename(csv), base=base, oracle=orc,
                            best_k=int(best.k), best=float(best.pooled),
                            gain=float(best.pooled - base)))
        df["file"] = os.path.basename(csv)
        all_rows.append(df)

    if len(results) > 1:
        generality(results)

    if all_rows:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.concat(all_rows).to_csv(a.out, sep="\t", index=False)
        print(f"\nWritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
