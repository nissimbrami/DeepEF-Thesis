#!/usr/bin/env python3
"""W13-select — WHICH k mutations to measure, not just how many.

THE OPPORTUNITY
---------------
w13_measured_offset.py draws the k calibration mutations at RANDOM and reaches
pooled 0.7330 at k=20 against an oracle of 0.7468 -- 98% of the recoverable gain.

But the offset estimate is a MEAN, and the variance of a mean depends on which points you
average. Random sampling is the worst-case assumption: it presumes you cannot choose. In
the real workflow you can -- a wet lab picks which mutants to make.

If a designed choice reaches the same accuracy at k=8 that random needs k=20 for, that is
60% less bench work for the same result. On a 27-protein campaign that is 324 fewer
measurements.

STRATEGIES COMPARED
    random      the current baseline
    spread      pick predictions spanning the predicted ddG range (max coverage)
    central     pick predictions near the protein's predicted median
    extremes    pick the most and least destabilising predictions
    stratified  quantile bins of the prediction, one draw per bin
    diverse     greedy max-min distance in prediction space

All are computable from PREDICTIONS ALONE -- no labels are needed to CHOOSE which mutations
to measure, only to measure them. That is what makes this deployable: the lab picks the
mutants before running any assay.

LEAKAGE
    The chosen calibration mutations are removed from evaluation for that protein, by index
    set, exactly as in w13_measured_offset.py. Enforced mechanically, not by discipline.

HONEST FRAMING, to be repeated wherever this is reported
    This is FEW-SHOT, not zero-shot. It uses test-protein labels at inference. The realistic
    scenario is a lab measuring a handful of mutants before committing to a campaign. It
    must never be presented as zero-shot prediction.

USAGE
    python w13_select.py eval.csv [--k 3 5 8 12 20] [--reps 40]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

EXCLUDE = {"2K5H"}
MIN_EVAL_LEFT = 5          # mutations that must remain after calibration is removed


def load(path):
    d = pd.read_csv(path)
    need = {"protein", "ddG", "pred_ddG"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    d = d[["protein", "ddG", "pred_ddG"]].dropna()
    d = d[~d.protein.astype(str).str.upper().isin(EXCLUDE)]
    return d


def pooled(y, p):
    if len(y) < 3 or np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(p, y)[0, 1])


# --------------------------------------------------------------------------------------
# Selection strategies. Each returns k indices, using PREDICTIONS ONLY.
# --------------------------------------------------------------------------------------

def sel_random(pred, k, rng):
    return rng.permutation(len(pred))[:k]


def sel_spread(pred, k, rng):
    """k points evenly spaced along the sorted prediction range."""
    order = np.argsort(pred)
    picks = np.linspace(0, len(pred) - 1, k).round().astype(int)
    return order[picks]


def sel_central(pred, k, rng):
    """The k predictions closest to the protein's median."""
    return np.argsort(np.abs(pred - np.median(pred)))[:k]


def sel_extremes(pred, k, rng):
    """Half from each tail. Maximum leverage, maximum sensitivity to outliers."""
    order = np.argsort(pred)
    half = k // 2
    return np.concatenate([order[:half], order[-(k - half):]])


def sel_stratified(pred, k, rng):
    """One random draw from each of k quantile bins: coverage plus randomness."""
    order = np.argsort(pred)
    bins = np.array_split(order, k)
    return np.array([b[rng.integers(len(b))] for b in bins if len(b)])


def sel_diverse(pred, k, rng):
    """Greedy max-min in prediction space, seeded at the median."""
    chosen = [int(np.argmin(np.abs(pred - np.median(pred))))]
    while len(chosen) < k:
        d = np.min(np.abs(pred[:, None] - pred[chosen][None, :]), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return np.array(chosen)


STRATEGIES = {
    "random": sel_random,
    "spread": sel_spread,
    "central": sel_central,
    "extremes": sel_extremes,
    "stratified": sel_stratified,
    "diverse": sel_diverse,
}
DETERMINISTIC = {"spread", "central", "extremes", "diverse"}


def evaluate(d, k, strategy, reps, rng, eval_frac=0.5):
    """Pooled PCC after removing an offset measured from k selected mutations.

    CONFOUND THIS GUARDS AGAINST -- found by testing, and it is not subtle.
    A naive implementation lets each strategy pick its calibration mutations from the whole
    protein and evaluates on whatever is left. Then the strategies are scored on DIFFERENT
    mutations: `central` removes the middle of the predicted range and is evaluated on the
    extremes, which are easier to rank, so it scores higher for a reason that has nothing to
    do with offset estimation. In a synthetic test that inflated `central` above the oracle.

    The fix: reserve a FIXED evaluation half per protein first, identical for every
    strategy, and let all strategies choose their calibration mutations only from the other
    half. Then every number in the table is computed on the same mutations and the
    comparison is about the offset estimate alone.
    """
    fn = STRATEGIES[strategy]
    n_reps = 1 if (strategy in DETERMINISTIC and k > 0) else reps
    scores = []
    for rep in range(max(1, n_reps)):
        ys, ps = [], []
        for _, g in d.groupby("protein"):
            n = len(g)
            gy, gp = g.ddG.values, g.pred_ddG.values

            # Fixed split, seeded per protein so it is IDENTICAL across strategies.
            split_rng = np.random.default_rng(abs(hash(_)) % (2**31))
            perm = split_rng.permutation(n)
            n_eval = max(MIN_EVAL_LEFT, int(round(eval_frac * n)))
            eval_idx = perm[:n_eval]
            pool_idx = perm[n_eval:]                  # calibration may only come from here
            if len(pool_idx) < k:
                continue

            if k == 0:
                ys.append(gy[eval_idx]); ps.append(gp[eval_idx])
                continue

            local = np.unique(np.asarray(fn(gp[pool_idx], k, rng), dtype=int))
            calib = pool_idx[local]
            off = (gp[calib] - gy[calib]).mean()
            ys.append(gy[eval_idx]); ps.append(gp[eval_idx] - off)
        if ys:
            scores.append(pooled(np.concatenate(ys), np.concatenate(ps)))
    s = np.array([v for v in scores if np.isfinite(v)])
    if not len(s):
        return np.nan, np.nan, 0
    return float(s.mean()), float(s.std()), len(s)


def oracle(d, eval_frac=0.5):
    """Offset from ALL of a protein's labels, scored on the SAME fixed evaluation half.

    Computed on the identical mutations as every strategy, so the ceiling is comparable
    rather than being measured on a different set.
    """
    ys, ps = [], []
    for name, g in d.groupby("protein"):
        n = len(g)
        gy, gp = g.ddG.values, g.pred_ddG.values
        split_rng = np.random.default_rng(abs(hash(name)) % (2**31))
        perm = split_rng.permutation(n)
        n_eval = max(MIN_EVAL_LEFT, int(round(eval_frac * n)))
        eval_idx = perm[:n_eval]
        off = (gp - gy).mean()
        ys.append(gy[eval_idx]); ps.append(gp[eval_idx] - off)
    return pooled(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 3, 5, 8, 12, 20])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/W13_SELECT.tsv")
    a = ap.parse_args()

    d = load(a.csv)
    rng = np.random.default_rng(a.seed)
    orc = oracle(d)

    print(f"{os.path.basename(a.csv)}   {d.protein.nunique()} proteins, "
          f"{len(d)} mutations")
    print(f"oracle (all labels) = {orc:.4f}\n")

    hdr = f"{'k':>4} " + "".join(f"{s:>13}" for s in STRATEGIES)
    print(hdr); print("-" * len(hdr))

    rows = []
    for k in a.k:
        line = f"{k:>4} "
        for strat in STRATEGIES:
            m, sd, n = evaluate(d, k, strat, a.reps, np.random.default_rng(a.seed))
            rows.append(dict(k=k, strategy=strat, mean=m, sd=sd, n_reps=n,
                             gain=m - rows[0]["mean"] if rows else 0.0))
            line += f"{m:>9.4f}" + (f"±{sd:.3f}" if sd == sd and sd > 0 else "     ")
        print(line)

    df = pd.DataFrame(rows)
    base = df[(df.k == 0) & (df.strategy == "random")]["mean"].iloc[0]
    df["gain"] = df["mean"] - base

    print(f"\nbaseline k=0: {base:.4f}    oracle: {orc:.4f}    "
          f"recoverable: {orc - base:+.4f}\n")

    print("=== HOW MANY MEASUREMENTS DOES EACH STRATEGY NEED? ===")
    rand20 = df[(df.k == 20) & (df.strategy == "random")]["mean"]
    target = float(rand20.iloc[0]) if len(rand20) else np.nan
    if np.isfinite(target):
        print(f"target = random at k=20 -> {target:.4f}\n")
        for strat in STRATEGIES:
            sub = df[(df.strategy == strat) & (df["mean"] >= target - 1e-9)]
            if len(sub):
                kmin = int(sub.k.min())
                saving = (20 - kmin) / 20 * 100
                verdict = (f"{saving:.0f}% fewer measurements" if kmin < 20
                           else "no saving")
                print(f"  {strat:<12} reaches it at k={kmin:<3} ({verdict})")
            else:
                print(f"  {strat:<12} never reaches it within the tested k")

    print("\n=== THE k=1 THRESHOLD ===")
    k1 = df[(df.k == 1) & (df.strategy == "random")]
    if len(k1) and k1["mean"].iloc[0] < base:
        print(f"  random k=1 is WORSE than k=0 ({k1['mean'].iloc[0]:.4f} vs "
              f"{base:.4f}).")
        print("  A single measurement estimates the offset with more noise than the offset")
        print("  itself carries signal, so the correction hurts. Check whether a designed")
        print("  choice at k=1 avoids this -- if it does, the threshold is an artefact of")
        print("  random sampling and not a property of the method.")
        for strat in STRATEGIES:
            if strat == "random":
                continue
            v = df[(df.k == 1) & (df.strategy == strat)]["mean"]
            if len(v) and v.iloc[0] > base:
                print(f"    {strat}: {v.iloc[0]:.4f} -- above baseline, so it does.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nWritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
