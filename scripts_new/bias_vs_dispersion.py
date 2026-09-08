#!/usr/bin/env python3
"""T-E3 — decompose b_p into the inert global constant and the dispersion that matters.

WHY
---
b_p was measured to be 96.3% ONE GLOBAL CONSTANT. A global constant added to every
prediction of every protein cannot change any Pearson correlation -- correlation is
invariant to a shift. So a lever that reduces MAE(b_p) while raising std(b_p) has made
things worse while appearing to help.

That is exactly the Flory coil's failure mode: MAE fell 4.965 -> 3.952 (-20%, celebrated)
while std(b_p) rose 0.9972 -> 1.1231 (+13%, the number that actually matters).

THE DECOMPOSITION
    b_p = global + residual_p
    global   = mean_p(b_p)    -> inert, cannot move any correlation
    std(b_p) = THE TARGET

RULE THIS SCRIPT ENFORCES
    A lever passes only if std_b_p FALLS.
    MAE falling while std rises is a regression dressed as a win.

DIAGNOSTIC IDENTITY
    If every protein is off in the same direction (n_under == n_total), then
    MAE(b_p) == |global| exactly. When that holds, MAE is measuring bias only and carries
    no information about correlation. The script tests and reports this.

USAGE
    python bias_vs_dispersion.py <eval.csv> [more.csv ...] [--baseline <eval.csv>]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

EXCLUDE_PROTEINS = {"2K5H"}
MIN_MUTS = 3

# Reference values on the no-coil baseline, for the pass/fail verdict.
BASELINE_STD_B = 0.9972
BASELINE_MAE_B = 5.0749


def per_protein_table(csv, exclude=EXCLUDE_PROTEINS):
    """One row per protein: offset b_p, slope a_p, length, size."""
    df = pd.read_csv(csv)
    for col in ("protein", "ddG", "pred_ddG"):
        if col not in df.columns:
            raise ValueError(f"{csv}: missing column '{col}'")
    df = df[~df["protein"].isin(exclude)].dropna(subset=["ddG", "pred_ddG"])

    rows = []
    for name, g in df.groupby("protein"):
        if len(g) < MIN_MUTS:
            continue
        b = float((g["pred_ddG"] - g["ddG"]).mean())
        if g["ddG"].std(ddof=0) > 0:
            a = float(np.polyfit(g["ddG"], g["pred_ddG"], 1)[0])
            r = float(pearsonr(g["ddG"], g["pred_ddG"])[0])
        else:
            a, r = np.nan, np.nan                      # undefined, not zero
        length = np.nan
        for cand in ("length", "L", "seq_len", "n_res"):
            if cand in g.columns:
                length = float(g[cand].iloc[0])
                break
        rows.append(dict(protein=name, b_p=b, a_p=a, pp_pcc=r,
                         n_mut=len(g), length=length))
    return pd.DataFrame(rows)


def decompose(csv):
    t = per_protein_table(csv)
    if t.empty:
        raise ValueError(f"{csv}: no protein had >= {MIN_MUTS} mutations")

    b = t.b_p.values
    glob = float(b.mean())
    var_within = float(b.var(ddof=0))
    std_b = float(np.sqrt(var_within))
    # Fraction of mean-square b_p explained by the shared constant.
    frac_global = glob ** 2 / (glob ** 2 + var_within) if (glob ** 2 + var_within) > 0 else np.nan

    n_under = int((b < 0).sum())
    n_over = int((b > 0).sum())
    mae = float(np.abs(b).mean())
    all_same_sign = (n_under == len(b)) or (n_over == len(b))

    out = dict(
        file=os.path.basename(csv),
        n_proteins=len(t),
        global_constant=round(glob, 4),
        std_b_p=round(std_b, 4),                       # THE TARGET
        mae_b_p=round(mae, 4),                         # the misleading one
        frac_variance_global=round(frac_global, 4),
        n_under=n_under, n_over=n_over,
        mae_equals_bias=bool(all_same_sign and abs(mae - abs(glob)) < 1e-6),
        a_p_median=round(float(t.a_p.median(skipna=True)), 4),
        a_p_min=round(float(t.a_p.min(skipna=True)), 4),
        a_p_max=round(float(t.a_p.max(skipna=True)), 4),
        a_p_below_0p3=int((t.a_p < 0.3).sum()),
        pp_median=round(float(t.pp_pcc.median(skipna=True)), 4),
    )

    if t.length.notna().sum() >= 3:
        ok = t.dropna(subset=["length", "b_p"])
        out["corr_b_length"] = round(float(pearsonr(ok.b_p, ok.length)[0]), 4)
        ok2 = t.dropna(subset=["length", "a_p"])
        out["corr_a_length"] = round(float(pearsonr(ok2.a_p, ok2.length)[0]), 4)
    return out, t


def verdict(res, baseline_std=BASELINE_STD_B, baseline_mae=BASELINE_MAE_B):
    """PASS only if the dispersion falls. MAE is reported but never decides."""
    d_std = res["std_b_p"] - baseline_std
    d_mae = res["mae_b_p"] - baseline_mae
    if d_std < -0.01:
        v = f"PASS   std_b_p {d_std:+.4f} vs baseline -- the dispersion fell"
    elif d_std > 0.01:
        v = f"FAIL   std_b_p {d_std:+.4f} vs baseline -- the dispersion ROSE"
    else:
        v = f"FLAT   std_b_p {d_std:+.4f} -- within noise"
    if d_mae < -0.01 and d_std > 0.01:
        v += ("\n       TRAP: MAE fell {:+.4f} while std ROSE. This is the coil's failure "
              "mode -- shifting an inert constant while worsening what matters."
              .format(d_mae))
    return v


def main(argv):
    files = [a for a in argv if not a.startswith("--")]
    base_std, base_mae = BASELINE_STD_B, BASELINE_MAE_B
    if "--baseline" in argv:
        bpath = argv[argv.index("--baseline") + 1]
        bres, _ = decompose(bpath)
        base_std, base_mae = bres["std_b_p"], bres["mae_b_p"]
        files = [f for f in files if f != bpath]
        print(f"Baseline from {os.path.basename(bpath)}: "
              f"std_b_p={base_std}  mae_b_p={base_mae}\n")

    all_rows = []
    for csv in files:
        try:
            res, table = decompose(csv)
        except Exception as exc:
            print(f"ERROR {csv}: {exc}", file=sys.stderr)
            continue
        all_rows.append(res)

        print("=" * 78)
        print(res["file"])
        print("=" * 78)
        print(f"  global constant (inert)     {res['global_constant']:>9.4f}")
        print(f"  std(b_p)   <-- THE TARGET   {res['std_b_p']:>9.4f}")
        print(f"  MAE(b_p)   (misleading)     {res['mae_b_p']:>9.4f}")
        print(f"  frac of mean-square that is the global constant "
              f"{res['frac_variance_global']:.4f}")
        print(f"  under/over predicted        {res['n_under']}/{res['n_over']} "
              f"of {res['n_proteins']}")
        if res["mae_equals_bias"]:
            print("  NOTE  every protein errs in the same direction, so "
                  "MAE(b_p) == |global constant| identically.")
            print("        MAE here measures BIAS ONLY and cannot predict any change "
                  "in correlation.")
        print(f"  slope a_p   median {res['a_p_median']:.4f}   "
              f"min {res['a_p_min']:.4f}   max {res['a_p_max']:.4f}   "
              f"collapsed(<0.3): {res['a_p_below_0p3']}")
        if "corr_b_length" in res:
            print(f"  corr(b_p, length) {res['corr_b_length']:+.4f}   "
                  f"corr(a_p, length) {res['corr_a_length']:+.4f}")
            if abs(res["corr_b_length"]) > 0.4:
                print("  WARN  strong length dependence in the offset -- "
                      "the reference state is distorting with chain length.")
        print("\n  " + verdict(res, base_std, base_mae).replace("\n", "\n  "))
        print()

    if all_rows:
        out = "results/BIAS_VS_DISPERSION.tsv"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, sep="\t", index=False)
        print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
