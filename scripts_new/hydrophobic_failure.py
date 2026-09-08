#!/usr/bin/env python3
"""Idea 28 — the hydrophobic ranking failure, and whether side-chain volume explains it.

THE FINDING THIS QUANTIFIES
---------------------------
The model compresses mutations to hydrophobic residues about twice as hard as others:
corr(compression, hydropathy) = -0.734, consistent across all ten checkpoints. Crucially
the RANKING falls by the same amount -- so this is NOT calibration. It is missing
information.

THE PROPOSED MECHANISM
----------------------
The dataset carries four backbone atoms per residue (N, CA, C, CB). There are no side
chains. Whether a large hydrophobic residue can be buried is a question about side-chain
PACKING, and packing is invisible from CB alone.

THE DISCRIMINATING PREDICTION
-----------------------------
If the cause were hydrophobicity itself, the failure would track a hydropathy scale.
If the cause is packing, it should track SIDE-CHAIN VOLUME instead -- and tryptophan and
tyrosine become the test: both are bulky, but W is hydrophobic while Y is amphipathic.

    tracks hydropathy -> W fails, Y does not
    tracks volume     -> BOTH fail

The second was observed. This script tests it properly, with partial correlations, rather
than asserting it from two residues.

USAGE
    python hydrophobic_failure.py eval.csv [eval2.csv ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EXCLUDE = {"2K5H"}

# Kyte-Doolittle hydropathy.
KD = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
      "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
      "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}

# Side-chain van der Waals volume, A^3 (Zamyatnin 1972). Glycine has no side chain.
VOL = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
       "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
       "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
       "W": 227.8, "Y": 193.6}


def parse_mut(s):
    s = str(s).strip()
    if len(s) < 3:
        return None, None, None
    try:
        return s[0], int(s[1:-1]), s[-1]
    except ValueError:
        return None, None, None


def per_target_residue(csv):
    """Group mutations by the residue they introduce, and score each group."""
    df = pd.read_csv(csv)
    df = df[~df["protein"].isin(EXCLUDE)].dropna(subset=["ddG", "pred_ddG"])
    parsed = df["mut"].apply(parse_mut)
    df["wt_aa"] = [p[0] for p in parsed]
    df["pos"] = [p[1] for p in parsed]
    df["mt_aa"] = [p[2] for p in parsed]
    df = df.dropna(subset=["mt_aa"])

    rows = []
    for aa, g in df.groupby("mt_aa"):
        if aa not in KD or len(g) < 20 or g["ddG"].std(ddof=0) == 0:
            continue
        slope = float(np.polyfit(g["ddG"], g["pred_ddG"], 1)[0])
        rows.append(dict(
            target_aa=aa, n=len(g),
            slope=slope,                                    # compression
            pcc=float(pearsonr(g["ddG"], g["pred_ddG"])[0]),  # ranking
            hydropathy=KD[aa], volume=VOL[aa],
            mean_true=float(g["ddG"].mean()),
            mean_pred=float(g["pred_ddG"].mean()),
        ))
    return pd.DataFrame(rows).sort_values("volume")


def partial_corr(x, y, z):
    """corr(x, y) with z regressed out of both. Separates volume from hydropathy."""
    x, y, z = map(np.asarray, (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(pearsonr(rx, ry)[0])


def report(t, label):
    print(f"\n{'=' * 78}\n{label}   ({len(t)} residue types, "
          f"{int(t.n.sum())} mutations)\n{'=' * 78}")
    print(f"{'aa':>3} {'n':>6} {'vol':>7} {'hydro':>7} {'slope':>8} {'pcc':>8}")
    for _, r in t.iterrows():
        flag = "  <-- bulky" if r.volume > 180 else ""
        print(f"{r.target_aa:>3} {int(r.n):>6} {r.volume:>7.1f} {r.hydropathy:>7.1f} "
              f"{r.slope:>8.4f} {r.pcc:>8.4f}{flag}")

    print("\n--- which variable does the failure track? ---")
    for var in ("hydropathy", "volume"):
        rs = pearsonr(t[var], t.slope)[0]
        rp = pearsonr(t[var], t.pcc)[0]
        print(f"  {var:<11} vs slope {rs:+.4f}   vs ranking(pcc) {rp:+.4f}")

    print("\n--- partial correlations (each controlling for the other) ---")
    pv_s = partial_corr(t.volume, t.slope, t.hydropathy)
    ph_s = partial_corr(t.hydropathy, t.slope, t.volume)
    pv_p = partial_corr(t.volume, t.pcc, t.hydropathy)
    ph_p = partial_corr(t.hydropathy, t.pcc, t.volume)
    print(f"  volume | hydropathy   vs slope {pv_s:+.4f}   vs ranking {pv_p:+.4f}")
    print(f"  hydropathy | volume   vs slope {ph_s:+.4f}   vs ranking {ph_p:+.4f}")

    print("\n--- the W / Y discriminator ---")
    for aa in ("W", "Y", "F", "L", "I"):
        r = t[t.target_aa == aa]
        if len(r):
            r = r.iloc[0]
            print(f"  {aa}  vol {r.volume:>6.1f}  hydro {r.hydropathy:>+5.1f}  "
                  f"slope {r.slope:.4f}  pcc {r.pcc:.4f}")
    w = t[t.target_aa == "W"]
    y = t[t.target_aa == "Y"]
    med = t.slope.median()
    if len(w) and len(y):
        wf, yf = w.slope.iloc[0] < med, y.slope.iloc[0] < med
        print()
        if wf and yf:
            print("  BOTH W and Y are compressed. W is hydrophobic and Y is amphipathic,")
            print("  so a hydropathy explanation cannot cover both. The failure tracks")
            print("  SIDE-CHAIN VOLUME -- i.e. packing, which four backbone atoms cannot")
            print("  express. This is missing information, not miscalibration, and it is")
            print("  the case for wiring the side-chain block.")
        elif wf and not yf:
            print("  W is compressed and Y is not. The failure tracks HYDROPATHY rather")
            print("  than volume, and the side-chain-packing explanation is not supported.")
        else:
            print("  Neither is compressed relative to the median -- the pattern does not")
            print("  reproduce on this checkpoint. Check the basis before interpreting.")

    print("\n--- is it calibration or ranking? ---")
    print("  If only the slope degrades, a calibration lever can fix it.")
    print("  If the RANKING degrades too, the information is absent and no calibration")
    print("  lever can recover it.")
    lo = t[t.slope < t.slope.median()]
    hi = t[t.slope >= t.slope.median()]
    print(f"  compressed half:   mean pcc {lo.pcc.mean():.4f}")
    print(f"  uncompressed half: mean pcc {hi.pcc.mean():.4f}")
    gap = hi.pcc.mean() - lo.pcc.mean()
    print(f"  ranking gap {gap:+.4f}", end="  ")
    print("-> ranking falls with compression: MISSING INFORMATION"
          if gap > 0.05 else "-> ranking holds: calibration only")


def main(argv):
    for csv in argv:
        try:
            t = per_target_residue(csv)
        except Exception as exc:
            print(f"ERROR {csv}: {exc}", file=sys.stderr)
            continue
        if len(t) < 8:
            print(f"SKIP {csv}: only {len(t)} residue types with >= 20 mutations")
            continue
        report(t, os.path.basename(csv))
        out = f"results/HYDROPHOBIC_{os.path.basename(csv).replace('.csv', '')}.tsv"
        os.makedirs("results", exist_ok=True)
        t.to_csv(out, sep="\t", index=False)
        print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
