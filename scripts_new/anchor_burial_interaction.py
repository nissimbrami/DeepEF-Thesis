#!/usr/bin/env python3
"""Idea 15 — do the WT anchor and the burial feature COMPOSE, or do they overlap?

WHY THIS IS THE MOST VALUABLE UNRUN EXPERIMENT
----------------------------------------------
The two strongest single levers act through DIFFERENT measured channels:

    --wt_anchor_weight   raised a_p from 0.499 to 0.775 (+55%)
    --burial_features    raised corr(burial, prediction) from 0.284 to 0.331,
                         toward the 0.556 present in the truth

If those channels are independent, the effects add and the combination lands well above
either. If they are two routes to the same correction, the combination equals the better
of the two and we have learned that the model had ONE deficiency, not two.

Either answer is a result. Neither is available from single-lever runs.

WHAT THIS SCRIPT DOES
    Scores the 2x2 (anchor on/off x burial on/off) and decomposes the interaction:

        interaction = (both - anchor) - (burial - baseline)

        interaction ~ 0    -> additive: independent channels
        interaction < 0    -> overlapping: they correct the same thing
        interaction > 0    -> synergistic: each enables the other

USAGE
    python anchor_burial_interaction.py \
        --baseline eval_base.csv --anchor eval_anchor.csv \
        --burial eval_burial.csv --both eval_both.csv \
        [--sigma 0.0072]
"""
import sys
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

EXCLUDE = {"2K5H"}
MIN_MUTS = 3


def metrics(csv):
    """pooled, per-protein, slope and offset dispersion on the canonical basis."""
    df = pd.read_csv(csv)
    df = df[~df["protein"].isin(EXCLUDE)].dropna(subset=["ddG", "pred_ddG"])

    pooled = float(pearsonr(df["ddG"], df["pred_ddG"])[0])

    pp, slopes, offsets = [], [], []
    for _, g in df.groupby("protein"):
        if len(g) < MIN_MUTS or g["ddG"].std(ddof=0) == 0:
            continue
        pp.append(pearsonr(g["ddG"], g["pred_ddG"])[0])
        slopes.append(np.polyfit(g["ddG"], g["pred_ddG"], 1)[0])
        offsets.append((g["pred_ddG"] - g["ddG"]).mean())

    return dict(
        pooled=pooled,
        pp=float(np.mean(pp)),
        a_p_median=float(np.median(slopes)),
        std_b_p=float(np.std(offsets, ddof=0)),
        n_proteins=int(df["protein"].nunique()),
    )


def interaction_report(base, anch, bur, both, sigma):
    """Two-factor decomposition on every metric that matters."""
    print(f"{'metric':<14} {'baseline':>10} {'anchor':>10} {'burial':>10} "
          f"{'both':>10} {'  main_A':>9} {'  main_B':>9} {'interact':>9}")
    print("-" * 92)

    verdicts = {}
    for key, better_is_up in [("pooled", True), ("pp", True),
                              ("a_p_median", True), ("std_b_p", False)]:
        b0, a1, b1, ab = base[key], anch[key], bur[key], both[key]
        main_a = a1 - b0                       # anchor alone
        main_b = b1 - b0                       # burial alone
        inter = (ab - a1) - (b1 - b0)          # what the second adds on top of the first
        print(f"{key:<14} {b0:>10.4f} {a1:>10.4f} {b1:>10.4f} {ab:>10.4f} "
              f"{main_a:>+9.4f} {main_b:>+9.4f} {inter:>+9.4f}")
        verdicts[key] = (main_a, main_b, inter, better_is_up)

    print()
    thr = 2 * sigma / np.sqrt(3)               # resolvable effect at 3 seeds
    print(f"Resolution threshold at 3 seeds: {thr:.4f}  (2*sigma/sqrt(3), sigma={sigma})")
    print()

    ma, mb, inter, up = verdicts["pooled"]
    sign = 1.0 if up else -1.0

    if abs(inter) < thr:
        print("ADDITIVE. The interaction is below the resolution threshold, so the two")
        print("levers act through independent channels and their effects add. Report both")
        print("as separate contributions and keep both in the final model.")
    elif sign * inter < -thr:
        print("OVERLAPPING. The combination gains less than the sum. The anchor and the")
        print("burial feature are two routes to the SAME correction -- the model had one")
        print("deficiency, not two. Keep the cheaper lever; report the redundancy, which is")
        print("itself a finding about what the model was missing.")
    else:
        print("SYNERGISTIC. The combination exceeds the sum: each lever makes the other")
        print("more effective. That is the strongest possible outcome here and it should")
        print("be stated explicitly -- it means burial supplies information the anchor")
        print("needs in order to calibrate, or the reverse.")

    print()
    if verdicts["a_p_median"][2] < -thr and verdicts["pooled"][2] > 0:
        print("NOTE  the slope interaction is negative while pooled still rises. That is")
        print("      the anchor+designed failure mode returning: the score improves while")
        print("      the model compresses. Check a_p per protein before accepting.")

    if both["std_b_p"] > base["std_b_p"]:
        print("WARN  std(b_p) is HIGHER in the combination than at baseline. Since 96.3%")
        print("      of b_p is an inert global constant, a rise in the dispersion is a")
        print("      real regression even if pooled improved.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--burial", required=True)
    p.add_argument("--both", required=True)
    p.add_argument("--sigma", type=float, default=0.0072)
    a = p.parse_args()

    arms = {}
    for name, path in [("baseline", a.baseline), ("anchor", a.anchor),
                       ("burial", a.burial), ("both", a.both)]:
        arms[name] = metrics(path)
        if arms[name]["n_proteins"] != 27:
            print(f"WARN {name}: {arms[name]['n_proteins']} proteins, expected 27 "
                  f"-- the arms are not on the same basis", file=sys.stderr)

    interaction_report(arms["baseline"], arms["anchor"],
                       arms["burial"], arms["both"], a.sigma)

    pd.DataFrame(arms).T.to_csv("results/ANCHOR_BURIAL_2x2.tsv", sep="\t")
    print("\nWritten: results/ANCHOR_BURIAL_2x2.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
