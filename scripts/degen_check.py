#!/usr/bin/env python3
"""Degenerate-baseline check for the dG arm (the --dg_length_norm lesson, FINDINGS 5.3).

A lever can shrink std(b_p) by DELETING the prediction rather than removing a bias.
The tell: as pred_dG -> const, b_p -> -true_dG, so std(b_p) -> std(true WT dG) = 0.9042,
and the dG correlation collapses. This prints the diagnostics that separate the two cases.

Usage: degen_check.py <csv> [<csv> ...]
"""
import sys, csv, math
from collections import OrderedDict

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def std(v):
    n = len(v)
    if n < 2: return float('nan')
    m = sum(v)/n
    return math.sqrt(sum((a-m)**2 for a in v)/(n-1))

for path in sys.argv[1:]:
    prots = OrderedDict()
    with open(path) as f:
        for row in csv.DictReader(f):
            prots.setdefault(row['protein'], []).append(
                (float(row['deltaG']), float(row['pred_deltaG'])))
    wt_true = [v[0][0] for v in prots.values()]
    wt_pred = [v[0][1] for v in prots.values()]
    bp      = [p-t for t, p in zip(wt_true, wt_pred)]
    print("="*66)
    print(f"  {path.split('/')[-1]}")
    print(f"  n_proteins                     = {len(prots)}")
    print(f"  std(true WT dG)                = {std(wt_true):.4f}   [degenerate target 0.9042]")
    print(f"  std(pred WT dG)                = {std(wt_pred):.4f}   [-> 0 if prediction deleted]")
    print(f"  std(b_p) = std(pred-true)      = {std(bp):.4f}")
    print(f"  corr(pred WT, true WT)         = {pearson(wt_pred, wt_true):.4f}   [-> 0 if deleted]")
    print(f"  corr(b_p, true WT dG)          = {pearson(bp, wt_true):.4f}   [-> -1 if deleted]")
    # the decisive ratio: a genuine reference-state fix keeps spread AND correlation
    # NOTE: the healthy baseline itself has corr(pred WT, true WT) ~ 0.07, so LOW CORRELATION IS
    # NOT the degeneracy signature here -- the model never predicted absolute WT dG well.
    # The --dg_length_norm signature is SPREAD COLLAPSE: std(pred WT) -> 0 while std(b_p) -> 0.904.
    sp = std(wt_pred); sb = std(bp)
    degen = (sp < 0.35) or (abs(sb - std(wt_true)) < 0.06 and sp < 0.5)
    print(f"  VERDICT: {'DEGENERATE (pred spread collapsed)' if degen else 'prediction alive (NOT degenerate)'}")
