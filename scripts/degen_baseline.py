"""Degenerate baselines for the W5/dG arm.

The metric rule: W5 burial is scored on dG, so dG MAE must be compared against
what a MODEL-FREE predictor achieves. If a constant beats the model, the dG
number is not evidence of anything.
"""
import sys, csv, math
from collections import defaultdict

path = sys.argv[1]
rows = defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        rows[r['protein']].append((float(r['deltaG']), float(r['pred_deltaG'])))
dg  = [t[0] for v in rows.values() for t in v]
pdg = [t[1] for v in rows.values() for t in v]
n = len(dg)
mean_dg = sum(dg)/n
srt = sorted(dg); median_dg = srt[n//2]
def mae(p): return sum(abs(a-b) for a, b in zip(dg, p))/n
print('rows=%d proteins=%d' % (n, len(rows)))
print('  MODEL           dG MAE = %.4f' % mae(pdg))
print('  const=mean      dG MAE = %.4f   (mean  =%.4f)' % (mae([mean_dg]*n), mean_dg))
print('  const=median    dG MAE = %.4f   (median=%.4f)' % (mae([median_dg]*n), median_dg))
# per-protein-mean oracle: the strongest trivial competitor (uses true labels)
pp = []
for v in rows.values():
    m = sum(t[0] for t in v)/len(v)
    pp += [abs(t[0]-m) for t in v]
print('  per-prot mean   dG MAE = %.4f   (ORACLE, uses true labels)' % (sum(pp)/len(pp)))
