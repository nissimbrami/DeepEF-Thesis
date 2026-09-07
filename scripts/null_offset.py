"""THE NULL TEST the critic demanded, run before anything else is believed.

Claim under attack: "pooled ddG PCC 0.59 -> 0.71 after removing a per-protein
offset" is evidence that the model is miscalibrated but informative.

Null hypothesis: that lift is an ARITHMETIC PROPERTY of the estimator. Subtracting a
per-protein constant FITTED ON THE TEST LABELS removes between-protein variance from a
pooled correlation whether or not the constant means anything. A perfectly-ranking
predictor polluted with PURE RANDOM per-protein noise should show the same lift.

If the null reproduces our numbers, the headline result is about the estimator, not the
model, and only a FEATURE-PREDICTED offset can rescue it.

Uses the REAL per-protein structure (sizes, true ddG values) from the reference eval CSV,
so the only thing synthetic is the predictor itself.
"""
import csv, math, random
from collections import defaultdict

random.seed(0)
SRC = 'eval_results/abl_calib_ctrl_repro2_e14.csv'
prot = defaultdict(list)
with open(SRC) as f:
    for r in csv.DictReader(f):
        prot[r['protein']].append(float(r['ddG']))

def pearson(x, y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<=0 or syy<=0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

STD_B = 1.5741   # measured std(b_p), from calc_bp.py
A_P   = 0.4990   # measured median compression slope

def trial(noise_sd, use_slope):
    X, Y, per = [], [], []
    for p, ddgs in prot.items():
        b = random.gauss(0, STD_B)          # PURE RANDOM per-protein offset: no meaning
        a = A_P if use_slope else 1.0
        xs, ys = [], []
        for t in ddgs:
            pred = a*t + b + random.gauss(0, noise_sd)
            xs.append(t); ys.append(pred)
        X += xs; Y += ys
        per.append(pearson(xs, ys))
    pooled = pearson(X, Y)
    # oracle offset removal: subtract each protein's OWN fitted mean residual
    Xc, Yc = [], []
    i = 0
    for p, ddgs in prot.items():
        n=len(ddgs); xs=X[i:i+n]; ys=Y[i:i+n]; i+=n
        off = sum(y-x for x,y in zip(xs,ys))/n
        Xc += xs; Yc += [y-off for y in ys]
    return pooled, pearson(Xc, Yc), sum(q for q in per if q==q)/len([q for q in per if q==q])

print('%-46s %8s %8s %8s' % ('condition','pooled','offset-removed','per-prot'))
print('-'*76)
for sd in (0.0, 0.5, 1.0, 1.5):
    for slope in (False, True):
        po, oc, pp = trial(sd, slope)
        tag = 'noise=%.1f slope=%s' % (sd, '0.499' if slope else '1.0  ')
        print('%-46s %8.4f %8.4f %8.4f' % (tag, po, oc, pp))
print()
print('OUR MEASURED NUMBERS:                          pooled ~0.5900   removed 0.70-0.72   per-prot 0.7980')
