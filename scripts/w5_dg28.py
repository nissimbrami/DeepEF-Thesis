"""W5-on-dG at full n=28, using the UNION tensor resolver.

Recovered root: MutationFineTuning/test_protein_tensors holds 27/28 (incl. all 9 that
protein_tensors lacked); protein_tensors holds r18_3_TrROS_Hall which the other lacks.
The 18 proteins present in BOTH with coords are bit-identical, so the union is one
consistent data source, not a mix of two preprocessings.
"""
import csv, json, math, os, sys
sys.path.insert(0, '/home/nissimb/DeepPEF')
import torch
import train_utils as tu

EV = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/abl_calib_ctrl_repro2_e14.csv'
ROOTS = ['/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors',
         '/groups/keasar_group/casp15/meytav/protein_tensors']

def resolve(p):
    for r in ROOTS:
        c = os.path.join(r, p, 'coords_tensor.pt')
        if os.path.exists(c):
            return r, c, os.path.join(r, p, 'mask_tensor.pt')
    return None, None, None

def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def spear(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        for pos, i in enumerate(s): r[i] = pos
        return r
    return pear(rank(x), rank(y))

def pval(r, n):
    # two-sided t-test on r, df = n-2
    if n <= 3 or abs(r) >= 1: return float('nan')
    t = abs(r)*math.sqrt((n-2)/(1-r*r)); df = n-2
    # Student t survival via incomplete beta (continued fraction)
    x = df/(df+t*t)
    def betacf(a,b,x):
        MAX=200; EPS=3e-16; FPMIN=1e-300
        qab,qap,qam=a+b,a+1.0,a-1.0
        c=1.0; d=1.0-qab*x/qap
        if abs(d)<FPMIN: d=FPMIN
        d=1.0/d; h=d
        for m in range(1,MAX+1):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2))
            d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; de=d*c; h*=de
            if abs(de-1.0)<EPS: break
        return h
    def betai(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        lb=(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
        bt=math.exp(lb)
        if x < (a+1)/(a+b+2): return bt*betacf(a,b,x)/a
        return 1.0-bt*betacf(b,a,1-x)/b
    return betai(df/2.0, 0.5, x)

rows = {}
with open(EV) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r['protein'], []).append(
            (float(r['deltaG']), float(r['pred_deltaG']), float(r['ddG'])))
bp = {}
for p, v in rows.items():
    wt = [t for t in v if t[2] == 0.0]
    if wt: bp[p] = wt[0][1] - wt[0][0]

feat = {}; src = {}; missing = []
for p in bp:
    root, cf, mf = resolve(p)
    if cf is None:
        missing.append(p); continue
    src[p] = root
    x = torch.load(cf, map_location='cpu', weights_only=False)
    x = torch.as_tensor(x).float()
    m = (torch.as_tensor(torch.load(mf, map_location='cpu', weights_only=False)).float()
         if os.path.exists(mf) else torch.ones(x.shape[0]))
    b = tu.compute_burial(x, m)
    v = b[m > 0].flatten()
    if v.numel() == 0: continue
    feat[p] = dict(mean_burial=float(v.mean()),
                   frac_buried=float((v > v.median()).float().mean()),
                   length=float(int(m.sum())))

print('resolved %d / %d ; missing: %s' % (len(feat), len(bp), missing or 'NONE'))
from collections import Counter
print('source roots:', {os.path.basename(k): v for k, v in Counter(src.values()).items()})

ps = sorted(feat)
n = len(ps)
out = {'eval_csv': EV, 'n': n, 'roots': ROOTS, 'source': src, 'correlations': {}}
tgt = {'b_p': [bp[p] for p in ps], 'abs_b_p': [abs(bp[p]) for p in ps]}
print('\n%-16s %-10s %8s %8s %9s' % ('feature', 'target', 'pearson', 'spearman', 'p(pear)'))
print('-' * 56)
for fn in ('mean_burial', 'frac_buried', 'length'):
    xs = [feat[p][fn] for p in ps]
    for tn, ys in tgt.items():
        r, rs = pear(xs, ys), spear(xs, ys)
        pv = pval(r, n)
        out['correlations']['%s~%s' % (fn, tn)] = {'pearson': r, 'spearman': rs, 'n': n, 'p': pv}
        print('%-16s %-10s %8.3f %8.3f %9.4f' % (fn, tn, r, rs, pv))

thr = 1.96/math.sqrt(n-3)
print('\nn=%d: Fisher-z |r| threshold at p=0.05 is %.3f' % (n, thr))
out['threshold_p05'] = thr

# n=19 subset (the old sample) for a like-for-like comparison
old = sorted([p for p in ps if src[p] == ROOTS[1] or os.path.exists(
    os.path.join(ROOTS[1], p, 'coords_tensor.pt'))])
out['n19_subset'] = old
print('\n-- same stats on the OLD n=%d subset (proteins protein_tensors had) --' % len(old))
for fn in ('mean_burial', 'frac_buried'):
    xs = [feat[p][fn] for p in old]
    for tn in ('b_p', 'abs_b_p'):
        ys = [bp[p] if tn == 'b_p' else abs(bp[p]) for p in old]
        r = pear(xs, ys)
        out['correlations']['n19_%s~%s' % (fn, tn)] = {'pearson': r, 'n': len(old)}
        print('%-16s %-10s %8.3f' % (fn, tn, r))

json.dump(out, open('results/w5_on_dg_n28.json', 'w'), indent=1)
print('\nwrote results/w5_on_dg_n28.json')
