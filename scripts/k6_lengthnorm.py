"""K6: would --dg_length_norm shrink b_p? Answer it from existing predictions, no GPU.

b_p is the WT error, pred_dG(WT) - true_dG(WT), one scalar per protein. The energy head is a
plain SUM over residues, so E is EXTENSIVE: a near-constant per-residue bias beta contributes
beta*N to E and therefore beta*N to b_p. If that is what b_p is, dividing by N (or sqrt N)
should collapse its spread.

This does NOT require training. The flag divides the PREDICTION, so its effect on b_p can be
simulated exactly on predictions we already have:
    none  : b_p = pred - true
    n     : b_p = pred/N - true
    sqrtn : b_p = pred/sqrt(N) - true
The honest caveat: a trained model would adapt its scale, so this bounds the mechanism rather
than predicting the trained outcome. What it CAN settle is whether b_p is length-driven at all.
"""
import csv, math, os, glob, json, collections

DIRS = ['/groups/keasar_group/casp15/meytav/protein_tensors',
        '/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors']

def length_of(p):
    for D in DIRS:
        f = os.path.join(D, p, 'mask_tensor.pt')
        if os.path.exists(f):
            import torch
            m = torch.load(f, map_location='cpu', weights_only=False)
            return int(m.sum())
    return None

def std(v):
    n = len(v); m = sum(v)/n
    return math.sqrt(sum((x-m)**2 for x in v)/n)

def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else float('nan')

out = {}
print('%-46s %8s %8s %8s %8s' % ('eval csv', 'std none', 'std /n', 'std /sqN', 'r(N,b_p)'))
print('-' * 84)
for f in sorted(glob.glob('eval_results/abl_*.csv'))[:12]:
    wt = {}
    for r in csv.DictReader(open(f)):
        if float(r['ddG']) == 0.0 and r['protein'] not in wt:
            wt[r['protein']] = (float(r['pred_deltaG']), float(r['deltaG']))
    Ns, b_none, b_n, b_sq = [], [], [], []
    for p, (pr, tr) in wt.items():
        N = length_of(p)
        if not N: continue
        Ns.append(float(N))
        b_none.append(pr - tr)
        b_n.append(pr/N - tr)
        b_sq.append(pr/math.sqrt(N) - tr)
    if len(Ns) < 5: continue
    out[os.path.basename(f)] = dict(n=len(Ns), std_none=std(b_none), std_n=std(b_n),
                                    std_sqrtn=std(b_sq), r_N_bp=pear(Ns, b_none))
    print('%-46s %8.4f %8.4f %8.4f %+8.4f' % (os.path.basename(f)[4:-4][:46],
          std(b_none), std(b_n), std(b_sq), pear(Ns, b_none)))

if out:
    ks = list(out)
    print('\nMEAN over %d CSVs:  std(b_p) none=%.4f  /n=%.4f  /sqrtn=%.4f   corr(N,b_p)=%+.4f'
          % (len(ks), sum(out[k]['std_none'] for k in ks)/len(ks),
             sum(out[k]['std_n'] for k in ks)/len(ks),
             sum(out[k]['std_sqrtn'] for k in ks)/len(ks),
             sum(out[k]['r_N_bp'] for k in ks)/len(ks)))
    json.dump(out, open('results/k6_lengthnorm.json', 'w'), indent=1)
    print('wrote results/k6_lengthnorm.json')
