"""ProtT5 embedding probe: can the per-protein offset b_p (or slope a_p) be read out
of the 1024-d sequence representation the model already receives?

Q1 contextual-vs-lookup: decides whether W6 descriptors are a linear-algebra question.
Q2 LOPO ridge from the MEAN embedding -> b_p. n=28, p=1024. Expect overfit; report null.
Q3 embedding distance -> |b_p| difference. Better powered than regression at n=28.
Q4 per-dimension correlation, Bonferroni 0.05/1024.

n=28 => |r|<0.374 is indistinguishable from zero at p=.05.
"""
import os, sys, csv, math, json, glob
import numpy as np, torch

ROOTS = ['/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors',
         '/groups/keasar_group/casp15/meytav/protein_tensors']
OUT = '/home/nissimb/DeepPEF/results'

def emb_dir(p):
    for r in ROOTS:
        d = os.path.join(r, p, 'prott5_embeddings')
        if os.path.isdir(d) and os.listdir(d):
            return d
    return None

def fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan'), float('nan')
    sxx = ((x - x.mean())**2).sum()
    if sxx <= 0: return float('nan'), float('nan')
    a = ((x - x.mean())*(y - y.mean())).sum()/sxx
    return a, y.mean() - a*x.mean()

def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan')
    sx = x.std(); sy = y.std()
    if sx <= 0 or sy <= 0: return float('nan')
    return float(((x-x.mean())*(y-y.mean())).mean()/(sx*sy))

CSVS = sorted(glob.glob('/home/nissimb/DeepPEF/eval_results/*.csv'))
print('eval CSVs: %d' % len(CSVS))

def calib(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r['protein'], []).append(
                (float(r['deltaG']), float(r['pred_deltaG']),
                 float(r['ddG']), float(r['pred_ddG'])))
    out = {}
    for p, v in rows.items():
        dg = [t[0] for t in v]; pdg = [t[1] for t in v]
        ddg = [t[2] for t in v]; pddg = [t[3] for t in v]
        wt = [i for i, t in enumerate(v) if t[2] == 0.0]
        b_wt = (pdg[wt[0]] - dg[wt[0]]) if wt else float('nan')
        a_p, _ = fit(ddg, pddg)
        out[p] = dict(b_p=b_wt, a_p=a_p, pcc=pear(ddg, pddg), n=len(v))
    return out

allc = [calib(c) for c in CSVS]
prots = sorted(set.intersection(*[set(c) for c in allc]))
print('proteins common to all CSVs: %d' % len(prots))
bp = np.array([np.nanmean([c[p]['b_p'] for c in allc]) for p in prots])
ap = np.array([np.nanmean([c[p]['a_p'] for c in allc]) for p in prots])
pcc = np.array([np.nanmean([c[p]['pcc'] for c in allc]) for p in prots])
print('b_p: mean %.4f std %.4f  min %.4f max %.4f' % (bp.mean(), bp.std(ddof=1), bp.min(), bp.max()))
print('a_p: mean %.4f std %.4f  median %.4f' % (ap.mean(), ap.std(ddof=1), np.median(ap)))

print('')
print('=== Q1 contextual vs lookup ===')
AA = 'ACDEFGHIKLMNPQRSTVWY'
MEAN = {}
q1 = {}
for p in prots:
    d = emb_dir(p)
    shards = sorted(os.listdir(d))
    e0 = torch.load(os.path.join(d, shards[0]), map_location='cpu', weights_only=False)
    wt = e0[0].float().numpy()
    MEAN[p] = wt.mean(0)
    q1[p] = dict(L=int(wt.shape[0]), dim=int(wt.shape[1]), n_shards=len(shards),
                 n_variants_shard0=int(e0.shape[0]))
L0 = q1[prots[0]]
print('shape per shard = [n_variants, L, %d]' % L0['dim'])
print('example %s: L=%d shards=%d variants/shard=%d' % (prots[0], L0['L'], L0['n_shards'], L0['n_variants_shard0']))

def lookup_test(p):
    d = emb_dir(p)
    ohp = None
    for r in ROOTS:
        c = os.path.join(r, p, 'one_hot_encodings.pt')
        if os.path.exists(c):
            ohp = c; break
    if ohp is None: return None
    e = torch.load(os.path.join(d, sorted(os.listdir(d))[0]), map_location='cpu', weights_only=False)
    o = torch.load(ohp, map_location='cpu', weights_only=False)
    v = e[0].float(); ids = o[0][:, :20].argmax(-1)
    L = min(v.shape[0], ids.shape[0]); v = v[:L]; ids = ids[:L]
    worst = 0.0; nrm = float(v.norm(dim=1).mean())
    for t in range(20):
        m = ids == t
        if int(m.sum()) < 2: continue
        s = v[m]
        worst = max(worst, float(torch.cdist(s, s).max()))
    return worst, nrm

print('')
print('within-residue-type max distance vs mean ||v|| (a lookup would give exactly 0):')
lt = {}
for p in prots[:10]:
    r = lookup_test(p)
    if r:
        lt[p] = r
        print('  %-20s within-type maxdist=%7.4f   mean||v||=%6.4f' % (p, r[0], r[1]))

print('')
print('=== Q2 LOPO ridge: mean embedding (1024-d) -> b_p ===')
X = np.stack([MEAN[p] for p in prots])
n, dim = X.shape
print('X shape = (%d, %d)' % (n, dim))

def lopo_ridge(X, y, lam):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        Xt = X[tr]; yt = y[tr]
        mu = Xt.mean(0); sd = Xt.std(0); sd[sd < 1e-8] = 1.0
        Z = (Xt - mu)/sd; zc = (X[i] - mu)/sd
        ym = yt.mean()
        A = Z.T @ Z + lam*np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (yt - ym))
        pred[i] = zc @ w + ym
    return pred

def r2(pred, y):
    return float(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum())

LAMS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
res_q2 = {}
for tgt_name, tgt in (('b_p', bp), ('a_p', ap)):
    print('  target %s:' % tgt_name)
    best = None
    for lam in LAMS:
        pr = lopo_ridge(X, tgt, lam)
        rr = r2(pr, tgt); pc = pear(pr, tgt)
        print('    lam=%-9.0e heldout R2=%+.4f  PCC=%+.4f' % (lam, rr, pc))
        if best is None or rr > best[1]: best = (lam, rr, pc)
    res_q2[tgt_name] = dict(best_lam=best[0], best_r2=best[1], best_pcc=best[2])
    rng = np.random.RandomState(0)
    null = []
    for _ in range(200):
        yp = rng.permutation(tgt)
        null.append(r2(lopo_ridge(X, yp, best[0]), yp))
    null = np.array(null)
    pv = float((null >= best[1]).mean())
    print('    PERM NULL (200 perms, lam=%.0e): mean R2=%+.4f  p95=%+.4f  p(null>=obs)=%.3f'
          % (best[0], null.mean(), np.percentile(null, 95), pv))
    res_q2[tgt_name].update(null_mean=float(null.mean()),
                            null_p95=float(np.percentile(null, 95)), perm_p=pv)

print('')
print('=== Q3 embedding distance -> |b_p| difference ===')
Xn = X/np.linalg.norm(X, axis=1, keepdims=True)
De = 1 - Xn @ Xn.T
Db = np.abs(bp[:, None] - bp[None, :])
Da = np.abs(ap[:, None] - ap[None, :])
iu = np.triu_indices(n, 1)
res_q3 = {}
for nm, Dt in (('b_p', Db), ('a_p', Da)):
    r = pear(De[iu], Dt[iu])
    rng = np.random.RandomState(1); null = []
    for _ in range(5000):
        pm = rng.permutation(n)
        null.append(pear(De[iu], Dt[np.ix_(pm, pm)][iu]))
    null = np.array(null); pv = float((np.abs(null) >= abs(r)).mean())
    print('  Mantel r(emb dist, |d%s|) = %+.4f  perm p=%.4f  (pairs=%d, proteins=%d)'
          % (nm, r, pv, len(iu[0]), n))
    res_q3[nm] = dict(mantel_r=r, perm_p=pv)

print('  1-NN readout:')
res_nn = {}
for nm, tgt in (('b_p', bp), ('a_p', ap)):
    nnerr = []; rnderr = []
    rng = np.random.RandomState(2)
    for i in range(n):
        dd = De[i].copy(); dd[i] = np.inf
        j = int(dd.argmin())
        nnerr.append(abs(tgt[i]-tgt[j]))
        others = [k for k in range(n) if k != i]
        rnderr.append(float(np.mean([abs(tgt[i]-tgt[k]) for k in others])))
    nnerr = np.array(nnerr); rnderr = np.array(rnderr)
    d = rnderr - nnerr; obs = float(d.mean())
    null = np.array([float((d*rng.choice([-1, 1], n)).mean()) for _ in range(10000)])
    pv = float((null >= obs).mean())
    print('    %s: 1-NN mean|diff|=%.4f  random-other=%.4f  gain=%+.4f  p=%.4f'
          % (nm, nnerr.mean(), rnderr.mean(), obs, pv))
    res_nn[nm] = dict(nn=float(nnerr.mean()), rnd=float(rnderr.mean()), gain=obs, p=pv)

print('')
print('=== Q4 per-dimension correlation, Bonferroni 0.05/1024 ===')
try:
    import scipy.stats as st
    HAVE = True
except Exception:
    HAVE = False
print('scipy available: %s' % HAVE)

def pval_r(r, n):
    if not HAVE or not np.isfinite(r) or abs(r) >= 1: return float('nan')
    t = r*math.sqrt((n-2)/(1-r*r))
    return float(2*st.t.sf(abs(t), n-2))

res_q4 = {}
BONF = 0.05/1024
for nm, tgt in (('b_p', bp), ('a_p', ap)):
    rs = np.array([pear(X[:, k], tgt) for k in range(dim)])
    ps = np.array([pval_r(r, n) for r in rs])
    order = np.argsort(-np.abs(rs))
    print('  target %s (n=%d; p<.05 needs |r|>0.374)' % (nm, n))
    top = []
    for k in order[:8]:
        sig = 'BONF-SIG' if (HAVE and ps[k] < BONF) else ('p<.05' if (HAVE and ps[k] < .05) else 'ns')
        print('    dim %4d  r=%+.4f  p=%.3e  %s' % (k, rs[k], ps[k], sig))
        top.append(dict(dim=int(k), r=float(rs[k]), p=float(ps[k]), sig=sig))
    nsig05 = int((ps < .05).sum()) if HAVE else -1
    nbonf = int((ps < BONF).sum()) if HAVE else -1
    print('    dims p<.05: %d (chance expectation %.0f)   Bonferroni survivors: %d'
          % (nsig05, 0.05*dim, nbonf))
    res_q4[nm] = dict(top=top, n_p05=nsig05, expected_p05=0.05*dim, n_bonf=nbonf,
                      max_abs_r=float(np.abs(rs).max()))

json.dump(dict(n_proteins=int(n), dim=int(dim), proteins=prots,
               b_p=bp.tolist(), a_p=ap.tolist(), pcc=pcc.tolist(),
               q1=q1, lookup_test=dict((k, list(v)) for k, v in lt.items()),
               q2=res_q2, q3=res_q3, q3_nn=res_nn, q4=res_q4),
          open(os.path.join(OUT, 'embedding_probe.json'), 'w'), indent=1)
print('')
print('wrote results/embedding_probe.json')
