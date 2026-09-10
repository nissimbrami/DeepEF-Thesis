"""IFUM equilibrium-mixture feature test. No GPU. Plan: IFUM_FEATURE_TEST_PLAN.md"""
import numpy as np, pandas as pd, torch, os, glob
from scipy import stats

RT = 0.001987 * 298.15
TENS = "/groups/keasar_group/casp15/meytav/protein_tensors/%s/coords_tensor.pt"

def folded_ca(p):
    f = TENS % p
    if not os.path.exists(f): return None
    x = torch.load(f, map_location="cpu", weights_only=False).float()
    return x[:, 1, :]                                    # CA

def unfolded_map(N):
    s = torch.arange(N, dtype=torch.float32)
    sep = (s.unsqueeze(0) - s.unsqueeze(1)).abs()
    return torch.sqrt(6 * 1.927 * sep.clamp(min=1e-6) ** 0.598)   # Eq 2

def summarize(D, nb=16):
    """Compact, N-invariant summary: mean distance per |i-j| decile + a distance histogram."""
    N = D.shape[0]
    s = torch.arange(N)
    sep = (s.unsqueeze(0) - s.unsqueeze(1)).abs()
    out = []
    edges = np.linspace(1, N - 1, 9).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (sep >= a) & (sep < max(b, a + 1))
        out.append(D[m].mean().item() if m.any() else 0.0)
    hist = torch.histc(D[sep > 0], bins=nb, min=0.0, max=40.0)
    hist = hist / hist.sum().clamp(min=1)
    return np.concatenate([np.array(out), hist.numpy()])

d = pd.read_csv("eval_results/abl_calib_ctrl_repro2_e14.csv")
d = d[d.protein != "2K5H"]
prots = sorted(d.protein.unique())

rows = []
for p in prots:
    ca = folded_ca(p)
    if ca is None: continue
    N = ca.shape[0]
    Df = torch.cdist(ca, ca)
    Du = unfolded_map(N)
    g = d[d.protein == p]
    # subsample variants for tractability; coordinates are shared, only dG differs
    idx = np.linspace(0, len(g) - 1, min(60, len(g))).astype(int)
    for i in idx:
        dg = float(g.deltaG.values[i])
        pF = 1.0 / (1.0 + np.exp(-dg / RT))
        Dmix = pF * Df + (1 - pF) * Du
        rows.append(dict(protein=p, dG=dg, pF=pF,
                         **{("f%d" % k): v for k, v in enumerate(summarize(Df))},
                         **{("m%d" % k): v for k, v in enumerate(summarize(Dmix))}))
R = pd.DataFrame(rows)
print("proteins: %d   variant rows: %d" % (R.protein.nunique(), len(R)))
fcols = [c for c in R.columns if c.startswith("f")]
mcols = [c for c in R.columns if c.startswith("m")]

def loprobe(cols, label):
    """Leave-one-protein-out ridge. T-B: the honest test."""
    from sklearn.linear_model import RidgeCV
    pred = np.zeros(len(R)); 
    for p in R.protein.unique():
        te = R.protein == p; tr = ~te
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(R.loc[tr, cols], R.loc[tr, "dG"])
        pred[te.values] = mdl.predict(R.loc[te, cols])
    r = np.corrcoef(R.dG, pred)[0, 1]
    rmse = np.sqrt(((R.dG - pred) ** 2).mean())
    base = np.sqrt(((R.dG - R.dG.mean()) ** 2).mean())
    print("  %-28s LOPO r = %+.4f   RMSE %.4f   (predict-the-mean %.4f)" % (label, r, rmse, base))
    return r, rmse

print("\n=== T-A + T-B: can dG be recovered, leave-one-protein-out? ===")
rf, _ = loprobe(fcols, "D_folded only")
rm, _ = loprobe(mcols, "D_mix (Eq 5)")
print("  -> mixture adds %+.4f in r over the folded map alone" % (rm - rf))

print("\n=== T-C: does D_mix separate variants WITHIN a protein? ===")
ws = []
for p in R.protein.unique():
    s = R[R.protein == p]
    if len(s) < 10 or s.dG.std() < 1e-6: continue
    v = s[mcols].values
    keep = v.std(0) > 1e-12
    if keep.sum() == 0: continue
    pc = (v[:, keep] - v[:, keep].mean(0)) / v[:, keep].std(0)
    u, _, _ = np.linalg.svd(pc, full_matrices=False)
    ws.append(abs(np.corrcoef(u[:, 0], s.dG)[0, 1]))
print("  |corr(top PC of D_mix, dG)| within protein: mean %.4f  median %.4f  (n=%d)"
      % (np.mean(ws), np.median(ws), len(ws)))
fs = []
for p in R.protein.unique():
    s = R[R.protein == p]
    v = s[fcols].values
    fs.append(v.std(0).max())
print("  max within-protein sd of D_folded features: %.3e  (0 = constant, as expected)" % np.mean(fs))

print("\n=== T-D: how does this compare to the model we already have? ===")
mp = []
for p in R.protein.unique():
    g = d[d.protein == p]
    mp.append(np.corrcoef(g.deltaG, g.pred_deltaG)[0, 1])
print("  our model's within-protein dG r: mean %.4f  median %.4f" % (np.mean(mp), np.median(mp)))
