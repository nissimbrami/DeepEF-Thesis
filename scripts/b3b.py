import pandas as pd, numpy as np, re, glob, os, torch
from scipy import stats

MUT = "data/Processed_K50_dG_datasets/mutation_datasets"
KD = dict(zip("ACDEFGHIKLMNPQRSTVWY",
   [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))

def burial_from_coords(protein):
    """CB neighbour-count burial per residue, same definition as compute_burial."""
    import glob as g
    cands = g.glob("/groups/keasar_group/casp15/meytav/protein_tensors/%s/coords_tensor.pt" % protein)
    if not cands: return None
    x = torch.load(cands[0], map_location="cpu", weights_only=False).float()
    cb = x[:, 3, :] if x.shape[1] >= 4 else x[:, 1, :]
    d = torch.cdist(cb, cb)
    n = ((d < 10.0).float().sum(1) - 1)
    return (n / 20.0).clamp(0, 1).numpy()

def load_mut(protein):
    f = os.path.join(MUT, protein + ".csv")
    if not os.path.exists(f): return None
    d = pd.read_csv(f)
    rows = []
    for nm, dg in zip(d["name"].astype(str), d["deltaG"].values):
        m = re.search(r"\.pdb_([A-Z])(\d+)([A-Z])$", nm)
        if m:
            rows.append((float(dg), m.group(1), int(m.group(2)), m.group(3)))
    df = pd.DataFrame(rows, columns=["deltaG", "wt_aa", "pos", "mut_aa"])
    df["k"] = df.deltaG.round(5)
    return df

def per_mut_err(tag):
    d = pd.read_csv("eval_results/abl_%s.csv" % tag)
    d = d[d.protein != "2K5H"]
    return d

print("B3b - does W5 help specifically at BURIED positions?")
print("")
ctrl = per_mut_err("calib_ctrl_repro2_e14")
w5s = ["gld_w5_dg_s42_e14", "w5_dg_s1_e14", "w5_dg_s2_e14", "w5_dg_s3_e12"]
acc = []
nprot = 0
for p in sorted(ctrl.protein.unique()):
    bur = burial_from_coords(p)
    if bur is None: continue
    mut = load_mut(p)
    if mut is None or len(mut) == 0: continue
    c = ctrl[ctrl.protein == p][["deltaG", "ddG", "pred_ddG"]].copy()
    c["k"] = c.deltaG.round(5)
    j = c.merge(mut.drop(columns=["deltaG"]), on="k", how="inner")
    if len(j) < 30: continue
    j = j[(j.pos >= 1) & (j.pos <= len(bur))]
    if len(j) < 30: continue
    j["bur"] = bur[j.pos.values - 1]
    j["err_ctrl"] = (j.pred_ddG - j.ddG).abs()
    for t in w5s:
        w = per_mut_err(t); w = w[w.protein == p][["deltaG", "pred_ddG"]].copy()
        w["k"] = w.deltaG.round(5)
        jj = j.merge(w.drop(columns=["deltaG"]), on="k", how="inner", suffixes=("", "_w5"))
        if len(jj) < 30: continue
        jj["err_w5"] = (jj.pred_ddG_w5 - jj.ddG).abs()
        jj["gain"] = jj.err_ctrl - jj.err_w5      # positive = W5 better
        acc.append(jj[["bur", "gain", "wt_aa", "mut_aa"]])
    nprot += 1
if not acc:
    print("  no joinable data")
else:
    A = pd.concat(acc, ignore_index=True)
    print("  proteins joined: %d   mutations x seeds: %d" % (nprot, len(A)))
    print("")
    buried = A[A.bur >= 0.5]; exposed = A[A.bur < 0.25]
    print("  %-26s %8s %10s %10s" % ("", "n", "mean gain", "median"))
    print("  %-26s %8d %10.5f %10.5f" % ("BURIED  (burial>=0.50)", len(buried), buried.gain.mean(), buried.gain.median()))
    print("  %-26s %8d %10.5f %10.5f" % ("EXPOSED (burial<0.25)", len(exposed), exposed.gain.mean(), exposed.gain.median()))
    d = buried.gain.mean() - exposed.gain.mean()
    t = stats.ttest_ind(buried.gain, exposed.gain, equal_var=False)
    print("")
    print("  buried - exposed = %+.5f   Welch t=%.3f  p=%.4g" % (d, t.statistic, t.pvalue))
    print("  corr(burial, gain) = %+.4f  (n=%d)" % (np.corrcoef(A.bur, A.gain)[0,1], len(A)))
    print("")
    hyd = A[A.mut_aa.map(KD) > 1.5]
    pol = A[A.mut_aa.map(KD) < -1.5]
    print("  mutations TO hydrophobic : n=%6d  mean gain %+.5f" % (len(hyd), hyd.gain.mean()))
    print("  mutations TO polar       : n=%6d  mean gain %+.5f" % (len(pol), pol.gain.mean()))
    bh = A[(A.bur >= 0.5) & (A.mut_aa.map(KD) > 1.5)]
    print("  BURIED + hydrophobic     : n=%6d  mean gain %+.5f" % (len(bh), bh.gain.mean()))
