"""Re-score --dg_length_norm on the metric it ACTS on (absolute dG / WT error / b_p),
not on the ddG offset it was refuted with (FINDING 3, RESEARCH_LOG L339)."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

def per_protein(csv):
    df=pd.read_csv(csv); rows=[]
    for p,g in df.groupby('protein'):
        if len(g)<3: continue
        wt=g.iloc[0]
        a,b=np.polyfit(g['ddG'].values,g['pred_ddG'].values,1)
        rows.append(dict(protein=p,
            n=len(g),
            wt_true=float(wt['deltaG']), wt_pred=float(wt['pred_deltaG']),
            wt_err=float(wt['pred_deltaG']-wt['deltaG']),
            mean_pred_dG=float(g['pred_deltaG'].mean()),
            mean_true_dG=float(g['deltaG'].mean()),
            a_p=float(a), b_p=float(b)))
    return pd.DataFrame(rows)

# protein length from the AlphaFold PDB (CA count)
PDB='data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
def calen(pid):
    f=os.path.join(PDB,pid+'.pdb')
    if not os.path.exists(f): return np.nan
    return sum(1 for l in open(f) if l.startswith('ATOM') and l[12:16].strip()=='CA')

for csv in sorted(glob.glob('eval_results/abl_*.csv')):
    d=per_protein(csv)
    d['L']=[calen(p) for p in d['protein']]
    d=d.dropna(subset=['L'])
    if len(d)<10: continue
    print("=== %s  (n=%d)"%(os.path.basename(csv),len(d)))
    for t in ['wt_err','wt_pred','mean_pred_dG','b_p','a_p']:
        r,p=pearsonr(d['L'],d[t]); rh,ps=spearmanr(d['L'],d[t])
        s=' <<<' if p<0.05 else ''
        print("   L vs %-14s r=%+.3f p=%.2e  rho=%+.3f p=%.2e%s"%(t,r,p,rh,ps,s))
    # the reference check: is TRUE dG length-dependent here?
    r,p=pearsonr(d['L'],d['wt_true'])
    print("   L vs %-14s r=%+.3f p=%.2e   (truth: is dG extensive on this set?)"%('wt_TRUE_dG',r,p))
    print()
