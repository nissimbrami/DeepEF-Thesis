"""Re-score contact order (declared DEAD on ddG) on the dG-side targets b_p / WT-error."""
import os, glob, math, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

PDBDIR='data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'

def ca_coords(path):
    xs=[]
    for line in open(path):
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            xs.append((float(line[30:38]),float(line[38:46]),float(line[46:54])))
    return np.array(xs)

def contact_order(X, cutoff=8.0):
    n=len(X)
    if n<3: return np.nan
    D=np.linalg.norm(X[:,None,:]-X[None,:,:],axis=-1)
    sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    m=(D<cutoff)&(sep>=3)
    if m.sum()==0: return np.nan
    return sep[m].mean()/n   # relative contact order

def lro(X, cutoff=8.0):
    """long-range order: contacts with |i-j|>12, normalised by n"""
    n=len(X)
    D=np.linalg.norm(X[:,None,:]-X[None,:,:],axis=-1)
    sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    m=(D<cutoff)&(sep>12)
    return m.sum()/2.0/n

# ---- per-protein calibration, calib_diag conventions ----
def calib(csv):
    df=pd.read_csv(csv)
    out={}
    for p,g in df.groupby('protein'):
        if len(g)<3: continue
        wt=g.iloc[0]
        b_wt_err = float(wt['pred_deltaG']-wt['deltaG'])
        a,b = np.polyfit(g['ddG'].values, g['pred_ddG'].values, 1)
        out[p]=dict(a_p=float(a), b_p=float(b), wt_err=b_wt_err,
                    pcc=float(pearsonr(g['ddG'],g['pred_ddG'])[0]))
    return out

def main():
    csv='eval_results/abl_calib_ctrl_repro2_e14.csv'
    cal=calib(csv)
    rows=[]
    for p,v in cal.items():
        pid=p.split('.')[0].split('_')[0]
        cands=glob.glob(os.path.join(PDBDIR,pid+'.pdb'))+glob.glob(os.path.join(PDBDIR,p+'.pdb'))
        if not cands:
            # try exact protein name
            c2=os.path.join(PDBDIR,p+'.pdb')
            if os.path.exists(c2): cands=[c2]
        if not cands: continue
        X=ca_coords(cands[0])
        if len(X)<10: continue
        rows.append(dict(protein=p, n=len(X), CO=contact_order(X), LRO=lro(X), **v))
    d=pd.DataFrame(rows).dropna()
    print("matched proteins:", len(d))
    if len(d)<10:
        print("PDB name mismatch - listing a few:", list(cal.keys())[:5])
        return
    d['abs_wt_err']=d['wt_err'].abs()
    d['abs_b_p']=d['b_p'].abs()
    print()
    print("%-8s %-14s %8s %10s %8s %10s"%("feat","target","r","p","rho","p"))
    for f in ['CO','LRO']:
        for t in ['wt_err','abs_wt_err','b_p','abs_b_p','a_p','pcc']:
            r,pp=pearsonr(d[f],d[t]); rh,ps=spearmanr(d[f],d[t])
            star=' ***' if pp<0.05 else ''
            print("%-8s %-14s %+8.3f %10.2e %+8.3f %10.2e%s"%(f,t,r,pp,rh,ps,star))
    d.to_csv('/tmp/co_bp.csv',index=False)

main()
