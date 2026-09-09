import csv, math, statistics as st
from collections import OrderedDict
SHIFT=3.0824
def mean(x): return sum(x)/len(x)
def std(x):
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def pcc(a,b):
    if len(a)<3: return float('nan')
    ma,mb=mean(a),mean(b)
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    if da==0 or db==0: return float('nan')
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(da*db)

def load(path, correct2k5h=True):
    P=OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            p=r['protein']
            dg=float(r['deltaG']); pdg=float(r['pred_deltaG'])
            dd=float(r['ddG']); pdd=float(r['pred_ddG'])
            if correct2k5h and p=='2K5H': dd-=SHIFT
            P.setdefault(p,[]).append((dg,pdg,dd,pdd))
    return P

def analyse(path, drop2k5h_for_ddg=True, correct=True):
    P=load(path, correct)
    # --- dG side: WT row per protein (ddG==0 row) ---
    wt_true=[]; wt_pred=[]; provs=[]
    for p,rows in P.items():
        w=[r for r in rows if r[2]==0.0] if not correct or p!='2K5H' else [r for r in rows if abs(r[2]+SHIFT)<1e-9]
        if not w: w=[min(rows,key=lambda r:abs(r[2]))]
        wt_true.append(w[0][0]); wt_pred.append(w[0][1]); provs.append(p)
    if correct:
        i=provs.index('2K5H') if '2K5H' in provs else None
        if i is not None: wt_true[i]+=SHIFT
    bp=[a-b for a,b in zip(wt_pred,wt_true)]
    dgmae=mean([abs(v) for v in bp])
    # --- ddG side (per-protein), 27 proteins ---
    aps=[]; rs=[]; ss=[]; ppcc=[]
    pool_t=[]; pool_p=[]
    for p,rows in P.items():
        if drop2k5h_for_ddg and p=='2K5H': continue
        t=[r[2] for r in rows]; q=[r[3] for r in rows]
        if len(t)<3: continue
        r_=pcc(t,q); st_=std(t); sq_=std(q)
        s_=sq_/st_ if st_>0 else float('nan')
        aps.append(r_*s_); rs.append(r_); ss.append(s_); ppcc.append(r_)
        pool_t+=t; pool_p+=q
    med=lambda x: st.median(x)
    return dict(n_prot=len(P), dgMAE=dgmae, mean_bp=mean(bp), std_bp=std(bp),
                std_pred_wt=std(wt_pred), std_true_wt=std(wt_true),
                corr_bp_true=pcc(bp,wt_true), ap=med(aps), r=med(rs), s=med(ss),
                ppcc=med(ppcc), pooled=pcc(pool_t,pool_p), n_ddg=len(aps))

runs=[('dG arm e0','eval_results/abl_gld_dg_coil_s42_e0.csv'),
      ('dG arm e4','eval_results/abl_gld_dg_coil_s42_e4.csv'),
      ('dG arm e8','eval_results/abl_gld_dg_coil_s42_e8.csv'),
      ('dG arm e12','eval_results/abl_gld_dg_coil_s42_e12.csv'),
      ('dG arm e13','eval_results/abl_gld_dg_coil_s42_e13.csv'),
      ('dG arm e14','eval_results/abl_gld_dg_coil_s42_e14.csv'),
      ('control e14','eval_results/abl_calib_ctrl_repro2_e14.csv')]
print("2K5H CORRECTED (+3.0824); ddG stats on 27 proteins (2K5H dropped); dG stats on 28")
hdr=f"{'run':<14}{'dGMAE':>8}{'std_bp':>8}{'stdPredWT':>10}{'corr(bp,t)':>11}{'ppPCC':>8}{'a_p':>8}{'r':>7}{'s':>7}{'pooled':>8}"
print(hdr); print('-'*len(hdr))
for name,path in runs:
    try:
        m=analyse(path)
        print(f"{name:<14}{m['dgMAE']:>8.4f}{m['std_bp']:>8.4f}{m['std_pred_wt']:>10.4f}{m['corr_bp_true']:>11.4f}"
              f"{m['ppcc']:>8.4f}{m['ap']:>8.4f}{m['r']:>7.4f}{m['s']:>7.4f}{m['pooled']:>8.4f}")
    except Exception as e:
        print(f"{name:<14} ERROR {e}")
m=analyse(runs[0][1]); print('\nstd(true WT dG) [degenerate attractor, corrected] =', round(m['std_true_wt'],4))
