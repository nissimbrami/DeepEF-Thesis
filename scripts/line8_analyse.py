"""LINE 8 analysis: what does the trained network actually USE?

Reads the block-ablation CSVs and reports, for every condition:
    dG MAE           |a*pred + b - true|              (affine sidecar applied, as in
                                                       the trusted evaluate.py path)
    std(b_p)         SD across proteins of the WT dG error   <- the offset spread
    pooled ddG PCC   over all rows, ddG anchored on the WT row per protein
    PP ddG PCC       median of the per-protein ddG PCC
    median a_p       median per-protein ddG slope (pred_ddG ~ a*true_ddG)

METRIC RULE. ddG cancels anything identical between WT and mutant. A block ablated in
the UNFOLDED pass only is identical across mutants of one protein ONLY in so far as the
unfolded graph does not depend on the mutation -- but it DOES (one_hot and emb are
per-mutant). So no ablation here is a pure no-op on ddG by construction; the cancellation
argument applies to the geometry blocks D and Fb, which are mutation-independent.
That prediction is checked explicitly below and is a correctness test of the pipeline.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else '.'
AFF = json.load(open(os.path.join(
    REPO, 'Megascale-fineTuning/models/'
    'PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/affine.json')))
A_, B_ = float(AFF['a']), float(AFF['b'])

fs = sorted(glob.glob(os.path.join(REPO, 'results/line8/blocks_s*.csv')))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print('shards=%d rows=%d proteins=%d' % (len(fs), len(df), df['protein'].nunique()))
conds = [c[5:] for c in df.columns if c.startswith('pred_')]
print('conditions: %s' % conds)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    sxx = ((x - x.mean()) ** 2).sum()
    if sxx <= 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).sum() / sxx)


rows = []
per_prot = {}
for c in conds:
    p = A_ * df['pred_' + c].values + B_          # same affine as the trusted eval path
    t = df['deltaG'].values
    err = p - t
    mae = float(np.abs(err).mean())

    bp, ap, ppp = [], [], []
    ddg_all, pddg_all = [], []
    for name, g in df.groupby('protein', sort=True):
        pg = A_ * g['pred_' + c].values + B_
        tg = g['deltaG'].values
        bp.append(float(pg[0] - tg[0]))            # WT row is row 0 by construction
        ddg = tg - tg[0]
        pddg = pg - pg[0]
        ddg_all.append(ddg); pddg_all.append(pddg)
        ap.append(slope(ddg, pddg))
        ppp.append(pearson(ddg, pddg))
    ddg_all = np.concatenate(ddg_all); pddg_all = np.concatenate(pddg_all)
    per_prot[c] = dict(b_p=bp, a_p=ap, pcc=ppp)
    rows.append(dict(cond=c, dG_MAE=mae, std_bp=float(np.std(bp)),
                     pooled_ddG_PCC=pearson(ddg_all, pddg_all),
                     PP_ddG_PCC=float(np.nanmedian(ppp)),
                     median_a_p=float(np.nanmedian(ap)),
                     dG_PCC=pearson(t, p)))

R = pd.DataFrame(rows).set_index('cond')
order = ['base'] + [c for c in ['D_f','D_u','D_b','Fb_f','Fb_u','Fb_b','emb_f','emb_u',
                                'emb_b','oh_f','oh_u','oh_b','geom_f','geom_b','seq_b']
                    if c in R.index]
R = R.loc[order]
pd.set_option('display.width', 200)
print()
print(R.round(4).to_string())

b = R.loc['base']
print()
print('--- delta vs base (dG MAE / std_bp: negative = BETTER; PCC: negative = WORSE) ---')
for c in R.index:
    if c == 'base':
        continue
    r = R.loc[c]
    print('  %-8s dMAE=%+8.4f  dstd(b_p)=%+8.4f  dPooledPCC=%+8.4f  dPP_PCC=%+8.4f  d_a_p=%+8.4f'
          % (c, r.dG_MAE - b.dG_MAE, r.std_bp - b.std_bp,
             r.pooled_ddG_PCC - b.pooled_ddG_PCC, r.PP_ddG_PCC - b.PP_ddG_PCC,
             r.median_a_p - b.median_a_p))

# --- energy decomposition: cov(E_u, E_f) --------------------------------------
print()
print('=== ENERGY DECOMPOSITION (base condition, per protein, WT row) ===')
wt = df.groupby('protein', sort=True).first().reset_index()
Ef, Eu = wt['E_f'].values, wt['E_u'].values
dg_pred = Eu - Ef
print('n proteins = %d' % len(wt))
print('var(E_u) = %.4f   var(E_f) = %.4f   cov = %.4f' % (Eu.var(), Ef.var(), np.cov(Eu, Ef)[0,1]))
print('corr(E_u, E_f) = %.4f' % pearson(Eu, Ef))
print('var(E_u - E_f) = %.4f' % dg_pred.var())
print('  identity check: var(Eu)+var(Ef)-2cov = %.4f' % (Eu.var()+Ef.var()-2*np.cov(Eu,Ef)[0,1]))
print('share of var(dG_pred) attributed to E_u alone  = %.1f%%' % (100*Eu.var()/dg_pred.var()))
print('share of var(dG_pred) attributed to E_f alone  = %.1f%%' % (100*Ef.var()/dg_pred.var()))
print('share removed by the shared term -2cov(Eu,Ef)  = %.1f%%'
      % (-200*np.cov(Eu,Ef)[0,1]/dg_pred.var()))
wt_err = (A_*dg_pred + B_) - wt['deltaG'].values
print('corr(E_u, wt_err) = %.4f   corr(E_f, wt_err) = %.4f'
      % (pearson(Eu, wt_err), pearson(Ef, wt_err)))
# variance decomposition of wt_err onto the two channels
print('corr(E_u - mean, E_f - mean) again for clarity = %.4f' % pearson(Eu, Ef))

# also over ALL rows, not just WT
print()
print('--- over all %d rows (within+between protein) ---' % len(df))
print('corr(E_u, E_f) pooled = %.4f' % pearson(df['E_u'], df['E_f']))
wp = []
for name, g in df.groupby('protein'):
    wp.append(pearson(g['E_u'], g['E_f']))
print('corr(E_u, E_f) WITHIN protein: median %.4f  min %.4f  max %.4f'
      % (np.nanmedian(wp), np.nanmin(wp), np.nanmax(wp)))

out = dict(n_proteins=int(df['protein'].nunique()), n_rows=int(len(df)),
           affine=dict(a=A_, b=B_),
           table=R.reset_index().to_dict('records'),
           energy=dict(var_Eu=float(Eu.var()), var_Ef=float(Ef.var()),
                       cov=float(np.cov(Eu, Ef)[0,1]), corr=pearson(Eu, Ef),
                       var_dgpred=float(dg_pred.var()),
                       corr_Eu_wterr=pearson(Eu, wt_err),
                       corr_Ef_wterr=pearson(Ef, wt_err),
                       corr_within_median=float(np.nanmedian(wp))),
           per_protein={c: {k: list(map(float, v)) for k, v in d.items()}
                        for c, d in per_prot.items()},
           proteins=sorted(df['protein'].unique().tolist()))
op = os.path.join(REPO, 'results/line8/summary.json')
json.dump(out, open(op, 'w'), indent=1)
print()
print('wrote %s' % op)
