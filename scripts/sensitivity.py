# -*- coding: utf-8 -*-
"""Sensitivity analyses for the b_p corrector.

(A) pLDDT subset: plddt.csv covers only 24 of the 28 test proteins (the 4 de-novo
    designed ones have no pLDDT, and their AlphaFold PDB B-factors are all 0.00, so
    imputing would fabricate data). Re-run LOPO on those 24 WITH and WITHOUT the two
    pLDDT columns to see whether pLDDT carries offset signal.

(B) Small feature sets: n=28 with 22 features is a severe overfitting regime. Re-run
    LOPO restricted to the few features with the strongest prior support, including
    mean_rel_SASA (the a_p correlate) and length.

(C) Univariate LOPO: single-feature ridge, every feature, to find whether ANY single
    covariate predicts b_p out of sample.
"""
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else '/home/nissimb/DeepPEF'
spec = importlib.util.spec_from_file_location(
    'oc', os.path.join(REPO, 'scripts', 'offset_corrector.py'))
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)
cat = oc._load_catalogue_module(REPO)

eval_dir = os.path.join(REPO, 'eval_results')
pdb_dir = os.path.join(REPO, 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs')
plddt_csv = os.path.join(REPO, 'data/Processed_K50_dG_datasets/plddt.csv')
PRIMARY = 'abl_calib_ctrl_repro2_e14.csv'
ALPHAS = [30.0]  # pinned: matches permutation null, keeps runtime tractable

names = list(cat.recover_bp(os.path.join(eval_dir, PRIMARY))['protein'])
f1 = cat.structural_features(names, pdb_dir, plddt_csv)
f2 = oc.extra_structural_features(names, pdb_dir)
feat = f1.merge(f2, on='protein', how='left')
cand = [c for c in feat.columns
        if c != 'protein' and pd.api.types.is_numeric_dtype(feat[c])]
base_dropped = [c for c in cand if feat[c].isna().any() or feat[c].nunique() < 3]
BASE = [c for c in cand if c not in base_dropped]
csvs = sorted(f for f in os.listdir(eval_dir) if f.endswith('.csv'))
res = {}


def lopo_eval(cols, subset=None):
    """Return per-CSV (raw, ridge_pooled, r2) using feature set `cols`, optionally
    restricted to a subset of protein names."""
    rows = []
    for f in csvs:
        path = os.path.join(eval_dir, f)
        df = pd.read_csv(path)
        bp = cat.recover_bp(path)
        m = bp.merge(feat, on='protein', how='left')
        if subset is not None:
            m = m[m['protein'].isin(subset)]
            df = df[df['protein'].isin(subset)]
        m = m.sort_values('protein').reset_index(drop=True)
        y = m['b_p'].values.astype(float)
        X = m[cols].values.astype(float)
        nm = list(m['protein'])
        raw, _ = oc.pooled_pcc(df)
        oracle, _ = oc.pooled_pcc(df, {n: y[i] for i, n in enumerate(nm)})
        pred, _ = oc.lopo_ridge(X, y, ALPHAS)
        rid, _ = oc.pooled_pcc(df, {n: pred[i] for i, n in enumerate(nm)})
        rows.append(dict(csv=f, raw=raw, ridge=rid, oracle=oracle,
                         r2=oc.r2(y, pred), n=len(nm)))
    return rows


# ---- (A) pLDDT subset -------------------------------------------------------
pl = pd.read_csv(plddt_csv)
have = set(pl['protein'].unique())
subset24 = [n for n in names if n in have]
print('pLDDT subset: %d proteins (missing %d)' % (len(subset24), 28 - len(subset24)))
with_pl = BASE + ['plddt_mean', 'plddt_min']
a_wo = lopo_eval(BASE, subset24)
a_w = lopo_eval(with_pl, subset24)
print('\n(A) 24-protein subset, WITHOUT vs WITH pLDDT')
for r1, r2_ in zip(a_wo, a_w):
    print('  %-38s raw %.4f | noPL %.4f (R2 %+.3f) | PL %.4f (R2 %+.3f) | oracle %.4f'
          % (r1['csv'], r1['raw'], r1['ridge'], r1['r2'],
             r2_['ridge'], r2_['r2'], r1['oracle']))
res['plddt_subset'] = dict(n_proteins=len(subset24), without=a_wo, with_plddt=a_w,
                           missing=[n for n in names if n not in have])

# ---- (B) small feature sets -------------------------------------------------
SETS = {
    'sasa_len': ['mean_rel_SASA', 'length'],
    'sasa_only': ['mean_rel_SASA'],
    'compact4': ['mean_rel_SASA', 'length', 'rel_contact_order', 'frac_hydrophobic'],
    'ss3': ['frac_helix', 'frac_strand', 'frac_coil'],
}
res['small_sets'] = {}
print('\n(B) small feature sets (28 proteins)')
for k, cols in SETS.items():
    rows = lopo_eval(cols)
    res['small_sets'][k] = dict(features=cols, rows=rows)
    print('  %-11s mean raw %.4f | mean ridge %.4f | mean R2 %+.3f | beats raw %d/10'
          % (k, np.mean([r['raw'] for r in rows]), np.mean([r['ridge'] for r in rows]),
             np.mean([r['r2'] for r in rows]),
             sum(1 for r in rows if r['ridge'] > r['raw'])))

# ---- (C) univariate ---------------------------------------------------------
print('\n(C) univariate LOPO R2 on primary CSV (%s)' % PRIMARY)
path = os.path.join(eval_dir, PRIMARY)
df = pd.read_csv(path)
bp = cat.recover_bp(path)
m = bp.merge(feat, on='protein', how='left').sort_values('protein').reset_index(drop=True)
y = m['b_p'].values.astype(float)
uni = []
for c in BASE:
    X = m[[c]].values.astype(float)
    pred, _ = oc.lopo_ridge(X, y, ALPHAS)
    rid, _ = oc.pooled_pcc(df, {n: pred[i] for i, n in enumerate(m['protein'])})
    uni.append(dict(feature=c, r2=oc.r2(y, pred), pooled=rid))
uni.sort(key=lambda d: -d['r2'])
for u in uni:
    print('  %-28s R2 %+.4f  pooled %.4f' % (u['feature'], u['r2'], u['pooled']))
res['univariate_primary'] = uni

with open(os.path.join(REPO, 'results/offset_corrector_sensitivity.json'), 'w') as fh:
    json.dump(res, fh, indent=2, default=float)
print('\n[wrote] results/offset_corrector_sensitivity.json')
