# -*- coding: utf-8 -*-
"""Permutation null for the LOPO offset corrector.

If the ridge has real feature-driven skill, the pooled-PCC gain it achieves must be
larger than the gain obtained when the b_p labels are SHUFFLED relative to the feature
rows. Shuffling destroys any true feature->b_p mapping while preserving the marginal
distribution of b_p and the whole LOPO machinery -- so it isolates exactly the quantity
in question.

Also reports the "corrector-vs-raw" delta, which is the honest headline: subtracting a
CONSTANT leaves Pearson unchanged (shift invariance), so the raw number, not the mean
baseline, is the null that a per-protein corrector must beat on pooled PCC.
"""
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else '/home/nissimb/DeepPEF'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200

spec = importlib.util.spec_from_file_location(
    'oc', os.path.join(REPO, 'scripts', 'offset_corrector.py'))
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)
cat = oc._load_catalogue_module(REPO)

eval_dir = os.path.join(REPO, 'eval_results')
pdb_dir = os.path.join(REPO, 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs')
plddt = os.path.join(REPO, 'data/Processed_K50_dG_datasets/plddt.csv')
PRIMARY = 'abl_calib_ctrl_repro2_e14.csv'

names = list(cat.recover_bp(os.path.join(eval_dir, PRIMARY))['protein'])
f1 = cat.structural_features(names, pdb_dir, plddt)
f2 = oc.extra_structural_features(names, pdb_dir)
feat = f1.merge(f2, on='protein', how='left')
cand = [c for c in feat.columns
        if c != 'protein' and pd.api.types.is_numeric_dtype(feat[c])]
dropped = [c for c in cand if feat[c].isna().any() or feat[c].nunique() < 3]
cols = [c for c in cand if c not in dropped]

ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
csvs = sorted(f for f in os.listdir(eval_dir) if f.endswith('.csv'))
out = {}
for f in csvs:
    path = os.path.join(eval_dir, f)
    df = pd.read_csv(path)
    bp = cat.recover_bp(path)
    m = bp.merge(feat, on='protein', how='left').sort_values('protein').reset_index(drop=True)
    y = m['b_p'].values.astype(float)
    X = m[cols].values.astype(float)
    nm_list = list(m['protein'])
    raw, _ = oc.pooled_pcc(df)
    pred, _ = oc.lopo_ridge(X, y, [30.0])  # matched to permutation null
    real, _ = oc.pooled_pcc(df, {n: pred[i] for i, n in enumerate(nm_list)})
    d_real = real - raw
    rng = np.random.RandomState(0)
    gains, r2s = [], []
    for _ in range(NPERM):
        yp = rng.permutation(y)
        pp, _ = oc.lopo_ridge(X, yp, [30.0])
        # score the permuted-label model against the TRUE offsets it must correct:
        # subtract its predictions and recompute pooled PCC
        g, _ = oc.pooled_pcc(df, {n: pp[i] for i, n in enumerate(nm_list)})
        gains.append(g - raw)
        r2s.append(oc.r2(yp, pp))
    gains = np.array(gains)
    p = float((np.sum(gains >= d_real) + 1) / (len(gains) + 1))
    out[f] = dict(raw=raw, ridge=real, delta_vs_raw=d_real,
                  perm_delta_mean=float(gains.mean()),
                  perm_delta_p95=float(np.percentile(gains, 95)),
                  perm_delta_max=float(gains.max()),
                  p_value=p, r2_real=oc.r2(y, pred),
                  perm_r2_mean=float(np.mean(r2s)), n_perm=NPERM)
    print('%-38s raw %.4f ridge %.4f d %+.4f | perm mean %+.4f p95 %+.4f | p=%.3f'
          % (f, raw, real, d_real, gains.mean(), np.percentile(gains, 95), p))

with open(os.path.join(REPO, 'results/offset_corrector_perm.json'), 'w') as fh:
    json.dump(out, fh, indent=2, default=float)
ps = [v['p_value'] for v in out.values()]
print('\nn CSVs with p<0.05: %d/%d' % (sum(1 for x in ps if x < 0.05), len(ps)))
print('[wrote] results/offset_corrector_perm.json')
