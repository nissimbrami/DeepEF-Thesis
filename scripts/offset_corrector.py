# -*- coding: utf-8 -*-
"""
offset_corrector.py -- turn the per-protein offset ORACLE into a METHOD.

The 0.70-0.72 pooled-PCC ceiling is obtained by subtracting b_p, the per-protein ddG
intercept FITTED ON THE TEST LABELS of that same protein. That is an oracle. This script
asks whether b_p can instead be PREDICTED from structure-only features on HELD-OUT
proteins, and whether subtracting the PREDICTION still lifts pooled PCC.

Protocol (leave-one-protein-out, n=28):
  for each held-out protein p:
      fit ridge on the OTHER 27 (standardiser + alpha chosen by INNER LOPO on those 27)
      predict b_p_hat(p)  -- p's own label never enters any fit, at any level
      subtract b_p_hat(p) from that protein's ddG_pred
  recompute pooled PCC over all pairs

Reference points reported side by side:
  raw            : no offset removed
  mean-baseline  : subtract the LOPO mean of the other 27 b_p (uses NO features)
  LOPO-ridge     : subtract the feature-model prediction
  oracle         : subtract the protein's own fitted b_p  (upper bound, not a method)

The mean-baseline is the honest null: it is what you get for free from knowing that b_p
has a non-zero mean. Any claim that features help must beat THAT, not the raw number.

Features come from scripts/catalogue_vs_bp.structural_features -- imported, not
duplicated -- plus radius of gyration, contact order and CA-geometry secondary-structure
fractions added here.
"""
import argparse
import importlib.util
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))


def _load_catalogue_module(repo):
    """Import scripts/catalogue_vs_bp.py for REUSE of recover_bp/structural_features."""
    path = os.path.join(repo, 'scripts', 'catalogue_vs_bp.py')
    if not os.path.exists(path):
        raise SystemExit('cannot find %s' % path)
    spec = importlib.util.spec_from_file_location('catalogue_vs_bp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ('recover_bp', 'structural_features'):
        if not hasattr(mod, fn):
            raise SystemExit('catalogue_vs_bp.py lacks %s' % fn)
    return mod


# ---------------------------------------------------------------- extra features
def extra_structural_features(names, pdb_dir):
    """Radius of gyration, contact order, and secondary-structure fractions.

    Secondary structure without DSSP: assign per-residue from CA-trace geometry.
    An i,i+3 CA distance in 4.5-6.5 A marks a helical turn; an i,i+2 CA distance
    above 6.4 A marks locally extended (strand-like) backbone. Crude but
    deterministic; it only has to serve as a covariate.
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    out = []
    for nm in names:
        p = os.path.join(pdb_dir, nm + '.pdb')
        rec = dict(protein=nm)
        if not os.path.exists(p):
            out.append(rec)
            continue
        model = parser.get_structure(nm, p)[0]
        res = [r for r in model.get_residues() if r.get_id()[0] == ' ' and 'CA' in r]
        ca = np.array([r['CA'].get_coord() for r in res], dtype=float)
        n = len(ca)
        if n < 6:
            out.append(rec)
            continue
        cen = ca.mean(axis=0)
        rec['radius_gyration'] = float(np.sqrt(((ca - cen) ** 2).sum(axis=1).mean()))
        rec['Rg_over_len_pow_038'] = float(rec['radius_gyration'] / (n ** 0.38))
        D = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        iu = np.triu_indices(n, k=3)
        close = D[iu] < 8.0
        seps = (iu[1] - iu[0])[close]
        rec['n_contacts_CA8'] = float(close.sum())
        rec['contact_density'] = float(close.sum()) / n
        rec['rel_contact_order'] = float(seps.mean()) / n if close.sum() else np.nan
        rec['abs_contact_order'] = float(seps.mean()) if close.sum() else np.nan
        d3 = np.linalg.norm(ca[:-3] - ca[3:], axis=1)
        d2 = np.linalg.norm(ca[:-2] - ca[2:], axis=1)
        helix = np.zeros(n, dtype=bool)
        for i in np.where((d3 > 4.5) & (d3 < 6.5))[0]:
            helix[i:i + 4] = True
        strand = np.zeros(n, dtype=bool)
        for i in np.where(d2 > 6.4)[0]:
            strand[i:i + 3] = True
        strand &= ~helix
        rec['frac_helix'] = float(helix.mean())
        rec['frac_strand'] = float(strand.mean())
        rec['frac_coil'] = float(1.0 - helix.mean() - strand.mean())
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- pooled metric
def pooled_pcc(df, offsets=None):
    """Pooled ddG PCC. offsets: dict protein -> value subtracted from that protein's
    ddG_pred. Row 0 of each group is WT, matching calib_diag exactly."""
    T, P = [], []
    for nm, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        dt = g['deltaG'].values - g['deltaG'].values[0]
        dp = g['pred_deltaG'].values - g['pred_deltaG'].values[0]
        if offsets is not None:
            dp = dp - offsets[nm]
        T.append(dt)
        P.append(dp)
    T = np.concatenate(T)
    P = np.concatenate(P)
    return float(stats.pearsonr(T, P)[0]), int(len(T))


# ---------------------------------------------------------------- LOPO ridge
def lopo_ridge(X, y, alphas):
    """Leave-one-protein-out ridge with INNER LOPO alpha selection on the training
    fold ONLY. Returns held-out predictions and the alpha chosen per fold."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    pred = np.zeros(n)
    chosen = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        Xtr, ytr = X[tr], y[tr]
        best_a, best_err = None, np.inf
        for a in alphas:
            errs = []
            for k in range(len(tr)):
                itr = np.array([j for j in range(len(tr)) if j != k])
                sc = StandardScaler().fit(Xtr[itr])
                m = Ridge(alpha=a).fit(sc.transform(Xtr[itr]), ytr[itr])
                errs.append((m.predict(sc.transform(Xtr[k:k + 1]))[0] - ytr[k]) ** 2)
            e = float(np.mean(errs))
            if e < best_err:
                best_err, best_a = e, a
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=best_a).fit(sc.transform(Xtr), ytr)
        pred[i] = m.predict(sc.transform(X[i:i + 1]))[0]
        chosen.append(float(best_a))
    return pred, chosen


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def run_one(csv_path, feat, feature_cols, cat_mod, alphas):
    df = pd.read_csv(csv_path)
    bp = cat_mod.recover_bp(csv_path)
    m = bp.merge(feat, on='protein', how='left').sort_values('protein').reset_index(drop=True)
    y = m['b_p'].values.astype(float)
    X = m[feature_cols].values.astype(float)
    if not np.isfinite(X).all():
        raise SystemExit('non-finite features in %s' % csv_path)
    names = list(m['protein'])
    n = len(names)

    raw, npairs = pooled_pcc(df)
    oracle, _ = pooled_pcc(df, {nm: y[i] for i, nm in enumerate(names)})
    mean_pred = np.array([y[[j for j in range(n) if j != i]].mean() for i in range(n)])
    meanb, _ = pooled_pcc(df, {nm: mean_pred[i] for i, nm in enumerate(names)})
    ridge_pred, chosen = lopo_ridge(X, y, alphas)
    ridgeb, _ = pooled_pcc(df, {nm: ridge_pred[i] for i, nm in enumerate(names)})

    return dict(
        eval_csv=os.path.basename(csv_path), n_proteins=n, n_pairs=npairs,
        pooled_raw=raw, pooled_mean_baseline=meanb, pooled_lopo_ridge=ridgeb,
        pooled_oracle=oracle,
        bp_std=float(y.std(ddof=1)), bp_mean=float(y.mean()),
        r2_lopo_ridge=r2(y, ridge_pred), r2_lopo_mean=r2(y, mean_pred),
        rmse_lopo_ridge=float(np.sqrt(((y - ridge_pred) ** 2).mean())),
        rmse_lopo_mean=float(np.sqrt(((y - mean_pred) ** 2).mean())),
        pearson_bp_vs_pred=float(stats.pearsonr(y, ridge_pred)[0]),
        alphas_chosen=chosen,
        per_protein=[dict(protein=names[i], b_p=float(y[i]),
                          b_p_hat_ridge=float(ridge_pred[i]),
                          b_p_hat_mean=float(mean_pred[i])) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--eval_dir', default=None)
    ap.add_argument('--primary', default='abl_calib_ctrl_repro2_e14.csv')
    ap.add_argument('--pdb_dir', default=None)
    ap.add_argument('--plddt', default=None)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()
    repo = args.repo
    eval_dir = args.eval_dir or os.path.join(repo, 'eval_results')
    pdb_dir = args.pdb_dir or os.path.join(
        repo, 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs')
    plddt = args.plddt or os.path.join(
        repo, 'data/Processed_K50_dG_datasets/plddt.csv')
    out_json = args.out_json or os.path.join(repo, 'results/offset_corrector.json')

    cat_mod = _load_catalogue_module(repo)
    csvs = sorted(f for f in os.listdir(eval_dir) if f.endswith('.csv'))
    primary_path = os.path.join(eval_dir, args.primary)

    names = list(cat_mod.recover_bp(primary_path)['protein'])
    f1 = cat_mod.structural_features(names, pdb_dir, plddt)
    f2 = extra_structural_features(names, pdb_dir)
    feat = f1.merge(f2, on='protein', how='left')

    cand = [c for c in feat.columns
            if c != 'protein' and pd.api.types.is_numeric_dtype(feat[c])]
    dropped = [c for c in cand if feat[c].isna().any() or feat[c].nunique() < 3]
    feature_cols = [c for c in cand if c not in dropped]
    print('features used (%d): %s' % (len(feature_cols), feature_cols))
    print('dropped constant/NaN (%d): %s' % (len(dropped), dropped))

    alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    results = []
    for f in csvs:
        r = run_one(os.path.join(eval_dir, f), feat, feature_cols, cat_mod, alphas)
        results.append(r)
        print('%-38s raw %.4f | mean %.4f | LOPO %.4f | oracle %.4f | R2 %+.3f'
              % (r['eval_csv'], r['pooled_raw'], r['pooled_mean_baseline'],
                 r['pooled_lopo_ridge'], r['pooled_oracle'], r['r2_lopo_ridge']))

    prim = [r for r in results if r['eval_csv'] == args.primary][0]
    agg = {k: dict(mean=float(np.mean([r[k] for r in results])),
                   std=float(np.std([r[k] for r in results], ddof=1)),
                   min=float(np.min([r[k] for r in results])),
                   max=float(np.max([r[k] for r in results])))
           for k in ('pooled_raw', 'pooled_mean_baseline', 'pooled_lopo_ridge',
                     'pooled_oracle', 'r2_lopo_ridge', 'r2_lopo_mean')}
    n_beat = int(sum(1 for r in results
                     if r['pooled_lopo_ridge'] > r['pooled_mean_baseline']))
    n_r2pos = int(sum(1 for r in results if r['r2_lopo_ridge'] > 0))

    out = dict(
        protocol=dict(
            design='leave-one-protein-out ridge; inner LOPO alpha selection on the 27 '
                   'training proteins only; held-out protein label never used in any fit',
            target='b_p = ddG-space per-protein intercept, '
                   'np.polyfit(ddg_true, ddg_pred, 1)[1]',
            n_proteins=28, n_features=len(feature_cols), features=feature_cols,
            dropped_features=dropped, alphas=alphas,
            baselines=dict(
                raw='no offset removed',
                mean_baseline='subtract LOPO mean of the other 27 b_p (uses NO features)',
                lopo_ridge='subtract ridge prediction from structure features',
                oracle='subtract the protein own fitted b_p -- ORACLE, not a method')),
        primary=prim, all_csvs=results, aggregate=agg,
        n_csvs_ridge_beats_mean=n_beat, n_csvs_total=len(results),
        n_csvs_ridge_r2_positive=n_r2pos)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print('\n[wrote] %s' % out_json)
    print('ridge beats mean-baseline on %d/%d CSVs; R2>0 on %d/%d'
          % (n_beat, len(results), n_r2pos, len(results)))


if __name__ == '__main__':
    main()
