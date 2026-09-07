# -*- coding: utf-8 -*-
"""INDEPENDENT re-derivation of the b_p-side replication sweep.

Deliberately independent of scripts/bp_replication.py and scripts/catalogue_vs_bp.py:
  - targets built from the CSV's OWN ddG / pred_ddG columns (not recomputed from deltaG)
  - slope/intercept via scipy.stats.linregress (not np.polyfit)
  - SASA via Bio.PDB ShrakeRupley with a denser sphere, own rel-SASA normalisation
  - correlations recomputed from scratch
Reads NOTHING from results/*.json.
"""
import os, re, json, glob
import numpy as np, pandas as pd
from scipy import stats

PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
PLDDT   = 'data/Processed_K50_dG_datasets/plddt.csv'

THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
 'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
 'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
MAXASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
 'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
 'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
HYDRO = set('AVILMFWCY')
DESIGNED_RE = re.compile(r'(?:HHH|HEEH|EEHEE|EHEE|EHHE|HHHH|_TrROS_|v2_)', re.IGNORECASE)

FEATURES = ['length','mean_hydropathy_KD','frac_hydrophobic','frac_charged','frac_glycine',
 'frac_proline','total_SASA','SASA_per_residue','mean_rel_SASA','frac_buried_rel_lt_0.25',
 'frac_exposed_rel_gt_0.5','mean_rel_SASA_hydrophobic','SASA_over_len_pow_073',
 'plddt_mean','plddt_min']
BP_TARGETS = ['b_p','abs_b_p','b_p_wt_error','abs_b_p_wt_error']
ANCHOR = ['abl_anchor_w0.3_s42_e14','abl_anchor_w1.0_s42_e14','abl_anchor_w3.0_s42_e13']
SIGMA  = ['abl_sigma_seed1_e13','abl_sigma_seed2_e10','abl_sigma_seed3_e13',
          'abl_sigma_seed4_e14','abl_sigma_seed42_e9']


def calib(csv_path):
    """Per-protein calibration from the CSV's own ddG / pred_ddG columns."""
    df = pd.read_csv(csv_path)
    rows = []
    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        dt = g['ddG'].values.astype(float)
        dp = g['pred_ddG'].values.astype(float)
        if np.std(dt) < 1e-8:
            a, b, pcc = 1.0, float(np.mean(dp) - np.mean(dt)), np.nan
        else:
            lr = stats.linregress(dt, dp)
            a, b = float(lr.slope), float(lr.intercept)
            pcc = float(lr.rvalue) if np.std(dp) > 1e-8 else np.nan
        wt_err = float(g['pred_deltaG'].values[0] - g['deltaG'].values[0])
        rows.append(dict(protein=str(name), n=int(len(g)), a_p=a, b_p=b, abs_b_p=abs(b),
                         pcc=pcc, b_p_wt_error=wt_err, abs_b_p_wt_error=abs(wt_err),
                         designed=bool(DESIGNED_RE.search(str(name)))))
    return pd.DataFrame(rows)


def features(names):
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley(probe_radius=1.40, n_points=960)
    pl = pd.read_csv(PLDDT)
    pcol = 'plddt' if 'plddt' in pl.columns else pl.columns[-1]
    gmean = pl.groupby('protein')[pcol].mean(); gmin = pl.groupby('protein')[pcol].min()
    out = []
    for nm in names:
        rec = {'protein': nm}
        p = os.path.join(PDB_DIR, nm + '.pdb')
        if not os.path.exists(p):
            out.append(rec); continue
        model = parser.get_structure(nm, p)[0]
        res = [r for r in model.get_residues() if r.get_id()[0] == ' '
               and THREE2ONE.get(r.get_resname().strip())]
        seq = [THREE2ONE[r.get_resname().strip()] for r in res]
        L = len(seq); rec['length'] = L
        rec['mean_hydropathy_KD'] = float(np.mean([KD[s] for s in seq]))
        rec['frac_hydrophobic']   = float(np.mean([s in HYDRO for s in seq]))
        rec['frac_charged']       = float(np.mean([s in 'DEKR' for s in seq]))
        rec['frac_glycine']       = float(np.mean([s == 'G' for s in seq]))
        rec['frac_proline']       = float(np.mean([s == 'P' for s in seq]))
        sr.compute(model, level='R')
        sasa = np.array([float(r.sasa) for r in res])
        rel  = np.array([sasa[i] / MAXASA[seq[i]] for i in range(L)])
        relh = np.array([rel[i] for i in range(L) if seq[i] in HYDRO])
        rec['total_SASA'] = float(sasa.sum())
        rec['SASA_per_residue'] = float(sasa.mean())
        rec['mean_rel_SASA'] = float(rel.mean())
        rec['frac_buried_rel_lt_0.25'] = float((rel < 0.25).mean())
        rec['frac_exposed_rel_gt_0.5'] = float((rel > 0.5).mean())
        rec['mean_rel_SASA_hydrophobic'] = float(relh.mean()) if len(relh) else np.nan
        rec['SASA_over_len_pow_073'] = float(sasa.sum() / (L ** 0.73))
        rec['plddt_mean'] = float(gmean[nm]) if nm in gmean.index else np.nan
        rec['plddt_min']  = float(gmin[nm])  if nm in gmin.index  else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def verdict(sc, nsig):
    if sc < 10:
        return 'DEAD'
    return 'ESTABLISHED' if nsig >= 6 else 'SUGGESTIVE'


def main():
    csvs = sorted(glob.glob('eval_results/*.csv'))
    print('eval CSVs: %d' % len(csvs))
    feat = None; per = {}
    for c in csvs:
        cb = calib(c)
        if feat is None:
            feat = features(list(cb['protein']))
            print('features recomputed from %d PDBs (ShrakeRupley n_points=960)' % len(feat))
        per[os.path.basename(c)[:-4]] = cb.merge(feat, on='protein', how='left')
    sets = set(tuple(sorted(v['protein'])) for v in per.values())
    print('identical protein set across all CSVs: %s (n=%d)' % (len(sets) == 1, len(feat)))

    res = []
    for f in FEATURES:
        for t in BP_TARGETS + ['a_p', 'pcc']:
            rs, ps, keys = [], [], []
            for k, m in per.items():
                s = m[[f, t]].dropna()
                if len(s) < 8 or s[f].std() < 1e-12 or s[t].std() < 1e-12:
                    continue
                r, p = stats.pearsonr(s[f], s[t])
                rs.append(float(r)); ps.append(float(p)); keys.append(k)
            if len(rs) < 10:
                continue
            ar, ap = np.array(rs), np.array(ps)
            pos, neg = int((ar > 0).sum()), int((ar < 0).sum())
            sc = max(pos, neg); nsig = int((ap < 0.05).sum())
            res.append(dict(feature=f, target=t, mean_r=float(ar.mean()),
                sd_r=float(ar.std(ddof=1)), min_r=float(ar.min()), max_r=float(ar.max()),
                sign_consistency=sc, sign='+' if pos > neg else '-',
                n_sig=nsig, verdict=verdict(sc, nsig),
                per_csv_r=dict(zip(keys, rs)), per_csv_p=dict(zip(keys, ps))))

    json.dump(dict(n_csvs=len(csvs), n_proteins=int(len(feat)), results=res),
              open('results/bp_replication_independent.json', 'w'), indent=1)

    H = '%-28s %-18s %7s %6s %7s %17s %6s  %s'
    def row(r):
        return H % (r['feature'], r['target'], '%+.3f' % r['mean_r'], '%.3f' % r['sd_r'],
            '%d/10%s' % (r['sign_consistency'], r['sign']),
            '%+.3f..%+.3f' % (r['min_r'], r['max_r']), '%d/10' % r['n_sig'], r['verdict'])

    print('\n' + '=' * 118)
    print('B_P-SIDE REPLICATION SWEEP (15 features x 4 b_p targets = 60 pairs, 10 checkpoints)')
    print('=' * 118)
    bp = sorted([r for r in res if r['target'] in BP_TARGETS],
                key=lambda r: (-r['sign_consistency'], -r['n_sig'], -abs(r['mean_r'])))
    print(H % ('feature', 'target', 'mean_r', 'sd', 'signs', 'range', 'sig', 'class'))
    print('-' * 118)
    for r in bp:
        print(row(r))

    print('\n--- reference: a_p / pcc rows (top 8) ---')
    for r in sorted([r for r in res if r['target'] in ('a_p', 'pcc')],
                    key=lambda r: (-r['sign_consistency'], -r['n_sig'], -abs(r['mean_r'])))[:8]:
        print(row(r))

    print('\n=== VERIFY THE FOUR REPORTED NUMBERS ===')
    for f, t in [('frac_buried_rel_lt_0.25', 'b_p_wt_error'), ('SASA_per_residue', 'abs_b_p'),
                 ('length', 'abs_b_p'), ('mean_rel_SASA', 'a_p')]:
        for r in res:
            if r['feature'] == f and r['target'] == t:
                print('\n%s vs %s  -> %s' % (f, t, r['verdict']))
                print('  mean_r=%+.4f sd=%.4f  signs=%d/10%s  sig=%d/10  range=%+.3f..%+.3f'
                      % (r['mean_r'], r['sd_r'], r['sign_consistency'], r['sign'],
                         r['n_sig'], r['min_r'], r['max_r']))
                for k in sorted(r['per_csv_r']):
                    print('    %-30s r=%+.4f p=%.4g' % (k, r['per_csv_r'][k], r['per_csv_p'][k]))

    print('\n' + '=' * 110)
    print('ANCHOR-WEIGHT SUPPRESSION GRADIENT (3 anchor arms vs 5 unanchored sigma seeds)')
    print('=' * 110)
    grad = []
    for r in res:
        if r['target'] not in BP_TARGETS + ['a_p'] or r['sign_consistency'] < 10:
            continue
        a = [abs(r['per_csv_r'][k]) for k in ANCHOR if k in r['per_csv_r']]
        s = [abs(r['per_csv_r'][k]) for k in SIGMA if k in r['per_csv_r']]
        if len(a) < 3 or len(s) < 5:
            continue
        u, pu = stats.mannwhitneyu(s, a, alternative='greater')
        grad.append(dict(feature=r['feature'], target=r['target'], anchor=float(np.mean(a)),
            sigma=float(np.mean(s)), gap=float(np.mean(s) - np.mean(a)),
            sep=bool(min(s) > max(a)), p=float(pu),
            anchor_r=[r['per_csv_r'][k] for k in ANCHOR],
            sigma_r=[r['per_csv_r'][k] for k in SIGMA]))
    grad.sort(key=lambda g: -g['gap'])
    G = '%-28s %-18s %10s %10s %8s %7s %9s'
    print(G % ('feature', 'target', '|r|anchor', '|r|sigma', 'gap', 'separ', 'MWU p'))
    print('-' * 100)
    for g in grad:
        print(G % (g['feature'], g['target'], '%.3f' % g['anchor'], '%.3f' % g['sigma'],
                   '%+.3f' % g['gap'], 'YES' if g['sep'] else 'no', '%.4f' % g['p']))
    json.dump(grad, open('results/bp_anchor_gradient_independent.json', 'w'), indent=1)

    print('\nPer-arm detail:')
    for g in grad:
        if (g['feature'], g['target']) in [('frac_buried_rel_lt_0.25', 'b_p_wt_error'),
                ('mean_rel_SASA', 'a_p'), ('SASA_per_residue', 'abs_b_p')]:
            print('\n  %s vs %s' % (g['feature'], g['target']))
            print('    anchor w0.3,w1.0,w3.0 : ' + ', '.join('%+.3f' % v for v in g['anchor_r']))
            print('    sigma seeds 1,2,3,4,42: ' + ', '.join('%+.3f' % v for v in g['sigma_r']))

    ne = sum(1 for r in bp if r['verdict'] == 'ESTABLISHED')
    ns = sum(1 for r in bp if r['verdict'] == 'SUGGESTIVE')
    nd = sum(1 for r in bp if r['verdict'] == 'DEAD')
    print('\nB_P TALLY: %d ESTABLISHED, %d SUGGESTIVE, %d DEAD (of %d)' % (ne, ns, nd, len(bp)))


main()
