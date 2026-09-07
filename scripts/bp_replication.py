"""INDEPENDENT re-derivation of the b_p-side replication sweep over all 10 eval CSVs.

Recomputes per-protein calibration (a_p, b_p, pcc, b_p_wt_error) from the raw eval
CSVs and recomputes structural covariates directly from the AlphaFold PDBs with
Shrake-Rupley SASA. Does NOT read results/repl/*.json or results/catalogue_vs_bp.json.
"""
import os, re, json, glob
import numpy as np, pandas as pd
from scipy import stats

PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
PLDDT = 'data/Processed_K50_dG_datasets/plddt.csv'

THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
 'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
 'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
MAXASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
 'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
 'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
HYDROPHOBIC = set('AVILMFWCY')
DESIGNED_RE = re.compile(r'(?:HHH|HEEH|EEHEE|EHEE|EHHE|HHHH|_TrROS_|v2_)', re.IGNORECASE)


def per_protein_fit(dt, dp):
    if np.std(dt) < 1e-8:
        return 1.0, float(np.mean(dp) - np.mean(dt))
    a, b = np.polyfit(dt, dp, 1)
    return float(a), float(b)


def recover_bp(csv_path):
    df = pd.read_csv(csv_path)
    rows = []
    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        dt = g['deltaG'].values - g['deltaG'].values[0]
        dp = g['pred_deltaG'].values - g['pred_deltaG'].values[0]
        a, b = per_protein_fit(dt, dp)
        if np.std(dt) > 1e-8 and np.std(dp) > 1e-8:
            pcc = float(np.corrcoef(dt, dp)[0, 1])
        else:
            pcc = float('nan')
        wt_err = float(g['pred_deltaG'].values[0] - g['deltaG'].values[0])
        rows.append(dict(protein=str(name), n=len(g), a_p=a, b_p=b, abs_b_p=abs(b),
                         pcc=pcc, b_p_wt_error=wt_err, abs_b_p_wt_error=abs(wt_err),
                         designed=bool(DESIGNED_RE.search(str(name)))))
    return pd.DataFrame(rows)


def structural_features(names):
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    pl = pd.read_csv(PLDDT)
    pmean = pl.groupby('protein')['plddt'].mean()
    pmin = pl.groupby('protein')['plddt'].min()
    out = []
    for nm in names:
        rec = dict(protein=nm)
        p = os.path.join(PDB_DIR, nm + '.pdb')
        if not os.path.exists(p):
            out.append(rec)
            continue
        model = parser.get_structure(nm, p)[0]
        residues = [r for r in model.get_residues() if r.get_id()[0] == ' ']
        seq = [THREE2ONE.get(r.get_resname().strip()) for r in residues]
        seq = [s for s in seq if s]
        L = len(seq)
        rec['length'] = L
        rec['mean_hydropathy_KD'] = float(np.mean([KD[s] for s in seq]))
        rec['frac_hydrophobic'] = float(np.mean([s in HYDROPHOBIC for s in seq]))
        rec['frac_charged'] = float(np.mean([s in 'DEKR' for s in seq]))
        rec['frac_glycine'] = float(np.mean([s == 'G' for s in seq]))
        rec['frac_proline'] = float(np.mean([s == 'P' for s in seq]))
        sr.compute(model, level='R')
        sasa, rel, relhyd = [], [], []
        for r in residues:
            aa = THREE2ONE.get(r.get_resname().strip())
            if aa is None:
                continue
            s = float(r.sasa)
            sasa.append(s)
            rr = s / MAXASA[aa]
            rel.append(rr)
            if aa in HYDROPHOBIC:
                relhyd.append(rr)
        rel = np.array(rel)
        rec['total_SASA'] = float(np.sum(sasa))
        rec['SASA_per_residue'] = float(np.mean(sasa))
        rec['mean_rel_SASA'] = float(np.mean(rel))
        rec['frac_buried_rel_lt_0.25'] = float(np.mean(rel < 0.25))
        rec['frac_exposed_rel_gt_0.5'] = float(np.mean(rel > 0.5))
        rec['mean_rel_SASA_hydrophobic'] = float(np.mean(relhyd)) if relhyd else float('nan')
        rec['SASA_over_len_pow_073'] = float(np.sum(sasa) / (L ** 0.73))
        rec['plddt_mean'] = float(pmean[nm]) if nm in pmean.index else float('nan')
        rec['plddt_min'] = float(pmin[nm]) if nm in pmin.index else float('nan')
        out.append(rec)
    return pd.DataFrame(out)


FEATURES = ['length', 'mean_hydropathy_KD', 'frac_hydrophobic', 'frac_charged',
 'frac_glycine', 'frac_proline', 'total_SASA', 'SASA_per_residue', 'mean_rel_SASA',
 'frac_buried_rel_lt_0.25', 'frac_exposed_rel_gt_0.5', 'mean_rel_SASA_hydrophobic',
 'SASA_over_len_pow_073', 'plddt_mean', 'plddt_min']
BP_TARGETS = ['b_p', 'abs_b_p', 'b_p_wt_error', 'abs_b_p_wt_error']
ALL_TARGETS = BP_TARGETS + ['a_p', 'pcc']

# anchor-weight arms vs unanchored sigma seeds, for the suppression-gradient check
ANCHOR_ARMS = ['abl_anchor_w0.3_s42_e14', 'abl_anchor_w1.0_s42_e14', 'abl_anchor_w3.0_s42_e13']
SIGMA_ARMS = ['abl_sigma_seed1_e13', 'abl_sigma_seed2_e10', 'abl_sigma_seed3_e13',
              'abl_sigma_seed4_e14', 'abl_sigma_seed42_e9']


def classify(r):
    if r['sign_consistency'] < 10:
        return 'DEAD'
    return 'ESTABLISHED' if r['n_sig_p05'] >= 6 else 'SUGGESTIVE'


def main():
    csvs = sorted(glob.glob('eval_results/*.csv'))
    print('N eval CSVs: %d' % len(csvs))
    for c in csvs:
        print('   ' + c)

    feat = None
    per_csv = {}
    for c in csvs:
        bp = recover_bp(c)
        if feat is None:
            feat = structural_features(list(bp['protein']))
            print('\nFeature table recomputed from PDBs: %d proteins' % len(feat))
        per_csv[os.path.basename(c)[:-4]] = bp.merge(feat, on='protein', how='left')

    sets = set(tuple(sorted(v['protein'])) for v in per_csv.values())
    assert len(sets) == 1, 'protein sets differ across CSVs'
    print('All %d CSVs share the same %d test proteins.' % (len(csvs), len(feat)))

    results = []
    for f in FEATURES:
        for t in ALL_TARGETS:
            rs, ps, ns, keys = [], [], [], []
            for k, m in per_csv.items():
                sub = m[[f, t]].dropna()
                if len(sub) < 8 or sub[f].std() < 1e-12 or sub[t].std() < 1e-12:
                    continue
                r, p = stats.pearsonr(sub[f], sub[t])
                rs.append(float(r)); ps.append(float(p)); ns.append(len(sub)); keys.append(k)
            if len(rs) < 10:
                continue
            ar, ap = np.array(rs), np.array(ps)
            pos, neg = int((ar > 0).sum()), int((ar < 0).sum())
            results.append(dict(feature=f, target=t, k=len(rs),
                mean_r=float(ar.mean()), sd_r=float(ar.std(ddof=1)),
                min_r=float(ar.min()), max_r=float(ar.max()),
                sign_consistency=max(pos, neg), sign='+' if pos > neg else '-',
                n_sig_p05=int((ap < 0.05).sum()), n=int(np.median(ns)),
                per_csv_r=dict(zip(keys, rs)), per_csv_p=dict(zip(keys, ps))))

    for r in results:
        r['verdict'] = classify(r)
    json.dump(dict(n_csvs=len(csvs), csvs=[os.path.basename(c) for c in csvs],
                   n_test_proteins=int(len(feat)), results=results),
              open('results/bp_replication_sweep.json', 'w'), indent=1)

    hdr = '%-28s %-18s %7s %6s %7s %17s %6s %4s  %s'
    line = lambda r: hdr % (r['feature'], r['target'], '%+.3f' % r['mean_r'],
        '%.3f' % r['sd_r'], '%d/10%s' % (r['sign_consistency'], r['sign']),
        '%+.3f..%+.3f' % (r['min_r'], r['max_r']), '%d/10' % r['n_sig_p05'],
        r['n'], r['verdict'])

    print('\n' + '=' * 112)
    print('B_P-SIDE REPLICATION SWEEP: %d features x %d b_p targets over %d eval CSVs'
          % (len(FEATURES), len(BP_TARGETS), len(csvs)))
    print('=' * 112)
    bp_rows = sorted([r for r in results if r['target'] in BP_TARGETS],
                     key=lambda r: (-r['sign_consistency'], -r['n_sig_p05'], -abs(r['mean_r'])))
    print(hdr % ('feature', 'target', 'mean_r', 'sd', 'signs', 'range', 'sig', 'n', 'class'))
    print('-' * 112)
    for r in bp_rows:
        print(line(r))

    print('\n--- reference: a_p / pcc rows, same sweep (top 10) ---')
    ref = sorted([r for r in results if r['target'] in ('a_p', 'pcc')],
                 key=lambda r: (-r['sign_consistency'], -r['n_sig_p05'], -abs(r['mean_r'])))
    for r in ref[:10]:
        print(line(r))

    print('\n=== VERIFICATION OF REPORTED NUMBERS ===')
    for f, t in [('frac_buried_rel_lt_0.25', 'b_p_wt_error'),
                 ('SASA_per_residue', 'abs_b_p'),
                 ('length', 'abs_b_p'),
                 ('mean_rel_SASA', 'a_p')]:
        for r in results:
            if r['feature'] == f and r['target'] == t:
                print('\n%s vs %s' % (f, t))
                print('  mean_r=%+.4f sd=%.4f signs=%d/10 sig=%d/10 range=%+.3f..%+.3f  -> %s'
                      % (r['mean_r'], r['sd_r'], r['sign_consistency'], r['n_sig_p05'],
                         r['min_r'], r['max_r'], r['verdict']))
                for k in sorted(r['per_csv_r']):
                    print('    %-30s r=%+.4f  p=%.4g' % (k, r['per_csv_r'][k], r['per_csv_p'][k]))

    print('\n' + '=' * 112)
    print('ANCHOR-WEIGHT SUPPRESSION GRADIENT CHECK')
    print('=' * 112)
    print('Hypothesis: anchor arms suppress the structural signal; unanchored sigma seeds show it strongest.')
    grad = []
    for r in results:
        if r['target'] not in BP_TARGETS + ['a_p']:
            continue
        if r['sign_consistency'] < 10:
            continue
        anc = [abs(r['per_csv_r'][k]) for k in ANCHOR_ARMS if k in r['per_csv_r']]
        sig = [abs(r['per_csv_r'][k]) for k in SIGMA_ARMS if k in r['per_csv_r']]
        if len(anc) < 3 or len(sig) < 5:
            continue
        u, pu = stats.mannwhitneyu(sig, anc, alternative='greater')
        grad.append(dict(feature=r['feature'], target=r['target'],
            mean_abs_r_anchor=float(np.mean(anc)), mean_abs_r_sigma=float(np.mean(sig)),
            gap=float(np.mean(sig) - np.mean(anc)),
            all_sigma_above_all_anchor=bool(min(sig) > max(anc)),
            mwu_p_onesided=float(pu),
            anchor_r=[r['per_csv_r'][k] for k in ANCHOR_ARMS],
            sigma_r=[r['per_csv_r'][k] for k in SIGMA_ARMS]))
    grad.sort(key=lambda g: -g['gap'])
    gh = '%-28s %-18s %10s %10s %8s %8s %10s'
    print(gh % ('feature', 'target', '|r|anchor', '|r|sigma', 'gap', 'separ', 'MWU p'))
    print('-' * 100)
    for g in grad:
        print(gh % (g['feature'], g['target'], '%.3f' % g['mean_abs_r_anchor'],
              '%.3f' % g['mean_abs_r_sigma'], '%+.3f' % g['gap'],
              'YES' if g['all_sigma_above_all_anchor'] else 'no',
              '%.4f' % g['mwu_p_onesided']))
    json.dump(grad, open('results/bp_anchor_gradient.json', 'w'), indent=1)

    print('\nPer-arm detail for the headline pairs:')
    for g in grad:
        if (g['feature'], g['target']) in [('frac_buried_rel_lt_0.25', 'b_p_wt_error'),
                                           ('mean_rel_SASA', 'a_p'),
                                           ('SASA_per_residue', 'abs_b_p')]:
            print('\n  %s vs %s' % (g['feature'], g['target']))
            print('    anchor arms (w0.3,w1.0,w3.0): ' + ', '.join('%+.3f' % v for v in g['anchor_r']))
            print('    sigma seeds (1,2,3,4,42)    : ' + ', '.join('%+.3f' % v for v in g['sigma_r']))

    nb = sum(1 for r in bp_rows if r['verdict'] == 'ESTABLISHED')
    ns = sum(1 for r in bp_rows if r['verdict'] == 'SUGGESTIVE')
    nd = sum(1 for r in bp_rows if r['verdict'] == 'DEAD')
    print('\nB_P-SIDE TALLY: %d ESTABLISHED, %d SUGGESTIVE, %d DEAD (of %d pairs)'
          % (nb, ns, nd, len(bp_rows)))


if __name__ == '__main__':
    main()
