# -*- coding: utf-8 -*-
"""
catalogue_vs_bp.py -- does any structural/complexity covariate explain our per-protein
offset b_p (or slope a_p, or per-protein PCC) on the 28 MegaScale test proteins?

THE METRIC RULE: b_p and a_p are per-protein calibration objects. Anything identical
between WT and mutant cancels in ddG, so a whole-protein property (BSA, ligands, chain
count, hydrophobicity, length) can ONLY be scored against b_p / a_p / per-protein PCC --
never against pooled ddG, and never against the pre-training decoy loss (which is decoy
discrimination on PDB and is already known not to predict ddG).

b_p / a_p conventions match cluster_run/code/calib_diag.py EXACTLY:
  - group by protein, row 0 of each group is WT
  - ddG_true = deltaG - deltaG[0];  ddG_pred = pred_deltaG - pred_deltaG[0]
  - a_p, b_p = np.polyfit(ddg_true, ddg_pred, 1)   (pred ~= a*true + b)
  - per-protein PCC = pearson(ddg_true, ddg_pred)
We additionally record b_p_wt_error = the model's raw WT dG error, which is the
"offset" in the dG sense (the quantity the coil lever moved).
"""
import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# Kyte-Doolittle hydropathy
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
      'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
      'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
             'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}
# Tien et al. 2013 theoretical max residue SASA (A^2), for relative SASA
MAXSASA = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'Q': 225, 'E': 223,
           'G': 104, 'H': 224, 'I': 197, 'L': 201, 'K': 236, 'M': 224, 'F': 240,
           'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174}
METALS = ('ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'NA', 'K', 'NI', 'CO')
DESIGNED_RE = re.compile(r'(?:HHH|HEEH|EEHEE|EHEE|EHHE|HHHH|_TrROS_|v2_)', re.IGNORECASE)


def per_protein_fit(ddg_true, ddg_pred):
    """Identical to calib_diag.per_protein_fit."""
    if np.std(ddg_true) < 1e-8:
        return 1.0, float(np.mean(ddg_pred) - np.mean(ddg_true))
    a, b = np.polyfit(ddg_true, ddg_pred, 1)
    return float(a), float(b)


def recover_bp(csv_path):
    df = pd.read_csv(csv_path)
    for col in ('protein', 'deltaG', 'pred_deltaG'):
        if col not in df.columns:
            raise SystemExit('missing column %r; got %s' % (col, list(df.columns)))
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
        rows.append(dict(protein=str(name), n=int(len(g)), a_p=a, b_p=b,
                         abs_b_p=abs(b), pcc=pcc, b_p_wt_error=wt_err,
                         abs_b_p_wt_error=abs(wt_err),
                         designed=bool(DESIGNED_RE.search(str(name)))))
    return pd.DataFrame(rows)


def structural_features(names, pdb_dir, plddt_csv):
    """Compute real structural covariates from the actual AlphaFold model of each test
    protein. These are the honest stand-ins for the catalogue's BSA / complexity /
    ligand / hydrophobicity columns, computed on OUR proteins."""
    import copy
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    pl = pd.read_csv(plddt_csv)
    pl_mean = pl.groupby('protein')['plddt'].mean()
    pl_min = pl.groupby('protein')['plddt'].min()
    out = []
    for nm in names:
        p = os.path.join(pdb_dir, nm + '.pdb')
        rec = dict(protein=nm)
        if not os.path.exists(p):
            out.append(rec)
            continue
        st = parser.get_structure(nm, p)
        model = st[0]
        chains = list(model)
        rec['n_chains'] = len(chains)
        het, metals = 0, 0
        for res in model.get_residues():
            hf = res.get_id()[0]
            if hf.strip() and hf != 'W':
                het += 1
                if res.get_resname().strip() in METALS:
                    metals += 1
        rec['n_het_residues'] = het
        rec['n_metal_residues'] = metals
        seq = [THREE2ONE.get(r.get_resname().strip())
               for r in model.get_residues() if r.get_id()[0] == ' ']
        seq = [s for s in seq if s]
        rec['length'] = len(seq)
        if seq:
            rec['mean_hydropathy_KD'] = float(np.mean([KD[s] for s in seq]))
            rec['frac_hydrophobic'] = float(np.mean([s in 'AVILMFWC' for s in seq]))
            rec['frac_charged'] = float(np.mean([s in 'DEKR' for s in seq]))
            rec['frac_glycine'] = float(np.mean([s == 'G' for s in seq]))
            rec['frac_proline'] = float(np.mean([s == 'P' for s in seq]))
        sr.compute(model, level='R')
        res_list = [r for r in model.get_residues() if r.get_id()[0] == ' ']
        sasa = np.array([r.sasa for r in res_list], dtype=float)
        rec['total_SASA'] = float(np.sum(sasa))
        rec['SASA_per_residue'] = float(np.mean(sasa)) if len(sasa) else float('nan')
        rel, hyd_exposed = [], []
        for r in res_list:
            one = THREE2ONE.get(r.get_resname().strip())
            if one and MAXSASA.get(one):
                rr = r.sasa / MAXSASA[one]
                rel.append(rr)
                if one in 'AVILMFWC':
                    hyd_exposed.append(rr)
        if rel:
            rel = np.array(rel)
            rec['mean_rel_SASA'] = float(np.mean(rel))
            rec['frac_buried_rel_lt_0.25'] = float(np.mean(rel < 0.25))
            rec['frac_exposed_rel_gt_0.5'] = float(np.mean(rel > 0.5))
        if hyd_exposed:
            rec['mean_rel_SASA_hydrophobic'] = float(np.mean(hyd_exposed))
        if rec.get('length'):
            rec['SASA_over_len_pow_073'] = float(rec['total_SASA'] / (rec['length'] ** 0.73))
        if len(chains) > 1:
            tot_iso = 0.0
            for ch in chains:
                c2 = copy.deepcopy(ch)
                sr.compute(c2, level='R')
                tot_iso += sum(r.sasa for r in c2.get_residues() if r.get_id()[0] == ' ')
            rec['interchain_BSA'] = float(tot_iso - rec['total_SASA'])
        else:
            rec['interchain_BSA'] = 0.0
        if nm in pl_mean.index:
            rec['plddt_mean'] = float(pl_mean[nm])
            rec['plddt_min'] = float(pl_min[nm])
        out.append(rec)
    return pd.DataFrame(out)


def corr_table(feat_df, targets, min_n=8):
    rows = []
    numeric = [c for c in feat_df.columns
               if c != 'protein' and pd.api.types.is_numeric_dtype(feat_df[c])]
    skip = set(targets) | {'n'}
    for col in numeric:
        if col in skip:
            continue
        for tgt in targets:
            sub = feat_df[[col, tgt]].dropna()
            if len(sub) < min_n or sub[col].nunique() < 3:
                continue
            x = sub[col].values.astype(float)
            y = sub[tgt].values.astype(float)
            pr, pp_ = stats.pearsonr(x, y)
            sr_, sp_ = stats.spearmanr(x, y)
            rows.append(dict(feature=col, target=tgt, n=int(len(sub)),
                             pearson_r=float(pr), pearson_p=float(pp_),
                             spearman_r=float(sr_), spearman_p=float(sp_),
                             max_abs_r=float(max(abs(pr), abs(sr_)))))
    t = pd.DataFrame(rows)
    if len(t):
        t = t.sort_values('max_abs_r', ascending=False).reset_index(drop=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_csv', default='eval_results/abl_calib_ctrl_repro2_e14.csv')
    ap.add_argument('--catalogue', default='data/FINAL_DATASET_100k_030926.csv')
    ap.add_argument('--pdb_dir', default='data/Processed_K50_dG_datasets/AlphaFold_model_PDBs')
    ap.add_argument('--plddt', default='data/Processed_K50_dG_datasets/plddt.csv')
    ap.add_argument('--out_json', default='results/catalogue_vs_bp.json')
    args = ap.parse_args()

    cat = pd.read_csv(args.catalogue, low_memory=False)
    cat_cols = []
    for c in cat.columns:
        s = cat[c]
        cat_cols.append(dict(column=c, dtype=str(s.dtype), n_unique=int(s.nunique()),
                             n_null=int(s.isna().sum()),
                             example=str(s.dropna().iloc[0])[:60] if s.notna().any() else None))

    bp = recover_bp(args.eval_csv)
    names = list(bp['protein'])

    pid = cat['PDB_ID_and_Entity'].astype(str)
    rawid = cat['protein_id'].astype(str)
    pdb4 = pid.str.replace(r'^\d+#', '', regex=True).str.split('_').str[0].str.upper()
    join = []
    for nm in names:
        key = nm.upper()
        exact = int((pdb4 == key).sum())
        sub_pid = int(pid.str.contains(nm, case=False, regex=False).sum())
        sub_raw = int(rawid.str.contains(nm, case=False, regex=False).sum())
        join.append(dict(protein=nm, pdb4_key=key, n_exact_pdb4=exact,
                         n_substring_PDB_ID_and_Entity=sub_pid,
                         n_substring_protein_id=sub_raw,
                         matched=bool(exact or sub_pid or sub_raw)))
    join_df = pd.DataFrame(join)
    n_matched = int(join_df['matched'].sum())

    feat = structural_features(names, args.pdb_dir, args.plddt)
    merged = bp.merge(feat, on='protein', how='left')

    targets = ['b_p', 'abs_b_p', 'a_p', 'pcc', 'b_p_wt_error', 'abs_b_p_wt_error']
    tbl = corr_table(merged, targets)

    n28 = len(merged)
    tcrit = stats.t.ppf(0.975, n28 - 2)
    rcrit = float(tcrit / np.sqrt(tcrit ** 2 + n28 - 2))

    out = {
        'eval_csv': args.eval_csv,
        'n_test_proteins': n28,
        'r_crit_p05': rcrit,
        'catalogue': {
            'path': args.catalogue,
            'shape': list(cat.shape),
            'columns': cat_cols,
            'column_roles': {
                'BSA_interface': ['BSA', 'BSA_Percentage', 'BSA_Numeric_x', 'BSA_Numeric_y'],
                'complexity_chains': ['Oligomeric State', 'Is Complex?', 'Chain Composition',
                                      'Global Symmetry'],
                'ligands': ['Ligands_x', 'Ligands_y'],
                'hydrophobicity': [],
                'resolution': ['Resolution (Angstrom)'],
                'pLDDT': [],
                'EC_number': [],
                'organism': [],
                'length': ['AA Length'],
            },
        },
        'join': {
            'key_tried': 'PDB 4-char id (catalogue PDB_ID_and_Entity / protein_id, '
                         'validation NN# prefix stripped) vs eval protein name; plus '
                         'case-insensitive substring scan of both catalogue id columns',
            'n_of_28_matched': n_matched,
            'per_protein': join,
        },
        'per_protein_calibration': bp.to_dict(orient='records'),
        'structural_features_source': {
            'pdb_dir': args.pdb_dir, 'plddt': args.plddt,
            'note': 'computed directly on the 28 test proteins because the catalogue join is empty',
        },
        'merged_table': merged.to_dict(orient='records'),
        'correlations': tbl.to_dict(orient='records'),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)

    print('=' * 78)
    print('CATALOGUE: %s  shape=%s' % (args.catalogue, cat.shape))
    for c in cat_cols:
        print('  %-32s %-9s nuniq=%6d nnull=%6d ex=%s'
              % (c['column'].encode('ascii', 'replace').decode(), c['dtype'],
                 c['n_unique'], c['n_null'], c['example']))
    print()
    print('JOIN: matched %d / %d of the test proteins' % (n_matched, len(names)))
    print()
    print('PER-PROTEIN CALIBRATION (%s)' % args.eval_csv)
    cols = ['protein', 'n', 'a_p', 'b_p', 'pcc', 'b_p_wt_error', 'designed']
    print(bp[cols].sort_values('b_p_wt_error').to_string(index=False))
    print()
    print('FEATURES (28 proteins, computed from AlphaFold models)')
    print(merged.drop(columns=['abs_b_p', 'abs_b_p_wt_error']).to_string(index=False))
    print()
    print('n=%d -> |r| < %.3f is indistinguishable from 0 at p=0.05' % (n28, rcrit))
    print()
    print('TOP CORRELATIONS (ranked by max(|pearson|,|spearman|))')
    if len(tbl):
        print(tbl.head(45).to_string(index=False))
    print()
    print('[wrote] %s' % args.out_json)


if __name__ == '__main__':
    main()
