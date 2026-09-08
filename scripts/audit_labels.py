"""INTEGRITY AUDIT part 2+3: labels and tensors for the 28 test proteins.

Checks per protein:
  L1 ddG column internally consistent with deltaG - row0 deltaG (in the EVAL csv)
  L2 duplicate mutation names in the source mutation file
  L3 duplicate mut_type entries
  L4 NaN / non-finite deltaG, ddG_ML
  L5 impossible dG values (outside a physical window)
  L6 number of distinct WT backgrounds
  L7 rows with ddG_ML == '-'  (the "unstable" flag)
  T1 coords / mask / one_hot / embedding shapes agree with each other
  T2 tensor row count == mutation-file row count (raw, and after the ins/del filter)
  T3 deltaG.pt length vs mutation file length
  T4 the aa_seq in the mutation file agrees with the one_hot argmax  <-- the alignment test
"""
import os, sys, glob, csv, math, json
import numpy as np, pandas as pd, torch

TENSOR_ROOT = '/groups/keasar_group/casp15/meytav/protein_tensors'
MUT_DIR     = 'data/Processed_K50_dG_datasets/mutation_datasets'
FIX_DIR     = 'data_fixed/mutation_datasets'
EVAL        = 'eval_results/abl_sigma_seed42_e9.csv'

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def mutfile(p):
    f = os.path.join(FIX_DIR, p + '.csv')
    if os.path.exists(f): return f, 'data_fixed'
    return os.path.join(MUT_DIR, p + '.csv'), 'data'

ev = pd.read_csv(EVAL)
proteins = sorted(ev['protein'].unique())
print("proteins in eval CSV:", len(proteins))

rows = []
for p in proteins:
    rec = {'protein': p}
    sub = ev[ev['protein'] == p].reset_index(drop=True)
    # L1: ddG == deltaG - deltaG[0]
    exp = sub['deltaG'].values - sub['deltaG'].values[0]
    bad = int(np.sum(np.abs(exp - sub['ddG'].values) > 1e-4))
    rec['eval_n'] = len(sub)
    rec['L1_ddG_inconsistent'] = bad
    rec['eval_zero_ddG_rows'] = int(np.sum(sub['ddG'].values == 0.0))
    rec['eval_nan'] = int(sub[['deltaG','pred_deltaG','ddG','pred_ddG']].isna().sum().sum())

    f, src = mutfile(p)
    rec['mutsrc'] = src
    m = pd.read_csv(f)
    rec['mut_rows_raw'] = len(m)
    mf = m[~m['mut_type'].astype(str).str.contains('ins|del')].reset_index(drop=True)
    rec['mut_rows_filtered'] = len(mf)
    rec['L2_dup_names'] = int(len(m) - m['name'].nunique())
    rec['L3_dup_muttype'] = int(len(m) - m['mut_type'].nunique())
    dg = pd.to_numeric(m['deltaG'], errors='coerce')
    rec['L4_nan_deltaG'] = int(dg.isna().sum())
    rec['L5_dg_min'] = float(np.nanmin(dg)); rec['L5_dg_max'] = float(np.nanmax(dg))
    rec['L5_impossible'] = int(np.sum((dg < -10) | (dg > 20)))
    wtn = set(str(x['name']).split('_wt')[0].strip()
              for _, x in m.iterrows() if str(x.get('mut_type','')).strip().lower() == 'wt')
    rec['L6_n_backgrounds'] = len(wtn)
    rec['L7_unstable_dash'] = int((m['ddG_ML'].astype(str) == '-').sum())

    # tensors
    d = os.path.join(TENSOR_ROOT, p)
    try:
        co = torch.load(os.path.join(d,'coords_tensor.pt'), map_location='cpu', weights_only=False)
        mk = torch.load(os.path.join(d,'mask_tensor.pt'), map_location='cpu', weights_only=False) \
             if os.path.exists(os.path.join(d,'mask_tensor.pt')) else None
        oh = torch.load(os.path.join(d,'one_hot_encodings.pt'), map_location='cpu', weights_only=False)
        gg = torch.load(os.path.join(d,'deltaG.pt'), map_location='cpu', weights_only=False)
        embs = sorted(glob.glob(os.path.join(d,'prott5_embeddings','prott5_embedding_*.pt')),
                      key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        eb = torch.vstack([torch.load(x, map_location='cpu', weights_only=False) for x in embs])
        rec['coords'] = tuple(co.shape); rec['one_hot'] = tuple(oh.shape)
        rec['emb'] = tuple(eb.shape); rec['dG_pt'] = tuple(gg.shape)
        rec['mask'] = tuple(mk.shape) if mk is not None else None
        # T1/T2/T3
        rec['T_nvar_oh'] = oh.shape[0]; rec['T_nvar_emb'] = eb.shape[0]; rec['T_nvar_dg'] = gg.shape[0]
        rec['T_L_coords'] = co.shape[0]; rec['T_L_oh'] = oh.shape[1]; rec['T_L_emb'] = eb.shape[1]
        rec['T1_nvar_agree'] = (oh.shape[0]==eb.shape[0]==gg.shape[0])
        rec['T1_len_agree']  = (co.shape[0]==oh.shape[1]==eb.shape[1])
        rec['T2_dg_vs_raw']  = int(gg.shape[0] - len(m))
        rec['T2_dg_vs_filt'] = int(gg.shape[0] - len(mf))
        # T3: does deltaG.pt match the CSV deltaG column, in order?
        n = min(gg.shape[0], len(m))
        gnp = gg.cpu().numpy().reshape(-1)[:n]
        rec['T3_max_absdiff_raw'] = float(np.nanmax(np.abs(gnp - dg.values[:n])))
        nf = min(gg.shape[0], len(mf))
        dgf = pd.to_numeric(mf['deltaG'], errors='coerce').values[:nf]
        rec['T3_max_absdiff_filt'] = float(np.nanmax(np.abs(gg.cpu().numpy().reshape(-1)[:nf] - dgf)))
        # T4: one_hot argmax vs aa_seq  (WT row 0)
        oh0 = oh[0].cpu().numpy()
        idx = oh0[:, :20].argmax(axis=1)
        decoded = ''.join(AA20[i] for i in idx)
        seq0 = str(m['aa_seq'].iloc[0])
        rec['T4_seqlen_csv'] = len(seq0)
        rec['T4_decoded_eq'] = (decoded == seq0)
        rec['T4_ident'] = float(np.mean([a==b for a,b in zip(decoded, seq0)])) if len(decoded)==len(seq0) else float('nan')
        rec['T4_decoded'] = decoded[:20]; rec['T4_csvseq'] = seq0[:20]
    except Exception as e:
        rec['tensor_error'] = repr(e)[:200]
    rows.append(rec)
    print('.', end='', flush=True)
print()
df = pd.DataFrame(rows)
df.to_csv('/home/nissimb/audit_labels.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(df[['protein','eval_n','L1_ddG_inconsistent','eval_zero_ddG_rows','eval_nan',
          'mut_rows_raw','mut_rows_filtered','L2_dup_names','L3_dup_muttype',
          'L4_nan_deltaG','L5_impossible','L6_n_backgrounds','L7_unstable_dash']].to_string(index=False))
print()
print(df[['protein','coords','one_hot','emb','dG_pt','mask','T1_nvar_agree','T1_len_agree',
          'T2_dg_vs_raw','T2_dg_vs_filt','T3_max_absdiff_raw','T3_max_absdiff_filt',
          'T4_seqlen_csv','T4_decoded_eq','T4_ident']].to_string(index=False))
if 'tensor_error' in df: print(df[df['tensor_error'].notna()][['protein','tensor_error']].to_string(index=False))
