"""P8 probe: compute pooled ddG PCC / offset-removal oracle / gain on every candidate basis."""
import os, glob, re
import numpy as np
import pandas as pd

EVAL = 'eval_results'
EXCLUDE_PROTEIN = '2K5H'

def pooled_and_oracle(df):
    d = df[df['protein'] != EXCLUDE_PROTEIN]
    t = d['ddG'].values.astype(float)
    p = d['pred_ddG'].values.astype(float)
    pooled = np.corrcoef(t, p)[0, 1]
    # offset-removal oracle: subtract each protein's own mean residual (t - p)
    q = p.copy()
    for prot, idx in d.groupby('protein').indices.items():
        off = np.mean(t[idx] - p[idx])
        q[idx] = p[idx] + off
    oracle = np.corrcoef(t, q)[0, 1]
    return pooled, oracle, d['protein'].nunique(), len(d)

def factorD(name):
    if 'D1_uemb' in name: return 'D1'
    if 'D0_coil' in name: return 'D0'
    return 'D0'   # non-factorial runs are the default (no --unfolded_emb zero)

rows = []
for f in sorted(glob.glob(os.path.join(EVAL, 'abl_*.csv'))):
    b = os.path.basename(f)
    tag = re.sub(r'^abl_', '', b)
    tag = re.sub(r'\.csv$', '', tag)
    m = re.search(r'_e(\d+)$', tag)
    ep = int(m.group(1)) if m else -1
    run = re.sub(r'_e\d+$', '', tag)
    df = pd.read_csv(f)
    po, orc, np_, n = pooled_and_oracle(df)
    rows.append(dict(file=b, tag=tag, run=run, epoch=ep, D=factorD(b),
                     pooled=po, oracle=orc, gain=orc - po, nprot=np_, nrow=n,
                     family=('p3' if run.startswith('p3_') else
                             'gld' if run.startswith('gld_') else 'orig')))
R = pd.DataFrame(rows)
R.to_csv('/home/nissimb/DeepPEF/results/05_infrastructure/_p8_all_csvs.csv', index=False)

pd.set_option('display.width', 200)
print(R[['file','family','D','epoch','pooled','oracle','gain']].to_string(index=False))
print()
print('=== families ===')
for fam, g in R.groupby('family'):
    print(fam, len(g), sorted(g.run.unique()))
