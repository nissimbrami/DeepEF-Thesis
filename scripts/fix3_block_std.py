"""FIX 3: is the slope term measuring a protein's spread, or a contiguous block's?

The slope term computes output.std() over one mini-batch of contiguous, UNSHUFFLED
variants. If the mutation CSV is ordered by position, each block spans a narrow
sequence window whose true ddG spread is systematically narrower than the protein's --
and the term would then calibrate the model to a block's spread, which is the exact
failure it was written to fix.

Decision rule (FIXES.md FIX 3):
  ratio >= 0.9  -> blocks are representative, the term is fine as written
  ratio <= 0.7  -> real; add shuffling behind a default-off flag as a fifth factor
"""
import os, re, sys, json
import numpy as np
import pandas as pd

MUT_DIR = sys.argv[1] if len(sys.argv) > 1 else './data/Processed_K50_dG_datasets/mutation_datasets'
MB = 64
MIN_VARIANTS = 3 * MB

def position_of(mut_type):
    m = re.search(r'(\d+)', str(mut_type))
    return int(m.group(1)) if m else None

rows = []
files = sorted(f for f in os.listdir(MUT_DIR) if f.endswith('.csv'))
for fn in files:
    try:
        df = pd.read_csv(os.path.join(MUT_DIR, fn))
    except Exception:
        continue
    if 'ddG_ML' not in df.columns or 'mut_type' not in df.columns:
        continue
    # match training: drop ins/del and multi-mutants, keep row order as the loader sees it
    df = df[~df['mut_type'].astype(str).str.contains('ins|del', na=False)]
    df = df[~df['mut_type'].astype(str).str.contains(':', na=False)].reset_index(drop=True)
    d = pd.to_numeric(df['ddG_ML'], errors='coerce').values[1:]   # skip WT at row 0
    keep = ~np.isnan(d)
    d = d[keep]
    if d.size < MIN_VARIANTS:
        continue
    prot_std = float(d.std())
    if prot_std <= 0:
        continue
    blocks = [d[i:i + MB].std() for i in range(0, d.size - MB + 1, MB)]
    if not blocks:
        continue
    # is the file position-ordered?
    pos = [position_of(x) for x in df['mut_type'].values[1:]][:len(keep)]
    pos = [p for p, k in zip(pos, keep) if k and p is not None]
    ordered = bool(len(pos) > 1 and all(pos[i] <= pos[i + 1] for i in range(len(pos) - 1)))
    rows.append(dict(protein=fn[:-4], n=int(d.size), n_blocks=len(blocks),
                     protein_std=prot_std, mean_block_std=float(np.mean(blocks)),
                     ratio=float(np.mean(blocks) / prot_std), position_ordered=ordered))

if not rows:
    print('NO PROTEINS QUALIFIED -- check the path:', MUT_DIR)
    sys.exit(1)

r = np.array([x['ratio'] for x in rows])
n_ord = sum(1 for x in rows if x['position_ordered'])
summary = dict(mini_batch=MB, n_proteins=len(rows),
               ratio_median=float(np.median(r)),
               ratio_q1=float(np.percentile(r, 25)), ratio_q3=float(np.percentile(r, 75)),
               ratio_min=float(r.min()), ratio_max=float(r.max()),
               n_position_ordered=n_ord, frac_position_ordered=n_ord / len(rows))
med = summary['ratio_median']
summary['verdict'] = ('FINE_AS_WRITTEN (ratio >= 0.9)' if med >= 0.9 else
                      'REAL_CONCERN (ratio <= 0.7) -- add shuffling behind a default-off flag' if med <= 0.7 else
                      'BORDERLINE (0.7 < ratio < 0.9) -- report as a stated property; do not change runs in flight')

print(json.dumps(summary, indent=2))
print('\nper-protein (first 15):')
for x in rows[:15]:
    print('  %-28s n=%5d  blocks=%3d  prot_std=%.3f  block_std=%.3f  ratio=%.3f  ordered=%s'
          % (x['protein'], x['n'], x['n_blocks'], x['protein_std'], x['mean_block_std'],
             x['ratio'], x['position_ordered']))
with open('results/fix3_block_std.json', 'w') as f:
    json.dump(dict(summary=summary, per_protein=rows), f, indent=2)
print('\nwrote results/fix3_block_std.json')
