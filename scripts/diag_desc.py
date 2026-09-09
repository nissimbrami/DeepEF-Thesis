#!/usr/bin/env python3
"""Why can the descriptor arm's loss not move? Two arms, two frozen-RMSE collapses, zero tests
of the actual hypothesis. Diagnose on CPU before spending another 12 GPU-hours.

Checks, in the order they would explain a frozen loss:
  1. does the descriptor CSV exist and parse?
  2. is the block constant (zero variance) -- a constant input carries no gradient
  3. does it contain NaN/inf -- one NaN poisons the whole forward pass
  4. what is its block-energy ratio vs one-hot (LORO's first death was 13.8x)
  5. does mordred_pca16_only actually REPLACE one-hot, or is one-hot still there?
"""
import os, sys, glob
import numpy as np, pandas as pd

ROOT='/home/nissimb/DeepPEF'
cands = glob.glob(f'{ROOT}/data*/aa_descriptors*.csv') + glob.glob(f'{ROOT}/data*/*mordred*.csv')
print("descriptor CSVs found:")
for c in cands: print("  ", c.replace(ROOT+'/',''), f"{os.path.getsize(c)} bytes")
if not cands:
    print("\nNONE FOUND -- if train.py cannot load the table it may fall back to a constant block,")
    print("which would explain a loss that never moves."); sys.exit(0)

for c in cands:
    print(f"\n=== {os.path.basename(c)} ===")
    try: d = pd.read_csv(c)
    except Exception as e: print("  UNREADABLE:", e); continue
    print(f"  shape {d.shape}  cols {list(d.columns)[:6]}")
    num = d.select_dtypes(include=[np.number])
    if num.empty: print("  NO NUMERIC COLUMNS -- the block would be empty/constant"); continue
    V = num.values.astype(float)
    print(f"  numeric block {V.shape}")
    print(f"  NaN: {int(np.isnan(V).sum())}   inf: {int(np.isinf(V).sum())}")
    sd = np.nanstd(V, axis=0)
    dead = int((sd < 1e-9).sum())
    print(f"  columns with ZERO variance: {dead} / {V.shape[1]}")
    if dead == V.shape[1]:
        print("  *** ALL COLUMNS CONSTANT -- no gradient can flow. This alone freezes the loss. ***")
    l2 = np.linalg.norm(np.nan_to_num(V), axis=1)
    print(f"  per-row L2: min {l2.min():.3f}  median {np.median(l2):.3f}  max {l2.max():.3f}")
    print(f"  vs one-hot L2 = 1.000  ->  energy ratio {np.median(l2):.2f}x")
    if np.median(l2) > 5:  print("  *** SCALE CLASH: this is the failure that killed the first arm ***")
    if np.median(l2) < 0.01: print("  *** BLOCK IS ~ZERO: contributes nothing, model sees no residue identity ***")
    if np.isnan(V).any(): print("  *** NaN PRESENT: one NaN poisons the entire forward pass ***")
