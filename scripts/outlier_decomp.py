# -*- coding: utf-8 -*-
"""Decompose the oracle gain: how much of it comes from the extreme-|b_p| proteins?

If the oracle gain is concentrated in a few outliers, then no smooth feature model can
recover it, and the "0.71 ceiling" is a statement about a handful of proteins rather than
about a learnable systematic miscalibration. Run over all 10 eval CSVs.
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

out = {}
for f in sorted(x for x in os.listdir(eval_dir) if x.endswith('.csv')):
    path = os.path.join(eval_dir, f)
    df = pd.read_csv(path)
    bp = cat.recover_bp(path)
    y = dict(zip(bp['protein'], bp['b_p']))
    raw, _ = oc.pooled_pcc(df)
    orc, _ = oc.pooled_pcc(df, y)
    gain = orc - raw
    order = sorted(y, key=lambda k: -abs(y[k]))
    rec = dict(raw=raw, oracle=orc, gain=gain, top=[])
    for k in (1, 2, 3, 5):
        keep = set(order[:k])
        off = {n: (v if n in keep else 0.0) for n, v in y.items()}
        p, _ = oc.pooled_pcc(df, off)
        rec['top'].append(dict(k=k, proteins=order[:k], pooled=p,
                               frac_of_oracle_gain=(p - raw) / gain if gain else float('nan')))
    keep2 = set(order[:2])
    off = {n: (0.0 if n in keep2 else v) for n, v in y.items()}
    p, _ = oc.pooled_pcc(df, off)
    rec['all_except_top2'] = dict(pooled=p, frac_of_oracle_gain=(p - raw) / gain)
    out[f] = rec
    t2 = rec['top'][1]
    print('%-38s raw %.4f orc %.4f | top2 %s = %.0f%% of gain | rest = %.0f%%'
          % (f, raw, orc, ','.join(order[:2]), 100 * t2['frac_of_oracle_gain'],
             100 * rec['all_except_top2']['frac_of_oracle_gain']))

fr = [v['top'][1]['frac_of_oracle_gain'] for v in out.values()]
print('\nmean frac of oracle gain from top-2 |b_p| proteins: %.0f%% (range %.0f-%.0f%%)'
      % (100 * np.mean(fr), 100 * min(fr), 100 * max(fr)))
with open(os.path.join(REPO, 'results/offset_corrector_outliers.json'), 'w') as fh:
    json.dump(out, fh, indent=2, default=float)
print('[wrote] results/offset_corrector_outliers.json')
