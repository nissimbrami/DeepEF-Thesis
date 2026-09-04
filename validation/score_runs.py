"""Score DeepEF fine-tuning eval CSVs (ddG metrics) — persistent replacement for /tmp helpers.

Eval CSVs (cols: protein, deltaG, pred_deltaG, ddG, pred_ddG) are produced by
Megascale-fineTuning/evaluate.py via --model_name eval_results/abl_<tag>_e<N>, and live in
eval_results/. This prints the per-epoch ddG PCC/SCC/RMSE/PCC-PP curve for each run tag.

Usage:
    conda run -n esm2_env_py38 python validation/score_runs.py            # all abl_* runs
    conda run -n esm2_env_py38 python validation/score_runs.py randscratch unf5core
"""
import os
import sys
import re
import glob
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
import benchmark_metrics as M

EVAL_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'eval_results')
BASELINE = "baseline 0.531/0.500/1.085/0.691 | freeze-best ddg_nopretrain 0.655 PCC / 0.738 PP"


def run_tags(files):
    tags = set()
    for f in files:
        m = re.search(r'abl_(.+)_e\d+\.csv$', os.path.basename(f))
        if m:
            tags.add(m.group(1))
    return sorted(tags)


def score(tag, eval_dir=EVAL_DIR):
    files = sorted(glob.glob(os.path.join(eval_dir, f'abl_{tag}_e*.csv')),
                   key=lambda x: int(re.search(r'_e(\d+)', x).group(1)))
    rows = []
    for f in files:
        e = int(re.search(r'_e(\d+)\.csv$', f).group(1))
        df = pd.read_csv(f)
        if 'pred_ddG' not in df.columns:
            continue
        per = [M.pearson(g['ddG'].values, g['pred_ddG'].values)
               for _, g in df.groupby('protein') if len(g) >= 3]
        rows.append((e,
                     M.pearson(df['ddG'].values, df['pred_ddG'].values),
                     M.spearman(df['ddG'].values, df['pred_ddG'].values),
                     M.rmse(df['ddG'].values, df['pred_ddG'].values),
                     float(np.nanmean(per)) if per else float('nan')))
    return rows


def main():
    want = sys.argv[1:]
    files = glob.glob(os.path.join(EVAL_DIR, 'abl_*.csv'))
    tags = want or run_tags(files)
    for tag in tags:
        rows = score(tag)
        if not rows:
            continue
        print(f"=== {tag} ===  epoch  PCC    SCC    RMSE   PCC-PP")
        for e, pcc, scc, rmse, pp in rows:
            print(f"           ep{e:<3}  {pcc:.3f}  {scc:.3f}  {rmse:.3f}  {pp:.3f}")
        bp = max(rows, key=lambda r: r[1]); bpp = max(rows, key=lambda r: r[4])
        print(f"   BEST PCC ep{bp[0]}={bp[1]:.3f} | BEST PCC-PP ep{bpp[0]}={bpp[4]:.3f}")
    print(BASELINE)


if __name__ == '__main__':
    main()
