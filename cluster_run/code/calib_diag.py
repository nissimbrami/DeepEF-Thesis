# TESTED: in isolation only -- validated end-to-end on /tmp/rescore_e{8,12,14}.csv and via
#   `--selftest` (synthetic ground-truth CSV). The decisive 0.77-0.81 affine-oracle check needs a
#   real Shahar per-mutation eval CSV, which lives ONLY on the cluster
#   (/mnt/new_groups/keasar_group/casp15/Shahar/DeepPEF/eval_results/); it is NOT on this machine.
#   The /tmp CSVs are a random-init dg-objective rescore (NOT one of the 52 evaluated checkpoints),
#   so their affine-oracle lands ~0.67, exactly as expected -- that is NOT the thesis validation.
#
# calib_diag.py — the PRIMARY MEASUREMENT INSTRUMENT of the thesis.
#
# The thesis claims per-protein calibration is the bottleneck: for each protein p,
#   pred_ddG ~= a_p * true_ddG + b_p
# where b_p is the OFFSET (= model's error on the WT deltaG) and a_p is the SLOPE
# (compression / under-reaction). Removing a_p and b_p with an oracle lifts every
# checkpoint ever evaluated to pooled 0.77-0.81. This script MEASURES those objects
# directly. It is not a reporting convenience: std(b) and the slope distribution are
# the actual quantities every lever is judged against.
#
# INPUT: a per-mutation CSV with columns:
#   protein, deltaG, pred_deltaG        (row 0 of each protein group = WT)
# ddG is computed WITHIN each protein group:  ddG = deltaG - deltaG.iloc[0]
# (Rule 3: pooled or per-protein . dG or ddG . which split & selection — this is
#  per-protein ddG on whatever split the CSV came from.)
#
# OUTPUTS (all on the ddG scale, groups with len>=3 for per-protein stats):
#   - pooled ddG PCC                    (raw, no correction)
#   - PP (mean per-protein ddG PCC)
#   - std(b): spread of per-protein offsets  b_p = mean(pred_ddG) - a_p*mean(true_ddG)
#   - slope distribution: min / median / max of a_p, and count(a_p < 0.3) (collapse)
#   - OFFSET-REMOVED pooled PCC: subtract each protein's b_p; the thesis oracle target
#   - AFFINE-REMOVED (a_p AND b_p) pooled PCC: the full oracle -> should land 0.77-0.81
#   - designed-fold subset: same stats restricted to designed folds (regex below)
#
# END-TO-END VALIDATION: on a real evaluated checkpoint, AFFINE-REMOVED pooled PCC must
# land in [0.77, 0.81]. That is the single most valuable self-check in the whole build:
# if the oracle does not reach 0.77-0.81, the identity in the thesis is not holding on
# this CSV and every downstream lever number is suspect.
#
# Usage:
#   python Megascale-fineTuning/calib_diag.py --csv /tmp/rescore_e12.csv
#   python Megascale-fineTuning/calib_diag.py --csv <eval.csv> --json out.json

import argparse
import json
import re

import numpy as np
import pandas as pd

# Designed-fold name patterns (K50/MegaScale designed scaffolds). Matched case-insensitively
# against the protein name. These are the folds where slope collapse (a_p -> 0.07-0.18) is worst.
DESIGNED_RE = re.compile(r'(?:HHH|HEEH|EEHEE|EHEE|EHHE|HHHH|_TrROS_|v2_)', re.IGNORECASE)


def _pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def per_protein_fit(ddg_true, ddg_pred):
    """Least-squares a,b for pred ~= a*true + b. Returns (a, b). Falls back to a=1 if
    true has no variance (cannot identify a slope from a constant regressor)."""
    if np.std(ddg_true) < 1e-8:
        return 1.0, float(np.mean(ddg_pred) - np.mean(ddg_true))
    a, b = np.polyfit(ddg_true, ddg_pred, 1)
    return float(a), float(b)


def analyze(df, label='ALL'):
    pooled_true, pooled_pred = [], []
    offset_removed_pred = []   # pred - b_p
    affine_removed_pred = []   # (pred - b_p) / a_p  -> back onto true scale
    pp_corrs = []
    slopes = []
    offsets = []

    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        ddg_true = g['deltaG'].values - g['deltaG'].values[0]
        ddg_pred = g['pred_deltaG'].values - g['pred_deltaG'].values[0]

        pooled_true.append(ddg_true)
        pooled_pred.append(ddg_pred)

        a, b = per_protein_fit(ddg_true, ddg_pred)
        # offset-removed: shift predictions down by b_p (oracle knows b_p)
        offset_removed_pred.append(ddg_pred - b)
        # affine-removed: also divide out the slope, mapping pred back to true scale.
        # guard tiny/negative slopes so a collapsed protein does not explode the metric.
        a_safe = a if abs(a) > 1e-3 else (1e-3 if a >= 0 else -1e-3)
        affine_removed_pred.append((ddg_pred - b) / a_safe)

        if len(g) >= 3:
            slopes.append(a)
            offsets.append(b)
            c = _pearson(ddg_true, ddg_pred)
            if not np.isnan(c):
                pp_corrs.append(c)

    if not pooled_true:
        print(f'[calib_diag:{label}] no usable protein groups.')
        return None

    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    por = np.concatenate(offset_removed_pred)
    par = np.concatenate(affine_removed_pred)

    pooled = _pearson(pt, pp)
    pooled_offset_removed = _pearson(pt, por)
    pooled_affine_removed = _pearson(pt, par)
    pp_mean = float(np.nanmean(pp_corrs)) if pp_corrs else float('nan')

    slopes = np.array(slopes, dtype=float)
    offsets = np.array(offsets, dtype=float)
    std_b = float(np.std(offsets)) if len(offsets) else float('nan')

    res = {
        'label': label,
        'n_proteins': int(df['protein'].nunique()),
        'n_pp_groups': int(len(slopes)),
        'pooled_ddG_PCC': pooled,
        'PP_mean_perprotein_PCC': pp_mean,
        'std_b': std_b,
        'slope_min': float(np.min(slopes)) if len(slopes) else float('nan'),
        'slope_median': float(np.median(slopes)) if len(slopes) else float('nan'),
        'slope_max': float(np.max(slopes)) if len(slopes) else float('nan'),
        'n_slope_collapsed_lt_0.3': int(np.sum(slopes < 0.3)) if len(slopes) else 0,
        'pooled_offset_removed_PCC': pooled_offset_removed,
        'pooled_affine_removed_PCC': pooled_affine_removed,
    }
    return res


def print_block(res):
    if res is None:
        return
    L = res['label']
    print(f'==================== calib_diag [{L}] ====================')
    print(f'[{L}] proteins={res["n_proteins"]}  pp_groups(len>=3)={res["n_pp_groups"]}')
    print(f'[{L}] pooled ddG PCC            = {res["pooled_ddG_PCC"]:.4f}')
    print(f'[{L}] PP (mean per-protein)     = {res["PP_mean_perprotein_PCC"]:.4f}')
    print(f'[{L}] std(b)  [offset spread]   = {res["std_b"]:.4f}')
    print(f'[{L}] slope a_p  min/med/max    = '
          f'{res["slope_min"]:.3f} / {res["slope_median"]:.3f} / {res["slope_max"]:.3f}')
    print(f'[{L}] slope collapsed (a<0.3)   = {res["n_slope_collapsed_lt_0.3"]}')
    print(f'[{L}] pooled OFFSET-removed PCC = {res["pooled_offset_removed_PCC"]:.4f}  (remove b_p)')
    print(f'[{L}] pooled AFFINE-removed PCC = {res["pooled_affine_removed_PCC"]:.4f}  '
          f'(remove a_p & b_p -> oracle, target 0.77-0.81)')


def selftest():
    """Prove the instrument on synthetic data of KNOWN ground truth (no real data, no GPU).
    Build proteins whose within-protein ddG is a clean signal with slope a=1 but each protein
    carries a large random offset b (std 3.0). Raw pooled PCC should be crushed by the offsets;
    OFFSET-removed pooled PCC should jump back to the within-protein signal (~0.97). Also injects
    designed-named folds so the DESIGNED regex is exercised. Exits nonzero on any mismatch."""
    rng = np.random.RandomState(0)
    rows = []
    names = ['prot_a', 'prot_b', 'prot_c', 'prot_d', 'prot_e',
             'HHH_1', 'HEEH_2', 'EEHEE_3', 'v2_x', '2abc_TrROS_y']
    # b_p is the per-protein OFFSET in ddG space: the model's WT-error minus its mean mutant-error.
    # Synthesize it directly by adding a per-protein constant to the MUTANT rows only (row 0 = WT
    # reference stays clean), so it does NOT cancel in ddG = pred - pred[0].
    b_vals = rng.normal(0.0, 3.0, size=len(names))  # target std(b) ~= 3.0
    for name, b in zip(names, b_vals):
        n = 40
        true = rng.normal(0.0, 1.5, size=n)
        true[0] = 0.0  # WT row: ddG reference is 0
        noise = rng.normal(0.0, 0.35, size=n)  # keeps within-protein PCC high but < 1
        pred = 1.0 * true + noise               # slope a=1, clean
        pred[0] = 0.0 + noise[0]                 # WT prediction: no offset on the reference row
        pred[1:] = pred[1:] + b                  # offset lands on mutants -> survives ddG
        for t, p in zip(true, pred):
            rows.append({'protein': name, 'deltaG': float(t), 'pred_deltaG': float(p)})
    df = pd.DataFrame(rows)

    res = analyze(df, label='SELFTEST')
    print_block(res)
    des_mask = df['protein'].astype(str).str.contains(DESIGNED_RE, regex=True)
    n_des = df.loc[des_mask, 'protein'].nunique()

    checks = {
        'slope_median in [0.7,1.3]': 0.7 <= res['slope_median'] <= 1.3,
        'std_b substantial (>1.5)': res['std_b'] > 1.5,
        'raw_pooled_depressed(<0.8)': res['pooled_ddG_PCC'] < 0.8,
        'offset_removed_recovered(>0.9)': res['pooled_offset_removed_PCC'] > 0.9,
        'offset_removed beats raw': res['pooled_offset_removed_PCC'] > res['pooled_ddG_PCC'],
        'designed_regex_found_5': n_des == 5,
    }
    print('---- selftest assertions ----')
    ok_all = True
    for k, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {k}')
        ok_all = ok_all and ok
    if not ok_all:
        raise SystemExit('[calib_diag] SELFTEST FAILED — instrument is not trustworthy.')
    print('[calib_diag] SELFTEST PASSED — offset-removal math, per-protein fit, WT-referenced '
          'ddG, and designed-fold detector all correct.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None,
                    help='per-mutation CSV: protein,deltaG,pred_deltaG (row0/group=WT)')
    ap.add_argument('--json', default=None, help='optional path to dump metrics as JSON')
    ap.add_argument('--selftest', action='store_true',
                    help='run the synthetic ground-truth self-test (no data/GPU needed) and exit')
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.csv is None:
        raise SystemExit('[calib_diag] provide --csv <file> or --selftest')

    df = pd.read_csv(args.csv)
    for col in ('protein', 'deltaG', 'pred_deltaG'):
        if col not in df.columns:
            raise SystemExit(f'[calib_diag] missing column {col!r}; got {list(df.columns)}')

    all_res = analyze(df, label='ALL')
    print_block(all_res)

    # designed-fold subset
    mask = df['protein'].astype(str).str.contains(DESIGNED_RE, regex=True)
    n_designed = df.loc[mask, 'protein'].nunique()
    if n_designed >= 1:
        des_res = analyze(df.loc[mask].copy(), label='DESIGNED')
        print_block(des_res)
    else:
        des_res = None
        print('[calib_diag] no designed-fold proteins matched; skipping DESIGNED block.')

    # end-to-end validation verdict on the affine oracle
    if all_res is not None:
        v = all_res['pooled_affine_removed_PCC']
        if not np.isnan(v):
            ok = 0.77 <= v <= 0.81
            verdict = 'IN RANGE (oracle validated)' if ok else 'OUT OF RANGE'
            print(f'[calib_diag] affine-oracle check: {v:.4f} -> {verdict} [target 0.77-0.81]')

    if args.json:
        out = {'ALL': all_res, 'DESIGNED': des_res}
        with open(args.json, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'[calib_diag] wrote metrics JSON to {args.json}')


if __name__ == '__main__':
    main()
