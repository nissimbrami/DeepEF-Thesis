# -*- coding: utf-8 -*-
"""Assemble results/OFFSET_CORRECTOR.md from the three result JSONs."""
import json
import os
import sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else '/home/nissimb/DeepPEF'
R = os.path.join(REPO, 'results')
main = json.load(open(os.path.join(R, 'offset_corrector.json')))
perm = None
sens = None
pp = os.path.join(R, 'offset_corrector_perm.json')
sp = os.path.join(R, 'offset_corrector_sensitivity.json')
op = os.path.join(R, 'offset_corrector_outliers.json')
outl = json.load(open(op)) if os.path.exists(op) else None
fap = os.path.join(R, 'offset_corrector_fixedalpha.json')
fixa = json.load(open(fap)) if os.path.exists(fap) else None
odp = os.path.join(R, 'offset_corrector_outlier_diag.json')
odiag = json.load(open(odp)) if os.path.exists(odp) else None
if os.path.exists(pp):
    perm = json.load(open(pp))
if os.path.exists(sp):
    sens = json.load(open(sp))

agg = main['aggregate']
rows = main['all_csvs']
prim = main['primary']
L = []
A = L.append

A('# Predicting the per-protein offset b_p: the corrector is a NEGATIVE result')
A('')
A('**Bottom line.** The 0.71 offset-removed ceiling is an ORACLE and it stays an oracle. '
  'Structure-only features do NOT predict b_p on held-out proteins: the leave-one-protein-out '
  'held-out R^2 of b_p itself is **negative on 8 of 10 evaluation CSVs** (mean R^2 = '
  '%+.3f), meaning the feature model predicts b_p *worse than the training mean does*. '
  'The pooled-PCC gain it produces is +%.4f on average -- about %.0f%% of the oracle gain of '
  '+%.4f.%s We report this as a '
  'negative result rather than dressing it up.'
  % (agg['r2_lopo_ridge']['mean'],
     np.mean([r['pooled_lopo_ridge'] - r['pooled_raw'] for r in rows]),
     100.0 * np.mean([r['pooled_lopo_ridge'] - r['pooled_raw'] for r in rows])
     / np.mean([r['pooled_oracle'] - r['pooled_raw'] for r in rows]),
     np.mean([r['pooled_oracle'] - r['pooled_raw'] for r in rows]),
     (' A matched permutation null (see below) puts the observed gain at p >= 0.05 on '
      '%d of %d CSVs.' % (sum(1 for v in perm.values() if v['p_value'] >= 0.05), len(perm)))
     if perm else ''))
A('')

if sens:
    _u = sens['univariate_primary']
    _np = sum(1 for x in _u if x['r2'] > 0)
    A('The single sharpest statement of the result: of the %d structural features, '
      '**%d achieve positive held-out R^2** when used alone to predict b_p. Not one '
      'structural covariate predicts the offset out of sample, and no combination of them '
      'does either.' % (len(_u), _np))
    A('')

A('## The three numbers, side by side')
A('')
A('Mean over all 10 evaluation CSVs (28 proteins, 28,314 mutation pairs each):')
A('')
A('| quantity | pooled ddG PCC | gain vs raw | is it a method? |')
A('|---|---|---|---|')
A('| raw (no correction) | %.4f | -- | yes (baseline) |'
  % agg['pooled_raw']['mean'])
A('| mean-offset baseline (no features) | %.4f | %+.4f | yes, trivially |'
  % (agg['pooled_mean_baseline']['mean'],
     agg['pooled_mean_baseline']['mean'] - agg['pooled_raw']['mean']))
A('| **LOPO-predicted offset removed** | **%.4f** | **%+.4f** | **yes -- this is the method** |'
  % (agg['pooled_lopo_ridge']['mean'],
     agg['pooled_lopo_ridge']['mean'] - agg['pooled_raw']['mean']))
A('| oracle offset removed | %.4f | %+.4f | **NO -- fitted on test labels** |'
  % (agg['pooled_oracle']['mean'],
     agg['pooled_oracle']['mean'] - agg['pooled_raw']['mean']))
A('')
A('The honest reading: the oracle buys +%.4f pooled PCC. A real, held-out corrector buys '
  '+%.4f. **%.0f%% of the oracle gain does not survive contact with held-out prediction.**'
  % (agg['pooled_oracle']['mean'] - agg['pooled_raw']['mean'],
     agg['pooled_lopo_ridge']['mean'] - agg['pooled_raw']['mean'],
     100.0 * (1 - (agg['pooled_lopo_ridge']['mean'] - agg['pooled_raw']['mean'])
              / (agg['pooled_oracle']['mean'] - agg['pooled_raw']['mean']))))
A('')

A('### An important technical point about the "mean baseline"')
A('')
A('Pearson correlation is invariant to adding a constant. Subtracting the *same* number from '
  'every protein therefore changes pooled PCC by **exactly zero** (verified: subtracting '
  '-1.0, 0.0, mean(b_p), or +1.0 all give pooled = 0.590991 on the primary CSV). The '
  'mean-baseline row above differs from raw only because leave-one-out makes the subtracted '
  'mean very slightly protein-dependent, and it comes out marginally *worse* than raw on '
  '10/10 CSVs. **So the null that a per-protein corrector must beat is the RAW number, not '
  'the mean baseline.** Reporting a lift over the mean baseline would overstate the result; '
  'we do not do that.')
A('')

A('## Per-CSV detail')
A('')
A('| eval CSV | raw | mean-base | LOPO ridge | oracle | d(ridge-raw) | held-out R^2 of b_p |')
A('|---|---|---|---|---|---|---|')
for r in rows:
    A('| %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.3f |'
      % (r['eval_csv'], r['pooled_raw'], r['pooled_mean_baseline'],
         r['pooled_lopo_ridge'], r['pooled_oracle'],
         r['pooled_lopo_ridge'] - r['pooled_raw'], r['r2_lopo_ridge']))
A('')
nb = sum(1 for r in rows if r['pooled_lopo_ridge'] > r['pooled_raw'])
A('Ridge beats raw on %d/10 CSVs; held-out R^2 > 0 on only **%d/10**. On the primary CSV '
  '(`%s`) the corrector *loses*: %.4f vs %.4f raw, with R^2 = %.3f and '
  'corr(b_p, predicted b_p) = %.4f -- i.e. essentially no relationship at all.'
  % (nb, main['n_csvs_ridge_r2_positive'], prim['eval_csv'],
     prim['pooled_lopo_ridge'], prim['pooled_raw'], prim['r2_lopo_ridge'],
     prim['pearson_bp_vs_pred']))
A('')

A('## Why it fails: the predictions miss exactly the proteins that matter')
A('')
A('The oracle gain is driven by the two extreme-offset proteins. The LOPO model predicts '
  'both at essentially zero, and gets the sign wrong on several mid-range proteins:')
A('')
A('| protein | true b_p | ridge prediction | mean prediction |')
A('|---|---|---|---|')
per = sorted(prim['per_protein'], key=lambda x: x['b_p'])
for r in [per[0], per[1], per[2]] + per[len(per) // 2 - 1:len(per) // 2 + 1] + [per[-3], per[-2], per[-1]]:
    A('| %s | %+.4f | %+.4f | %+.4f |'
      % (r['protein'], r['b_p'], r['b_p_hat_ridge'], r['b_p_hat_mean']))
A('')
A('`2K5H` (b_p = %+.4f) and `2KVS` (b_p = %+.4f) are the two proteins whose offsets the '
  'oracle most needs to remove. The ridge predicts %+.4f and %+.4f for them -- both '
  'effectively zero, and 2KVS with the wrong sign.'
  % (per[0]['b_p'], per[-1]['b_p'], per[0]['b_p_hat_ridge'], per[-1]['b_p_hat_ridge']))
A('')
if outl:
    o = outl[prim['eval_csv']]
    t1, t2 = o['top'][0], o['top'][1]
    fr = [v['top'][1]['frac_of_oracle_gain'] for v in outl.values()]
    A('This is decisive, and it is measurable. On the primary CSV, applying the oracle '
      'correction to **only the two largest-|b_p| proteins** (%s) and leaving the other 26 '
      'uncorrected already recovers **%.0f%% of the entire oracle gain** (%.4f -> %.4f of '
      'the full %.4f -> %.4f). Correcting all 26 *others* and leaving those two alone '
      'recovers only %.0f%% (-> %.4f).'
      % (', '.join('`%s`' % x for x in t2['proteins']),
         100 * t2['frac_of_oracle_gain'], o['raw'], t2['pooled'], o['raw'], o['oracle'],
         100 * o['all_except_top2']['frac_of_oracle_gain'],
         o['all_except_top2']['pooled']))
    A('')
    A('This replicates across all 10 CSVs: the top-2 |b_p| proteins carry a mean of '
      '**%.0f%% of the oracle gain** (range %.0f-%.0f%%), with `2K5H` in the top pair every '
      'time.' % (100 * np.mean(fr), 100 * min(fr), 100 * max(fr)))
    A('')
    A('So the "0.71 ceiling" is not a broad, systematic per-protein miscalibration that a '
      'feature model could learn -- it is dominated by two outlier proteins out of 28. A '
      'smooth structural regression cannot capture them, which is exactly what the negative '
      'held-out R^2 reports. **The oracle gain is the outliers.**')
    A('')

A('### The outliers are not structural outliers -- and one is not an offset at all')
A('')
if odiag:
    A('Checked directly: neither dominant protein is an outlier in feature space. The '
      'largest absolute z-score across all %d features is only %+.2f for `2K5H` (`%s`) and '
      '%+.2f for `2KVS` (`%s`). Both sit inside the structural distribution, so there is no '
      'structural signature for a regression to latch onto. That is the mechanism behind '
      'the negative R^2, not a modelling mistake.'
      % (main['protocol']['n_features'],
         odiag['2K5H']['top_z'][0][1], odiag['2K5H']['top_z'][0][0],
         odiag['2KVS']['top_z'][0][1], odiag['2KVS']['top_z'][0][0]))
A('')
if odiag:
    A('More importantly, `2KVS` is not really a "calibration offset" case at all: its slope '
      'is a_p = %.3f against a median of %.4f, and its per-protein PCC is %.3f against a '
      'median of %.2f. The model essentially fails on that protein outright; the large '
      'fitted b_p is absorbing that failure. Subtracting an offset is the wrong repair for '
      'it, and no offset predictor -- however good -- would be the right fix. (`2K5H` is '
      'the opposite case: a_p = %.3f and PCC = %.3f, a genuinely well-ranked protein '
      'carrying a real offset.)'
      % (odiag['2KVS']['a_p'], odiag['medians']['a_p'], odiag['2KVS']['pcc'],
         odiag['medians']['pcc'], odiag['2K5H']['a_p'], odiag['2K5H']['pcc']))
A('')
if odiag:
    _v = odiag['n_vs_abs_bp']
    A('|b_p| is also mildly related to the number of mutations measured per protein '
      '(r = %.3f, p = %.3f, n = %d), so part of the spread is estimation noise in b_p '
      'itself rather than a physical property waiting to be predicted.'
      % (_v['pearson_r'], _v['p'], _v['n']))
A('')

A('### The small positive gain is an artefact of the alpha search, not skill')
A('')
A('The main table selects the ridge penalty by an inner leave-one-out on each training '
  'fold. Repeating the whole experiment with the penalty **fixed** at alpha=30 (the value '
  'the permutation null uses, so the two are exactly comparable) removes that extra '
  'degree of freedom, and the corrector gets *worse than useless*:')
A('')
A('| | mean d(ridge-raw) | mean held-out R^2 | R^2 > 0 |')
A('|---|---|---|---|')
A('| inner-CV alpha (main table) | %+.4f | %+.3f | %d/10 |'
  % (np.mean([r['pooled_lopo_ridge'] - r['pooled_raw'] for r in rows]),
     agg['r2_lopo_ridge']['mean'], main['n_csvs_ridge_r2_positive']))
if fixa:
    A('| fixed alpha = %g | **%+.4f** | **%+.3f** | **%d/10** |'
      % (fixa['alpha'], fixa['mean_delta'], fixa['mean_r2'], fixa['n_r2_pos']))
A('')
A('With the penalty fixed, the mean pooled-PCC change is **negative**: subtracting the '
  'predicted offsets makes the pooled correlation slightly worse on average. The small '
  'positive number in the main table is variance introduced by the alpha search, not '
  'evidence of a learned structure->offset relationship.')
A('')

if perm:
    A('## Permutation null: the small gain is not significant')
    A('')
    A('Shuffling the b_p labels against the feature rows destroys any true feature->b_p '
      'mapping while preserving the b_p distribution and the entire LOPO machinery. '
      'Real model and null both use a fixed alpha=30 so the comparison is exactly matched. '
      '%d permutations per CSV. (The LOPO ridge here is a closed-form solve verified '
      'identical to the sklearn path to 2.5e-16.)'
      % list(perm.values())[0]['n_perm'])
    A('')
    A('| eval CSV | d(ridge-raw) | null mean d | null 95th pct | p |')
    A('|---|---|---|---|---|')
    for k, v in perm.items():
        A('| %s | %+.4f | %+.4f | %+.4f | %.3f |'
          % (k, v['delta_vs_raw'], v['perm_delta_mean'],
             v['perm_delta_p95'], v['p_value']))
    ps = [v['p_value'] for v in perm.values()]
    A('')
    A('CSVs with p < 0.05: **%d/%d**. %s'
      % (sum(1 for x in ps if x < 0.05), len(ps),
         'The observed gains sit inside the permutation null: the corrector is '
         'indistinguishable from shuffled labels.'
         if sum(1 for x in ps if x < 0.05) <= 2 else
         'Some CSVs reach nominal significance; with 10 CSVs and no multiplicity '
         'correction this is weak evidence at best.'))
    A('')

if sens:
    A('## Sensitivity analyses')
    A('')
    sub = sens['plddt_subset']
    A('### pLDDT (24 of 28 proteins)')
    A('')
    A('`plddt.csv` covers only 24 of the 28 test proteins. The 4 missing ones (%s) are all '
      'de-novo designed, and their AlphaFold PDB B-factor columns are all exactly 0.00, so '
      'there is no pLDDT to recover -- imputing would fabricate data. pLDDT is therefore '
      'excluded from the main 28-protein model and tested here on the 24-protein subset.'
      % ', '.join('`%s`' % m for m in sub['missing']))
    A('')
    A('| eval CSV | raw | LOPO without pLDDT | LOPO with pLDDT | oracle |')
    A('|---|---|---|---|---|')
    for a, b in zip(sub['without'], sub['with_plddt']):
        A('| %s | %.4f | %.4f (R2 %+.3f) | %.4f (R2 %+.3f) | %.4f |'
          % (a['csv'], a['raw'], a['ridge'], a['r2'], b['ridge'], b['r2'], a['oracle']))
    d = np.mean([b['r2'] - a['r2'] for a, b in zip(sub['without'], sub['with_plddt'])])
    A('')
    A('Mean change in held-out R^2 from adding pLDDT: **%+.3f**. Adding pLDDT does not '
      'rescue the corrector.' % d)
    A('')
    A('### Smaller feature sets (n=28 demands ruthless parsimony)')
    A('')
    A('| feature set | features | mean raw | mean LOPO | mean held-out R^2 | beats raw |')
    A('|---|---|---|---|---|---|')
    for k, v in sens['small_sets'].items():
        rr = v['rows']
        A('| %s | %s | %.4f | %.4f | %+.3f | %d/10 |'
          % (k, ', '.join('`%s`' % f for f in v['features']),
             np.mean([x['raw'] for x in rr]), np.mean([x['ridge'] for x in rr]),
             np.mean([x['r2'] for x in rr]),
             sum(1 for x in rr if x['ridge'] > x['raw'])))
    A('')
    A('### Univariate LOPO on the primary CSV')
    A('')
    A('Best single features by held-out R^2 (all should be read against the fact that R^2 <= 0 '
      'means "worse than predicting the training mean"):')
    A('')
    A('| feature | held-out R^2 | pooled PCC after correction |')
    A('|---|---|---|')
    for u in sens['univariate_primary'][:8]:
        A('| `%s` | %+.4f | %.4f |' % (u['feature'], u['r2'], u['pooled']))
    A('')
    npos = sum(1 for u in sens['univariate_primary'] if u['r2'] > 0)
    A('%d of %d single features achieve positive held-out R^2.'
      % (npos, len(sens['univariate_primary'])))
    A('')

A('## Method')
A('')
A('- **Target.** `b_p` = the per-protein ddG-space intercept, `np.polyfit(ddg_true, '
  'ddg_pred, 1)[1]`, with row 0 of each protein group as WT -- identical to '
  '`calib_diag.per_protein_fit`, reused via `scripts/catalogue_vs_bp.recover_bp`. '
  'Note this is the **ddG-space** intercept (std %.4f), not `b_p_wt_error`, the dG-space '
  'WT error (std 1.6030, matching the recorded std(b_p)=1.5741). The ddG intercept is the '
  'quantity the oracle actually subtracts, so it is the quantity a corrector must predict.'
  % prim['bp_std'])
A('- **Features (%d).** Computed from the AlphaFold models of the 28 test proteins via '
  '`scripts/catalogue_vs_bp.structural_features` (imported, not duplicated) plus radius of '
  'gyration, contact order and CA-geometry secondary-structure fractions: %s.'
  % (main['protocol']['n_features'],
     ', '.join('`%s`' % f for f in main['protocol']['features'])))
A('- **Dropped (%d).** %s -- zero variance or unavailable. As recorded, all 28 test proteins '
  'are single-chain, ligand-free, metal-free monomers, so chain count, HET count, metal count '
  'and interchain BSA have **zero variance here**: not measurable, not "no effect".'
  % (len(main['protocol']['dropped_features']),
     ', '.join('`%s`' % f for f in main['protocol']['dropped_features'])))
A('- **Protocol.** Leave-one-protein-out ridge. For each held-out protein the standardiser '
  'and the ridge are fit on the other 27 only, and the regularisation strength is chosen by '
  'an **inner** leave-one-out over those 27. The held-out protein\'s own b_p never enters any '
  'fit at any level. Its predicted offset is then subtracted from its ddG predictions and '
  'pooled PCC is recomputed over all pairs.')
A('- **Guard against the signature failure mode** (code runs, reports a number, feature never '
  'read): verified that shuffling the feature matrix changes the predictions '
  '(max|delta| = 0.456) and that zeroing the features changes them (max|delta| = 0.249), '
  'with the zero-feature model collapsing **exactly** onto the LOPO training mean '
  '(max|delta| = 0.00000000). The features are demonstrably read.')
A('- **G4 gate.** `python scripts/gate_g4_cpu.py` -> `baseline ... dG=-0.0030 width=1092`, '
  'ALL PASS. This work is analysis-only and touches no model code.')
A('')

A('## Honest caveats')
A('')
A('1. **n=28 with %d features is a severe overfitting regime** (~1.3 proteins per feature). '
  'Strong ridge regularisation and a nested inner loop were used, and the model still fails '
  'to generalise. With n=28 the standard error on any correlation is large: |r| < 0.374 is '
  'indistinguishable from zero at p=0.05.' % main['protocol']['n_features'])
A('2. **A negative held-out R^2 is the honest headline.** Mean R^2 = %+.3f across CSVs means '
  'the feature model is, on average, worse than a constant. The occasional positive pooled-PCC '
  'delta is not evidence of skill; it is what a noisy near-zero predictor does to a '
  'shift-invariant metric.' % agg['r2_lopo_ridge']['mean'])
A('3. **This does not prove b_p is unpredictable in principle** -- only that these 22 '
  'structure-only covariates, on these 28 small single-domain monomers, do not predict it. '
  'A larger and more diverse protein set, or features derived from the model\'s own internal '
  'state rather than from structure, remain open.')
A('4. **What this costs the thesis.** The offset-removal number (0.70-0.72) must be labelled '
  'in the text as an **oracle upper bound / diagnostic of headroom**, never as achieved '
  'performance. The achievable pooled PCC with a real corrector is %.4f with the alpha '
  'search and %s with the penalty fixed -- either way indistinguishable from the raw %.4f. '
  'The defensible sentence is: "removing a per-protein offset fitted on the test labels '
  'raises pooled PCC to 0.71, but that offset cannot be predicted from structure on '
  'held-out proteins, so 0.71 is an upper bound rather than an achieved result."'
  % (agg['pooled_lopo_ridge']['mean'],
     ('%.4f' % (agg['pooled_raw']['mean'] + fixa['mean_delta'])) if fixa else 'less',
     agg['pooled_raw']['mean']))
A('')
A('## Reproduce')
A('')
A('```bash')
A('python scripts/offset_corrector.py --repo /home/nissimb/DeepPEF   # main table')
A('python scripts/perm_null.py       /home/nissimb/DeepPEF 200       # permutation null')
A('python scripts/sensitivity.py     /home/nissimb/DeepPEF           # pLDDT / small sets')
A('python scripts/outlier_decomp.py  /home/nissimb/DeepPEF           # oracle-gain decomposition')
A('python scripts/make_report.py     /home/nissimb/DeepPEF           # regenerate this file')
A('```')
A('')
A('Artifacts: `results/offset_corrector.json`, `results/offset_corrector_perm.json`, '
  '`results/offset_corrector_sensitivity.json`, `results/offset_corrector_outliers.json`, '
  '`results/offset_corrector_fixedalpha.json`.')

open(os.path.join(R, 'OFFSET_CORRECTOR.md'), 'w').write('\n'.join(L) + '\n')
print('[wrote] results/OFFSET_CORRECTOR.md  (%d lines)' % len(L))
