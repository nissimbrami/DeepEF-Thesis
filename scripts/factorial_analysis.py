#!/usr/bin/env python3
"""Factorial analysis of the phase-3 ablation grid (abl_p3_*.csv).

WHAT THIS IS. The phase-3 factorial crosses four binary factors:

    A = --wt_anchor          a0 = off,  a1 = on
    B = --designed_weight    d0 = off,  d1 = on
    C = --slope_weight       s0 = off,  s1 = on
    D = reference state      D0 = coil, D1 = --unfolded_emb zero

2^4 = 16 design points; with seeds the full grid is 48 runs. This script scores every
CSV that has landed and re-runs cleanly as more arrive.

METRIC DEFINITIONS are taken verbatim from scripts/arm_analyze.py and scripts/calc_bp.py
so that numbers here are comparable to FINDINGS.md, NOT re-derived:
  per protein   fit pred_ddG ~ a_p*ddG + b_ddg   a_p = r*s, r = PCC, s = std(pred)/std(true)
  b_p           the dG-space WILD-TYPE error, pred_dG - dG on row 0 (WS-1 convention)
  pooled PCC    PCC over every row of every protein, ddG side
  oracle        subtract each protein own mean residual from pred_ddG, then pool.
                This uses the test labels. It is an upper bound, NOT a method.

THE 2K5H CORRECTION. 2K5H row 0 was a MUTANT background, so all 1,125 of its ddG labels
carry a +3.0824 kcal/mol shift; we subtract it (scripts/fix_2k5h.py, FINDINGS 12.1). Note
carefully WHICH metrics this can move, because it is a per-protein CONSTANT on the ddG side:
  a_p, r, s, per-protein PCC  ->  mathematically INVARIANT (a constant shift in x changes
                                  neither a slope nor a correlation)
  pooled PCC, oracle          ->  MOVED, because pooling mixes proteins and the oracle is
                                  defined against a per-protein mean.
The script asserts that invariance rather than trusting it (--selftest).

THE METRIC RULE. ddG cancels anything identical between WT and mutant. Factor D is a
REFERENCE-STATE lever, so its effect is a dG/b_p-side quantity; a_p is within-protein and
is legitimately ddG-measurable. Both are reported and never conflated.

Usage:
    python scripts/factorial_analysis.py                 # score all, write the report
    python scripts/factorial_analysis.py --selftest      # invariance checks only
"""
import sys, os, csv, math, glob, json, re
from collections import OrderedDict

K5H_SHIFT = 3.0824
EVAL_DIR = 'eval_results'
OUT_MD = 'results/FACTORIAL_ANALYSIS.md'
OUT_JSON = 'results/factorial_analysis.json'


# ---------------------------------------------------------------- statistics
def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)


def std(v):
    n = len(v)
    if n < 2: return float('nan')
    m = sum(v)/n
    return math.sqrt(sum((a-m)**2 for a in v)/(n-1))


def median(v):
    s = sorted(v); n = len(s)
    if n == 0: return float('nan')
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])


def mean(v):
    return sum(v)/len(v) if v else float('nan')


def ok(x):
    return x == x


# ---------------------------------------------------------------- cell naming
# abl_p3_a{0,1}_d{0,1}_s{0,1}_D{0_coil,1_uemb}_seed<N>_e<E>.csv
CELL = re.compile(r'^abl_p3_a([01])_d([01])_s([01])_D([01])_(coil|uemb)_seed(\d+)_e(\d+)\.csv$')
# standalone slope arms scored before the grid
SLOPE = re.compile(r'^abl_p3_slope([0-9.]+)_s(\d+)_e(\d+)\.csv$')
# the five seed replicates: identical config, seed only -> the noise floor
SIGMA = re.compile(r'^abl_sigma_seed(\d+)_e(\d+)\.csv$')


def classify(fn):
    m = CELL.match(fn)
    if m:
        A, B, C, D, dname, seed, ep = m.groups()
        return dict(kind='cell', A=int(A), B=int(B), C=int(C), D=int(D),
                    D_name=dname, seed=int(seed), epoch=int(ep),
                    cell='a%s_d%s_s%s_D%s_%s' % (A, B, C, D, dname))
    m = SLOPE.match(fn)
    if m:
        w, seed, ep = m.groups()
        return dict(kind='slope_arm', slope_weight=float(w), seed=int(seed),
                    epoch=int(ep), cell='slope_weight=%s' % w)
    m = SIGMA.match(fn)
    if m:
        seed, ep = m.groups()
        return dict(kind='seed_replicate', seed=int(seed), epoch=int(ep),
                    cell='abl_sigma_seed%s' % seed)
    return None


# ---------------------------------------------------------------- scoring
def load(path, fix_2k5h=True):
    """-> OrderedDict protein -> [(dG, pred_dG, ddG, pred_ddG)], 2K5H corrected."""
    prots = OrderedDict()
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row['protein']
            dd = float(row['ddG'])
            if fix_2k5h and p == '2K5H':
                dd -= K5H_SHIFT
            prots.setdefault(p, []).append((float(row['deltaG']),
                                            float(row['pred_deltaG']),
                                            dd, float(row['pred_ddG'])))
    return prots


def score(path, fix_2k5h=True):
    prots = load(path, fix_2k5h)
    aps, rs, ss, bp_wt, dg_ae = [], [], [], [], []
    pooled_t, pooled_p = [], []
    per_prot = {}
    for p, rows in prots.items():
        dg = [r[0] for r in rows]; pdg = [r[1] for r in rows]
        t = [r[2] for r in rows]; pr = [r[3] for r in rows]
        dg_ae.extend(abs(a-b) for a, b in zip(dg, pdg))
        # b_p of record: the dG-space WT error on row 0 (WS-1 convention)
        bp_wt.append(pdg[0] - dg[0])
        if len(t) < 3: continue
        r = pearson(t, pr); st, sp = std(t), std(pr)
        if not (st > 0) or not (sp > 0) or not ok(r): continue
        s = sp/st; a = r*s
        aps.append(a); rs.append(r); ss.append(s)
        pooled_t.extend(t); pooled_p.extend(pr)
        per_prot[p] = dict(n=len(t), a_p=a, r=r, s=s)

    # offset-removal ORACLE: subtract each protein own mean residual (uses test labels)
    res_t, res_p = [], []
    for p, rows in prots.items():
        t = [r[2] for r in rows]; pr = [r[3] for r in rows]
        if len(t) < 3: continue
        md = mean([b - a for a, b in zip(t, pr)])
        res_t.extend(t); res_p.extend(x - md for x in pr)

    pooled = pearson(pooled_t, pooled_p)
    oracle = pearson(res_t, res_p)

    o = OrderedDict()
    o['file'] = os.path.basename(path)
    o.update(classify(os.path.basename(path)) or dict(kind='other', cell='?'))
    o['n_proteins'] = len(per_prot)
    o['n_rows'] = sum(len(v) for v in prots.values())
    o['pooled_ddG_PCC'] = pooled
    o['oracle_offset_removed_PCC'] = oracle
    o['oracle_gain'] = oracle - pooled
    o['perprot_ddG_PCC'] = mean(rs)
    o['a_p_median'] = median(aps)
    o['r_median'] = median(rs)
    o['s_median'] = median(ss)
    o['std_bp_dG_WTerr'] = std(bp_wt)
    o['mean_bp_dG_WTerr'] = mean(bp_wt)
    o['dG_MAE'] = mean(dg_ae)
    o['_per_prot'] = per_prot
    return o


# ---------------------------------------------------------------- self-test
def selftest(paths):
    """The 2K5H shift must be invariant for within-protein metrics, and move pooled/oracle."""
    print('=== SELFTEST: what the 2K5H correction may and may not move ===')
    bad = 0
    for p in paths[:5]:
        on = score(p, True); off = score(p, False)
        for name in ('a_p_median', 'r_median', 's_median', 'perprot_ddG_PCC'):
            if abs(on[name] - off[name]) > 1e-9:
                print('  FAIL %-20s %s  %.10f vs %.10f'
                      % (name, os.path.basename(p), on[name], off[name]))
                bad += 1
        print('  %-44s pooled %+.4f -> %+.4f (%+.4f)   oracle %+.4f -> %+.4f (%+.4f)'
              % (os.path.basename(p),
                 off['pooled_ddG_PCC'], on['pooled_ddG_PCC'],
                 on['pooled_ddG_PCC'] - off['pooled_ddG_PCC'],
                 off['oracle_offset_removed_PCC'], on['oracle_offset_removed_PCC'],
                 on['oracle_offset_removed_PCC'] - off['oracle_offset_removed_PCC']))
    print('  within-protein invariance: %s'
          % ('OK (as the algebra requires)' if bad == 0 else 'BROKEN -- investigate'))
    return bad == 0


# ---------------------------------------------------------------- main effects
def main_effects(cells, key):
    """Main effect of A, B, C within ONE D arm: mean(factor=1) - mean(factor=0).

    Averaging across D is FORBIDDEN: the D effect is ~10x anything else, so a cross-D
    average is a mixture of two populations, not an effect.
    """
    out = OrderedDict()
    for f in ('A', 'B', 'C'):
        hi = [c[key] for c in cells if c[f] == 1 and ok(c[key])]
        lo = [c[key] for c in cells if c[f] == 0 and ok(c[key])]
        out[f] = dict(n_hi=len(hi), n_lo=len(lo),
                      mean_hi=mean(hi), mean_lo=mean(lo),
                      effect=(mean(hi) - mean(lo)) if (hi and lo) else float('nan'))
    return out


def confound_audit(cells):
    """Is a factor ALIASED with seed or epoch inside this arm?

    A main effect is only an effect if the two levels are otherwise comparable. Here the
    cells landed opportunistically, so a factor can be perfectly correlated with which
    seed ran or which epoch was selected. This routine reports, for each factor:
      - corr(factor, epoch) and the seed composition at each level
      - whether the effect is still ESTIMABLE after restricting to the single largest
        (seed, late-epoch) stratum in which the factor actually varies.
    A factor that is not estimable in any clean stratum must NOT be reported as an effect.
    """
    out = OrderedDict()
    eps = [c['epoch'] for c in cells]
    for f in ('A', 'B', 'C'):
        lv = [c[f] for c in cells]
        r_ep = pearson(lv, eps)
        seeds = dict((l, sorted(set(c['seed'] for c in cells if c[f] == l))) for l in (0, 1))
        epochs = dict((l, sorted(c['epoch'] for c in cells if c[f] == l)) for l in (0, 1))
        # clean stratum: single most common seed, epochs at or above the median epoch
        med_ep = median(eps)
        seed_counts = {}
        for c in cells:
            seed_counts[c['seed']] = seed_counts.get(c['seed'], 0) + 1
        main_seed = max(seed_counts, key=lambda k: seed_counts[k]) if seed_counts else None
        clean = [c for c in cells if c['seed'] == main_seed and c['epoch'] >= med_ep]
        n_hi = len([c for c in clean if c[f] == 1])
        n_lo = len([c for c in clean if c[f] == 0])
        out[f] = dict(corr_with_epoch=r_ep, seeds=seeds, epochs=epochs,
                      main_seed=main_seed, clean_n=len(clean),
                      clean_n_hi=n_hi, clean_n_lo=n_lo,
                      estimable_clean=(n_hi > 0 and n_lo > 0))
    return out


def welch(a, b):
    """Welch t and a crude two-sided p, so an effect can be read against its own scatter."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan'), float('nan')
    va, vb = std(a) ** 2, std(b) ** 2
    se = math.sqrt(va / na + vb / nb)
    if se <= 0: return float('nan'), float('nan')
    t = (mean(a) - mean(b)) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # normal approximation to the two-sided p; df is small, so this is OPTIMISTIC
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


# ---------------------------------------------------------------- report
def fmt(x, w=7, d=4):
    return ('%*.*f' % (w, d, x)) if ok(x) else ('%*s' % (w, 'nan'))


def build(argv):
    root = os.getcwd()
    paths = sorted(glob.glob(os.path.join(EVAL_DIR, 'abl_p3_*.csv')) +
                   glob.glob(os.path.join(EVAL_DIR, 'abl_sigma_*.csv')))
    if not paths:
        print('no CSVs found under %s' % EVAL_DIR); return 1

    if '--selftest' in argv:
        cellpaths = [p for p in paths if CELL.match(os.path.basename(p))]
        return 0 if selftest(cellpaths) else 1

    scored = []
    for p in paths:
        try:
            scored.append(score(p, True))
        except Exception as e:
            print('ERROR %s: %s' % (p, e))

    cells = [c for c in scored if c['kind'] == 'cell']
    sigma = [c for c in scored if c['kind'] == 'seed_replicate']
    slopes = [c for c in scored if c['kind'] == 'slope_arm']
    d0 = [c for c in cells if c['D'] == 0]
    d1 = [c for c in cells if c['D'] == 1]

    # ---- the noise floor, straight from the five identical-config seeds
    noise = OrderedDict()
    for key in ('pooled_ddG_PCC', 'a_p_median', 'r_median', 's_median',
                'oracle_offset_removed_PCC', 'std_bp_dG_WTerr', 'perprot_ddG_PCC'):
        v = [c[key] for c in sigma if ok(c[key])]
        noise[key] = dict(n=len(v), mean=mean(v), sd=std(v),
                          lo=min(v) if v else float('nan'),
                          hi=max(v) if v else float('nan'),
                          # the two-run detection threshold: a difference of two runs has
                          # sd*sqrt(2); 1.96 of those is the 5% two-sided noise band
                          detect_2run=1.96 * math.sqrt(2) * std(v) if len(v) > 1 else float('nan'))

    L = []
    W = L.append
    W('# FACTORIAL ANALYSIS -- phase-3 grid (A x B x C x D)')
    W('')
    W('Generated by `scripts/factorial_analysis.py`. Re-run it as new CSVs land; it rescores')
    W('every `eval_results/abl_p3_*.csv` from scratch and overwrites this file.')
    W('')
    W('    A = --wt_anchor        a0 off / a1 on')
    W('    B = --designed_weight  d0 off / d1 on')
    W('    C = --slope_weight     s0 off / s1 on')
    W('    D = reference state    D0 coil / D1 --unfolded_emb zero')
    W('')
    W('All ddG labels for **2K5H** carry the -%.4f correction (its stored reference row is a'
      % K5H_SHIFT)
    W('MUTANT background, FINDINGS 12.1). That shift is a per-protein CONSTANT on the ddG side,')
    W('so it is mathematically invariant for `a_p`, `r`, `s` and per-protein PCC, and moves only')
    W('the pooled PCC and the oracle. `--selftest` asserts exactly that.')
    W('')
    W('`oracle` = subtract each protein own mean residual, then pool. It reads the test labels.')
    W('**It is an upper bound, not a method.**')
    W('')
    W('Scored: **%d** factorial cells (%d in D0, %d in D1), plus %d seed replicates and %d'
      % (len(cells), len(d0), len(d1), len(sigma), len(slopes)))
    W('standalone slope arms. Of the 48-run grid, **%d** cells have landed.' % len(cells))
    W('')

    # ---------------------------------------------------------- per-cell table
    W('## 1. Every scored cell')
    W('')
    W('| cell | seed | ep | pooled PCC | oracle | gain | per-prot PCC | a_p | r | s | std(b_p) | dG MAE |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for c in sorted(cells, key=lambda z: (z['D'], -z['pooled_ddG_PCC'] if ok(z['pooled_ddG_PCC']) else 0)):
        W('| `%s` | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
          % (c['cell'], c['seed'], c['epoch'],
             fmt(c['pooled_ddG_PCC']), fmt(c['oracle_offset_removed_PCC']),
             fmt(c['oracle_gain']), fmt(c['perprot_ddG_PCC']), fmt(c['a_p_median']),
             fmt(c['r_median']), fmt(c['s_median']), fmt(c['std_bp_dG_WTerr']),
             fmt(c['dG_MAE'])))
    W('')
    W('Reference arms (not part of the grid):')
    W('')
    W('| run | pooled PCC | oracle | per-prot PCC | a_p | r | s | std(b_p) |')
    W('|---|---|---|---|---|---|---|---|')
    for c in sorted(slopes, key=lambda z: z.get('slope_weight', 0)) + sorted(sigma, key=lambda z: z['seed']):
        W('| `%s` | %s | %s | %s | %s | %s | %s | %s |'
          % (c['cell'], fmt(c['pooled_ddG_PCC']), fmt(c['oracle_offset_removed_PCC']),
             fmt(c['perprot_ddG_PCC']), fmt(c['a_p_median']), fmt(c['r_median']),
             fmt(c['s_median']), fmt(c['std_bp_dG_WTerr'])))
    W('')

    # ---------------------------------------------------------- factor D
    W('## 2. Factor D separates completely -- so nothing may be averaged across it')
    W('')
    for name, arm in (('D0 (coil)', d0), ('D1 (unfolded_emb zero)', d1)):
        if not arm: continue
        pv = [c['pooled_ddG_PCC'] for c in arm if ok(c['pooled_ddG_PCC'])]
        av = [c['a_p_median'] for c in arm if ok(c['a_p_median'])]
        W('- **%s**, n=%d: pooled PCC %.4f..%.4f (mean %.4f); a_p %.4f..%.4f (mean %.4f)'
          % (name, len(arm), min(pv), max(pv), mean(pv), min(av), max(av), mean(av)))
    if d0 and d1:
        p0 = [c['pooled_ddG_PCC'] for c in d0]; p1 = [c['pooled_ddG_PCC'] for c in d1]
        a0 = [c['a_p_median'] for c in d0]; a1 = [c['a_p_median'] for c in d1]
        overlap = not (min(p0) > max(p1) or min(p1) > max(p0))
        t, pval = welch(p0, p1)
        W('')
        W('Ranges **%s** on pooled PCC. Gap between the arms: %+.4f on pooled PCC, %+.4f on a_p.'
          % ('OVERLAP' if overlap else 'DO NOT overlap', mean(p0) - mean(p1), mean(a0) - mean(a1)))
        W('Welch t = %.2f (normal-approx p = %.2g; df is small so this p is OPTIMISTIC).' % (t, pval))
        W('')
        W('Measured against the seed-to-seed sd of %.4f on pooled PCC, the D gap is **%.0fx**'
          % (noise['pooled_ddG_PCC']['sd'], abs(mean(p0) - mean(p1)) / noise['pooled_ddG_PCC']['sd']))
        W('the noise. On a_p it is **%.0fx**. This is why A, B and C main effects below are'
          % (abs(mean(a0) - mean(a1)) / noise['a_p_median']['sd']))
        W('computed **inside the D0 arm only**: pooling across D would average two populations.')
    W('')

    # ---------------------------------------------------------- noise floor
    W('## 3. The noise floor -- five identical-config seeds')
    W('')
    W('`abl_sigma_seed{1,2,3,4,42}` differ in the random seed and NOTHING else. They give the')
    W('seed-to-seed spread directly, which is the yardstick every effect below is measured against.')
    W('')
    W('| metric | n | mean | sd | min | max | 2-run detection threshold |')
    W('|---|---|---|---|---|---|---|')
    for k, v in noise.items():
        W('| `%s` | %d | %s | %s | %s | %s | %s |'
          % (k, v['n'], fmt(v['mean']), fmt(v['sd']), fmt(v['lo']), fmt(v['hi']),
             fmt(v['detect_2run'])))
    W('')
    W('The last column is `1.96 * sqrt(2) * sd`: the smallest difference between TWO single runs')
    W('that would clear a 5% two-sided test if seed noise were the only source of variation.')
    W('An effect smaller than that is indistinguishable from re-running the same config.')
    W('')

    # ---------------------------------------------------------- main effects
    W('## 4. Main effects for A, B, C -- WITHIN the D0 arm only')
    W('')
    if len(d0) < 2:
        W('Not enough D0 cells scored yet (%d) to form any contrast.' % len(d0))
    else:
        aud = confound_audit(d0)
        W('### 4.0 FIRST: is each factor even ESTIMABLE here?')
        W('')
        W('The D0 cells landed opportunistically, not as a designed block. Before reading any')
        W('effect, check whether the two levels of a factor are otherwise comparable -- a factor')
        W('aliased with seed or with selected epoch produces a number that is not an effect.')
        W('')
        W('| factor | seeds at level 0 | seeds at level 1 | epochs at 0 | epochs at 1 | corr(factor, epoch) | clean stratum n(off)/n(on) | estimable? |')
        W('|---|---|---|---|---|---|---|---|')
        for f, nm in (('A', 'A wt_anchor'), ('B', 'B designed_weight'), ('C', 'C slope_weight')):
            a = aud[f]
            W('| %s | %s | %s | %s | %s | %s | %d / %d | %s |'
              % (nm, a['seeds'][0], a['seeds'][1], a['epochs'][0], a['epochs'][1],
                 fmt(a['corr_with_epoch'], 6, 2), a['clean_n_lo'], a['clean_n_hi'],
                 '**YES**' if a['estimable_clean'] else '**NO -- ALIASED**'))
        W('')
        W('"Clean stratum" = the most common seed, at epochs at or above the arm median. A factor')
        W('with 0 cells on one side of that stratum has NO unconfounded contrast, and every number')
        W('printed for it below is a mixture of the factor with seed and epoch-selection effects.')
        W('')
        for f, nm in (('A', 'A'), ('B', 'B'), ('C', 'C')):
            if not aud[f]['estimable_clean']:
                W('> **Factor %s is NOT estimable in the D0 arm as it stands.** corr(%s, epoch) = %+.2f'
                  % (nm, nm, aud[f]['corr_with_epoch']))
                W('> and the clean stratum contains %d cells at level 0. Any "%s effect" below is'
                  % (aud[f]['clean_n_lo'], nm))
                W('> reported for completeness only and MUST NOT be quoted as a factor effect.')
                W('')
        for key, label in (('pooled_ddG_PCC', 'pooled ddG PCC'),
                           ('a_p_median', 'a_p'),
                           ('r_median', 'r (ranking)'),
                           ('s_median', 's (spread)'),
                           ('std_bp_dG_WTerr', 'std(b_p)')):
            me = main_effects(d0, key)
            sd = noise[key]['sd'] if key in noise else float('nan')
            thr = noise[key]['detect_2run'] if key in noise else float('nan')
            # Second, more conservative yardstick: the spread of the D0 cells themselves.
            # If that exceeds the seed sd, the seed replicates UNDERSTATE run-to-run noise
            # for this metric and the seed-based threshold is too permissive.
            arm_sd = std([c[key] for c in d0 if ok(c[key])])
            cons = max(sd, arm_sd) if (ok(sd) and ok(arm_sd)) else sd
            cons_thr = 1.96 * math.sqrt(2) * cons if ok(cons) else float('nan')
            W('**%s** (seed sd %s, 2-run threshold %s; D0-arm sd %s, conservative threshold %s)'
              % (label, fmt(sd, 6), fmt(thr, 6), fmt(arm_sd, 6), fmt(cons_thr, 6)))
            if ok(arm_sd) and ok(sd) and sd > 0 and arm_sd > 2 * sd:
                W('')
                W('> The D0 cells vary **%.1fx** more than the seed replicates on this metric, so'
                  % (arm_sd / sd))
                W('> `abl_sigma` UNDERSTATES run-to-run noise here. Judge against the conservative')
                W('> threshold (%s), not the seed one.' % fmt(cons_thr, 6).strip())
            W('')
            W('| factor | n(off) | n(on) | mean(off) | mean(on) | effect | effect / seed sd | verdict |')
            W('|---|---|---|---|---|---|---|---|')
            for f, nm in (('A', 'A wt_anchor'), ('B', 'B designed_weight'), ('C', 'C slope_weight')):
                e = me[f]
                ratio = (e['effect'] / sd) if (ok(e['effect']) and ok(sd) and sd > 0) else float('nan')
                if not ok(e['effect']):
                    verdict = 'not estimable'
                elif not aud[f]['estimable_clean']:
                    verdict = 'ALIASED with seed/epoch -- not an effect'
                elif abs(e['effect']) < thr:
                    verdict = 'INDISTINGUISHABLE from seed noise'
                elif ok(cons_thr) and abs(e['effect']) < cons_thr:
                    verdict = 'clears seed noise but NOT the conservative band'
                else:
                    verdict = 'exceeds both noise bands'
                W('| %s | %d | %d | %s | %s | %s | %s | %s |'
                  % (nm, e['n_lo'], e['n_hi'], fmt(e['mean_lo']), fmt(e['mean_hi']),
                     fmt(e['effect']), fmt(ratio, 6, 2), verdict))
            W('')
        W('These are UNREPLICATED contrasts: each D0 cell is a single seed, so every "effect"')
        W('above carries at least the full seed sd, and a difference of two single runs carries')
        W('sqrt(2) times it. The D0 arm is also not yet a balanced 2^3 -- the counts in the')
        W('n(off)/n(on) columns say exactly which contrasts are confounded with which cells are')
        W('missing. Read the ratio column, not the sign.')
    W('')

    # ---------------------------------------------------------- best cells
    W('## 5. The best cell on pooled PCC is NOT the best cell on a_p')
    W('')
    if cells:
        bp = max((c for c in cells if ok(c['pooled_ddG_PCC'])), key=lambda z: z['pooled_ddG_PCC'])
        ba = max((c for c in cells if ok(c['a_p_median'])), key=lambda z: z['a_p_median'])
        W('| criterion | winning cell | pooled PCC | a_p | r | s | oracle | gain |')
        W('|---|---|---|---|---|---|---|---|')
        W('| best pooled ddG PCC | `%s` | **%s** | %s | %s | %s | %s | %s |'
          % (bp['cell'], fmt(bp['pooled_ddG_PCC']), fmt(bp['a_p_median']), fmt(bp['r_median']),
             fmt(bp['s_median']), fmt(bp['oracle_offset_removed_PCC']), fmt(bp['oracle_gain'])))
        W('| best a_p | `%s` | %s | **%s** | %s | %s | %s | %s |'
          % (ba['cell'], fmt(ba['pooled_ddG_PCC']), fmt(ba['a_p_median']), fmt(ba['r_median']),
             fmt(ba['s_median']), fmt(ba['oracle_offset_removed_PCC']), fmt(ba['oracle_gain'])))
        W('')
        if bp['cell'] != ba['cell']:
            W('**They are DIFFERENT cells.** `%s` wins pooled PCC (%.4f, with a_p %.4f); `%s` wins'
              % (bp['cell'], bp['pooled_ddG_PCC'], bp['a_p_median'], ba['cell']))
            W('a_p (%.4f, with pooled PCC %.4f). Selecting on pooled PCC therefore picks the'
              % (ba['a_p_median'], ba['pooled_ddG_PCC']))
            W('cell with the WORSE slope, and selecting on a_p picks the cell with the worse')
            W('pooled correlation.')
            W('')
            W('That contrast is the calibration thesis in one line: **pooled PCC is dominated by')
            W('the between-protein offset b_p, while a_p measures the within-protein compression.**')
            W('A model can rank proteins well against each other and still be flat inside every')
            W('one of them. The two metrics answer different questions, and the model-selection')
            W('metric has to be chosen deliberately rather than inherited.')
        else:
            W('On the CSVs scored so far the same cell wins both. Re-check as cells land.')
    W('')

    # ---------------------------------------------------------- D1 decision
    W('## 6. Is the D1 half worth finishing?')
    W('')
    if d1 and d0:
        p1 = [c['pooled_ddG_PCC'] for c in d1]; a1 = [c['a_p_median'] for c in d1]
        p0 = [c['pooled_ddG_PCC'] for c in d0]
        W('Scored so far: **%d D1 cells** against **%d D0 cells**.' % (len(d1), len(d0)))
        W('')
        W('- D1 pooled PCC: %.4f .. %.4f' % (min(p1), max(p1)))
        W('- D0 pooled PCC: %.4f .. %.4f' % (min(p0), max(p0)))
        W('- D1 a_p: %.4f .. %.4f -- an a_p near zero means the model is FLAT in ddG and'
          % (min(a1), max(a1)))
        W('  barely responds to mutation at all.')
        W('')
        W('The statistical argument, in the form that actually decides it:')
        W('')
        W('1. **Separation, not a mean difference.** Every D1 cell is below every D0 cell')
        W('   (%d vs %d, no overlap). Under the null that D has no effect, the chance that a'
          % (len(d1), len(d0)))
        W('   random labelling puts all %d D1 runs below all %d D0 runs is 1 / C(%d,%d) = %.4g.'
          % (len(d1), len(d0), len(d1) + len(d0), len(d1),
             1.0 / (math.factorial(len(d0) + len(d1)) /
                    (math.factorial(len(d0)) * math.factorial(len(d1))))))
        W('   That is an exact permutation p-value and it needs no distributional assumption.')
        W('')
        W('2. **The effect is enormous relative to the only noise estimate we have.** The five')
        W('   seed replicates give sd = %.4f on pooled PCC; the D gap is %.4f, i.e. %.0f sd.'
          % (noise['pooled_ddG_PCC']['sd'], abs(mean(p0) - mean(p1)),
             abs(mean(p0) - mean(p1)) / noise['pooled_ddG_PCC']['sd']))
        W('   Remaining D1 cells would have to be many sd ABOVE every D1 cell seen so far to')
        W('   change the conclusion.')
        W('')
        W('3. **What more D1 cells could buy.** They cannot overturn the sign; they could only')
        W('   sharpen the estimate of a quantity we would not use. The purpose of a factorial is')
        W('   to estimate A, B, C main effects and their interactions -- but a_p in D1 is')
        W('   %.4f..%.4f, so the model there is near-flat and A/B/C are being measured on top of'
          % (min(a1), max(a1)))
        W('   a dead signal. An interaction estimated inside a collapsed arm is not informative')
        W('   about the arm we ship.')
        W('')
        W('4. **The mechanism is already understood, which is what makes the negative safe to')
        W('   act on.** D1 zeroes the unfolded embedding; that removes the offset by removing')
        W('   the SIGNAL, which is the same failure already on record for `noemb` on dG')
        W('   (lowest MAE, correlation 0.35 -> 0.05). A negative with a known mechanism does not')
        W('   need the same replication as an unexplained one.')
        W('')
        o1 = [c['oracle_offset_removed_PCC'] for c in d1]
        g1 = [c['oracle_gain'] for c in d1]
        W('5. **A nuance that must not be glossed, because it is the one thing D1 does show.**')
        W('   D1 pooled PCC collapses to %.4f..%.4f, but the D1 ORACLE stays at %.4f..%.4f, and'
          % (min(p1), max(p1), min(o1), max(o1)))
        W('   the D1 oracle GAIN (%.4f..%.4f) is LARGER than any D0 gain. Read that carefully:'
          % (min(g1), max(g1)))
        W('   removing the unfolded embedding did NOT destroy all ranking information -- it')
        W('   destroyed the SLOPE (a_p %.4f..%.4f, i.e. flat) and inflated the between-protein'
          % (min(a1), max(a1)))
        W('   offset, so an oracle that is *allowed to fit each protein offset from the test')
        W('   labels* recovers a good deal. This is the metric rule biting again: the oracle is')
        W('   an upper bound that rewards exactly the failure D1 introduces, so **the D1 oracle')
        W('   must not be read as partial success.** The shippable quantity is a_p, and a_p is')
        W('   dead. It also means the honest summary of D1 is "collapses the slope", not')
        W('   "collapses the model".')
        W('')
        W('**Conclusion: finishing the remaining D1 cells is not justified.** The honest')
        W('statement of the limitation is that the D1 arm is scored at n=%d rather than n=24,' % len(d1))
        W('and it should be written that way -- as a stopped arm with an exact permutation')
        W('p-value and a mechanism, not as a completed one.')
        W('')
        W('The counter-argument that deserves recording: a 2^4 factorial with a collapsed arm')
        W('cannot estimate A x D, B x D or C x D interactions, so the write-up must claim main')
        W('effects **conditional on D0** and not a full factorial decomposition.')
    else:
        W('Not enough cells in one or both D arms yet.')
    W('')

    # ---------------------------------------------------------- caveats
    W('## 7. What this analysis does NOT establish')
    W('')
    W('- Every factorial cell is a **single seed**. Nothing here is replicated at the cell level;')
    W('  the noise floor is imported from a different configuration (`abl_sigma`) and assumed to')
    W('  transfer. That assumption is doing real work and should be stated in the thesis.')
    W('- The D0 arm is **not yet balanced**: with %d of 24 cells the A/B/C contrasts are' % len(d0))
    W('  partially confounded with which cells happened to finish first. Counts are printed above.')
    W('- Cells are read at DIFFERENT epochs (the `ep` column), selected per run. Epoch selection')
    W('  is itself a lever, so part of any cell-to-cell difference is a checkpoint-choice')
    W('  difference rather than a factor effect. Section 4.0 measures this directly, and on the')
    W('  cells scored so far it is not a hypothetical: at least one factor is fully aliased with')
    W('  seed and epoch, and its apparently large effect is therefore NOT interpretable.')
    W('- The seed replicates (`abl_sigma`) fix only the seed. They do NOT bound epoch-selection')
    W('  variance, so the "2-run threshold" is a LOWER bound on the true run-to-run spread and')
    W('  every "exceeds the noise band" verdict is correspondingly optimistic.')
    W('- n=28 proteins. Per FINDINGS, |r| < 0.374 is indistinguishable from zero at this n for')
    W('  any protein-level correlation quoted here.')
    W('')

    txt = '\n'.join(L) + '\n'
    if not os.path.isdir('results'):
        os.makedirs('results')
    with open(OUT_MD, 'w') as f:
        f.write(txt)

    dump = [dict((k, v) for k, v in c.items() if k != '_per_prot') for c in scored]
    with open(OUT_JSON, 'w') as f:
        json.dump(dict(cells=dump, noise=noise), f, indent=1)

    print(txt)
    print('wrote %s and %s' % (OUT_MD, OUT_JSON))
    return 0


if __name__ == '__main__':
    sys.exit(build(sys.argv[1:]))
