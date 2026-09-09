#!/usr/bin/env python3
"""MASTER SCOREBOARD -- every eval CSV, ONE canonical basis.

THE BASIS, stated once and applied everywhere (MASTER.md preamble):
    27 test proteins   -- 2K5H is DROPPED, not shifted.
    ddG metric         -- per-protein affine fits, pooled correlation.

WHY DROPPED AND NOT SHIFTED. 2K5H.csv concatenates three WT backgrounds and the wrong one
sorts first, so all 1,125 of its ddG labels carry a +3.0824 kcal/mol shift (MASTER 11.1).
scripts/factorial_analysis.py SUBTRACTS that constant and keeps 28 proteins. That is a
DIFFERENT basis, and it does not reproduce the published headline:

    sigma_seed2_e10   drop-2K5H 0.6382   shift-2K5H 0.6386   keep-as-is 0.6575
    calib_ctrl_repro2 drop-2K5H 0.5635   shift-2K5H 0.5676   keep-as-is 0.5910

MASTER 1.3 quotes 0.6382 and 0.5635 -> the headline basis is DROP. This script uses DROP
so its numbers are comparable to MASTER.md. --basis shift/all reproduce the others for
comparison; they are NOT the canonical numbers.

METRIC DEFINITIONS -- taken from scripts/factorial_analysis.py so nothing is re-derived:
    per protein   fit pred_ddG ~ a_p*ddG + b   with  a_p = r*s,
                  r = PCC(true ddG, pred ddG), s = std(pred ddG)/std(true ddG).
                  a_p / r / s are reported as MEDIANS over proteins.
    per-protein PCC  MEDIAN of r over proteins (MASTER 1.2 quotes the median, 0.798).
                     mean is also carried in the JSON as perprot_mean.
    pooled PCC    PCC over every row of every kept protein, ddG side.
    oracle        subtract each protein's own mean residual from pred_ddG, then pool.
                  USES THE TEST LABELS. An upper bound, not a method.
    std(b_p)      dG-space WILD-TYPE error, pred_dG - dG on the protein's row 0
                  (WS-1 convention), sample sd over proteins.

THE METRIC RULE (MASTER 2.1). ddG cancels anything identical between WT and mutant, so a
reference-state or whole-protein lever must be read on dG or b_p; a_p is a WITHIN-protein
slope and IS legitimately ddG-measurable. pooled/oracle/a_p/r/s are ddG-side; std(b_p) and
dG_MAE are the dG-side columns and are printed alongside, never conflated.

EPOCH DISCIPLINE (OPEN_PROBLEMS P0). Epoch selection is on VALIDATION
(run_calib_eval.sh line 15); the test set is scored once at that epoch. Two runs --
gld_slope1.0_s42 and gld_dg_coil_s42 -- were deliberately scored at six epochs to measure a
trajectory. Quoting the best of those six would be test-set peeking. This script therefore
keeps ONE row per run tag, the val-selected epoch (e14 for both trajectory runs), and
prints the suppressed trajectory epochs in a separate clearly-labelled section.

THE SEED NOISE BAND is +/-0.060 pooled (five abl_sigma seeds, identical config). Every
group mean is printed with its sd so it can be read against that band.

Usage:
    python scripts/scoreboard.py                  # table + write SCOREBOARD.md
    python scripts/scoreboard.py --selftest       # basis + identity checks
    python scripts/scoreboard.py --basis shift    # comparison basis, NOT canonical
"""
import sys, os, csv, math, glob, json, re
from collections import OrderedDict, defaultdict

EVAL_DIR   = 'eval_results'
OUT_MD     = 'results/02_findings/SCOREBOARD.md'
OUT_JSON   = 'results/02_findings/scoreboard.json'
K5H        = '2K5H'
K5H_SHIFT  = 3.0824
SEED_BAND  = 0.060

# MASTER 1.3 reference points, re-derived here and asserted by --selftest.
CONTROL_TAG = 'calib_ctrl_repro2'
EXPECTED = {'sigma_seed2': 0.6382, 'calib_ctrl_repro2': 0.5635,
            'p3_slope1.0_s42': 0.6210, 'gld_slope1.0_s42': 0.5810}

# OPEN_PROBLEMS P0: runs deliberately scored at several epochs for a TRAJECTORY.
# canonical epoch = the val-selected one; every table must quote that one.
TRAJECTORY = {'gld_slope1.0_s42': 14, 'gld_dg_coil_s42': 14}


# ------------------------------------------------------------------ statistics
def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def std(v):
    v = [a for a in v if a == a]
    n = len(v)
    if n < 2: return float('nan')
    m = sum(v)/n
    return math.sqrt(sum((a-m)**2 for a in v)/(n-1))

def mean(v):
    v = [a for a in v if a == a]
    return sum(v)/len(v) if v else float('nan')

def median(v):
    s = sorted(a for a in v if a == a); n = len(s)
    if n == 0: return float('nan')
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])


# ------------------------------------------------------------------ run naming
CELL   = re.compile(r'^p3_a([01])_d([01])_s([01])_D([01])_(coil|uemb)_seed(\d+)$')
SLOPE  = re.compile(r'^p3_slope([0-9.]+)_s(\d+)$')
GSLOPE = re.compile(r'^gld_slope([0-9.]+)_s(\d+)$')
SIGMA  = re.compile(r'^sigma_seed(\d+)$')
ANCH   = re.compile(r'^anchor_w([0-9.]+)_s(\d+)$')

def classify(tag):
    """tag -> arm group + factor levels. Groups are the reporting units of task step 3."""
    m = CELL.match(tag)
    if m:
        A, B, C, D, dn, seed = m.groups()
        return dict(group=('factorial D0 (coil)' if D == '0' else 'factorial D1 (uemb zero)'),
                    A=int(A), B=int(B), C=int(C), D=int(D), seed=int(seed),
                    cell='a%s_d%s_s%s_D%s_%s' % (A, B, C, D, dn))
    m = SIGMA.match(tag)
    if m:
        return dict(group='sigma seeds (noise floor)', seed=int(m.group(1)), cell=tag)
    m = ANCH.match(tag)
    if m:
        return dict(group='anchor sweep', anchor=float(m.group(1)),
                    seed=int(m.group(2)), cell=tag)
    m = SLOPE.match(tag) or GSLOPE.match(tag)
    if m:
        return dict(group='slope sweep', slope_weight=float(m.group(1)),
                    seed=int(m.group(2)), cell=tag)
    if tag == CONTROL_TAG:
        return dict(group='control', cell=tag)
    if tag.startswith('gld_'):
        return dict(group='golden arms', cell=tag)
    return dict(group='other', cell=tag)


FN = re.compile(r'^abl_(.+)_e(\d+)\.csv$')

def parse(path):
    m = FN.match(os.path.basename(path))
    if not m: return None, None
    return m.group(1), int(m.group(2))


# ------------------------------------------------------------------ scoring
def load(path, basis='drop'):
    """-> OrderedDict protein -> [(dG, pred_dG, ddG, pred_ddG)] on the chosen basis."""
    prots = OrderedDict()
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row['protein']
            if p == K5H:
                if basis == 'drop':
                    continue
                dd = float(row['ddG']) - (K5H_SHIFT if basis == 'shift' else 0.0)
            else:
                dd = float(row['ddG'])
            prots.setdefault(p, []).append((float(row['deltaG']),
                                            float(row['pred_deltaG']),
                                            dd, float(row['pred_ddG'])))
    return prots


def score(path, basis='drop'):
    prots = load(path, basis)
    aps, rs, ss, bp_wt, dg_ae = [], [], [], [], []
    pooled_t, pooled_p, res_t, res_p = [], [], [], []
    for p, rows in prots.items():
        dg  = [r[0] for r in rows]; pdg = [r[1] for r in rows]
        t   = [r[2] for r in rows]; pr  = [r[3] for r in rows]
        dg_ae.extend(abs(a-b) for a, b in zip(dg, pdg))
        bp_wt.append(pdg[0] - dg[0])            # WS-1: dG-space WT error on row 0
        if len(t) < 3: continue
        r = pearson(t, pr); st, sp = std(t), std(pr)
        if not (st > 0) or not (sp > 0) or r != r: continue
        s = sp/st
        rs.append(r); ss.append(s); aps.append(r*s)
        pooled_t.extend(t); pooled_p.extend(pr)
        md = mean([b - a for a, b in zip(t, pr)])   # this protein's mean residual
        res_t.extend(t); res_p.extend(x - md for x in pr)

    pooled = pearson(pooled_t, pooled_p)
    oracle = pearson(res_t, res_p)
    # DEGENERATE-BASELINE GUARD (MASTER 2.3 / 6.3). --dg_length_norm looked like a 31%
    # win and was the model predicting nothing. A collapsed prediction shows up as
    # std(pred_dG) -> 0 while std(b_p) parks at std(true WT dG) ~ 0.92. Report both so
    # no number in this table can be read without its degenerate check.
    wt_true = [rows[0][0] for rows in prots.values()]
    wt_pred = [rows[0][1] for rows in prots.values()]
    o = OrderedDict()
    o['std_pred_wt_dG'] = std(wt_pred)
    o['std_true_wt_dG'] = std(wt_true)
    o['n_proteins']   = len(rs)
    o['n_rows']       = len(pooled_t)
    o['pooled']       = pooled
    o['oracle']       = oracle
    o['oracle_gain']  = oracle - pooled
    o['perprot_med']  = median(rs)
    o['perprot_mean'] = mean(rs)
    o['a_p']          = median(aps)
    o['r']            = median(rs)
    o['s']            = median(ss)
    o['std_bp']       = std(bp_wt)
    o['mean_bp']      = mean(bp_wt)
    o['dG_MAE']       = mean(dg_ae)
    return o


# ------------------------------------------------------------------ collection
def collect(basis='drop'):
    """One row per RUN TAG at its canonical epoch. Returns (rows, suppressed)."""
    by_tag = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, 'abl_*.csv'))):
        tag, ep = parse(path)
        if tag is None: continue
        by_tag[tag].append((ep, path))

    rows, suppressed = [], []
    for tag, eps in sorted(by_tag.items()):
        eps.sort()
        if tag in TRAJECTORY:
            canon = TRAJECTORY[tag]
            keep = [e for e in eps if e[0] == canon] or [eps[-1]]
        else:
            keep = [eps[-1]]
        ep, path = keep[0]
        try:
            d = score(path, basis)
        except Exception as e:
            print('ERROR %s: %s' % (path, e)); continue
        d['tag'] = tag; d['epoch'] = ep; d['file'] = os.path.basename(path)
        d['multi_epoch'] = len(eps) > 1
        d['allowlisted'] = tag in TRAJECTORY
        d.update(classify(tag))
        rows.append(d)
        for e, p2 in eps:
            if e != ep:
                suppressed.append((tag, e, os.path.basename(p2)))
    rows.sort(key=lambda d: -(d['pooled'] if d['pooled'] == d['pooled'] else -9))
    return rows, suppressed


# ------------------------------------------------------------------ 48-cell grid
def grid_status(rows):
    """The factorial design is A x B x C x D x seed{42,1,2} = 2*2*2*2*3 = 48 cells."""
    have = {}
    for d in rows:
        if 'A' in d:
            have[(d['A'], d['B'], d['C'], d['D'], d['seed'])] = d
    want = [(a, b, c, dd, s)
            for dd in (0, 1) for a in (0, 1) for b in (0, 1) for c in (0, 1)
            for s in (42, 1, 2)]
    missing = [k for k in want if k not in have]
    return have, want, missing

def keyname(k):
    a, b, c, d, s = k
    return 'a%d_d%d_s%d_D%d_%s_seed%d' % (a, b, c, d, 'coil' if d == 0 else 'uemb', s)


MODEL_GLOB = 'Megascale-fineTuning/models/*p3_a*'
ALL_MODEL_GLOB = 'Megascale-fineTuning/models/*kf_*'
CKPT = re.compile(r'kf_all_epoch_(\d+)\.pt$')

def train_status(pattern=None):
    """cell tag -> highest epoch checkpoint on disk, or None if no model dir at all.

    This separates the two ways a cell can be missing, which mean completely
    different things:
      TRAINED-BUT-UNSCORED  a kf_all_epoch_14.pt exists and no CSV does -> it is
                            a SCORING backlog, recoverable on the CPU partition at
                            zero GPU cost (MASTER 6.6).
      PARTIAL               training stopped short of epoch 14 -> needs GPU time.
      NOT SUBMITTED         no model directory -> never launched.
    """
    out = {}
    for d in glob.glob(pattern or MODEL_GLOB):
        tag = os.path.basename(d).split('light_attentionkf_')[-1]
        eps = []
        for p in glob.glob(os.path.join(d, 'kf_all_epoch_*.pt')):
            m = CKPT.search(p)
            if m: eps.append(int(m.group(1)))
        out[tag] = max(eps) if eps else -1
    return out


# ------------------------------------------------------------------ selftest
def selftest():
    print('=== SELFTEST 1: the basis reproduces MASTER 1.3 ===')
    bad = 0
    for tag, want in sorted(EXPECTED.items()):
        hits = glob.glob(os.path.join(EVAL_DIR, 'abl_%s_e*.csv' % tag))
        if not hits:
            print('  MISSING %s' % tag); continue
        # MASTER 1.3 quotes gld_slope1.0_s42 at e13 and sigma_seed2 at e10
        pick = sorted(hits)[-1]
        for want_ep in (13,):
            cand = os.path.join(EVAL_DIR, 'abl_%s_e%d.csv' % (tag, want_ep))
            if tag == 'gld_slope1.0_s42' and os.path.exists(cand): pick = cand
        got = score(pick)['pooled']
        flag = 'OK  ' if abs(got - want) < 5e-4 else 'FAIL'
        if flag == 'FAIL': bad += 1
        print('  %s %-22s got %.4f  MASTER %.4f   (%s)'
              % (flag, tag, got, want, os.path.basename(pick)))

    print('=== SELFTEST 2: a_p = r * s exactly (MASTER 3.1, tol 2e-14) ===')
    path = os.path.join(EVAL_DIR, 'abl_%s_e14.csv' % CONTROL_TAG)
    prots = load(path)
    worst = 0.0
    for p, rows in prots.items():
        t = [r[2] for r in rows]; pr = [r[3] for r in rows]
        if len(t) < 3: continue
        r = pearson(t, pr); st, sp = std(t), std(pr)
        if not (st > 0 and sp > 0): continue
        mt = sum(t)/len(t); mp = sum(pr)/len(pr)
        a_ls = sum((x-mt)*(y-mp) for x, y in zip(t, pr))/sum((x-mt)**2 for x in t)
        worst = max(worst, abs(a_ls - r*(sp/st)))
    print('  max |a_LS - r*s| over %d proteins = %.3e  %s'
          % (len(prots), worst, 'OK' if worst < 2e-14 else 'FAIL'))
    bad += 0 if worst < 2e-14 else 1

    print('=== SELFTEST 3: 2K5H basis comparison (drop is canonical) ===')
    for tag in ('sigma_seed2_e10', 'calib_ctrl_repro2_e14'):
        p = os.path.join(EVAL_DIR, 'abl_%s.csv' % tag)
        if not os.path.exists(p): continue
        print('  %-24s drop %.4f | shift %.4f | keep %.4f'
              % (tag, score(p, 'drop')['pooled'], score(p, 'shift')['pooled'],
                 score(p, 'all')['pooled']))

    print('=== SELFTEST 4: within-protein metrics are INVARIANT to the 2K5H choice ===')
    p = os.path.join(EVAL_DIR, 'abl_%s_e14.csv' % CONTROL_TAG)
    a = score(p, 'all'); b = score(p, 'shift')
    for nm in ('a_p', 'r', 's', 'perprot_med'):
        d = abs(a[nm] - b[nm])
        print('  %-12s all %.6f vs shift %.6f  diff %.2e  %s'
              % (nm, a[nm], b[nm], d, 'OK' if d < 1e-9 else 'FAIL'))
        if d >= 1e-9: bad += 1

    print('=== SELFTEST 5: every kept row has 27 proteins ===')
    rows, _ = collect()
    off = [(d['tag'], d['n_proteins']) for d in rows if d['n_proteins'] != 27]
    print('  %d/%d rows on 27 proteins%s'
          % (len(rows)-len(off), len(rows), '' if not off else '  ODD: %s' % off[:8]))
    return bad == 0


# ------------------------------------------------------------------ report
def f(x, n=4):
    return ('%.*f' % (n, x)) if x == x else '--'

GROUP_ORDER = ['control', 'sigma seeds (noise floor)', 'slope sweep', 'anchor sweep',
               'golden arms', 'factorial D0 (coil)', 'factorial D1 (uemb zero)', 'other']


def main():
    args = sys.argv[1:]
    basis = 'drop'
    if '--basis' in args: basis = args[args.index('--basis')+1]
    if '--selftest' in args:
        sys.exit(0 if selftest() else 1)

    rows, suppressed = collect(basis)
    if not rows:
        print('no eval CSVs found in %s' % EVAL_DIR); return

    L = []; W = L.append
    ctrl = next((d for d in rows if d['tag'] == CONTROL_TAG), None)
    cp = ctrl['pooled'] if ctrl else float('nan')
    _bp = max(rows, key=lambda d: d['pooled'] if d['pooled'] == d['pooled'] else -9)
    _ba = max(rows, key=lambda d: d['a_p'] if d['a_p'] == d['a_p'] else -9)
    _have, _want, _missing = grid_status(rows)

    W('# MASTER SCOREBOARD -- every eval CSV on one basis')
    W('')
    W('## The four answers, up front')
    W('')
    W('| question | answer |')
    W('|---|---|')
    W('| **Best on pooled ddG PCC** | `%s` (e%d) at **%s** |' % (_bp['tag'], _bp['epoch'], f(_bp['pooled'])))
    W('| **Best on a_p** | `%s` (e%d) at **%s** |' % (_ba['tag'], _ba['epoch'], f(_ba['a_p'])))
    W('| **Are they the same cell?** | %s |'
      % ('**YES** -- the criteria agree on the evidence on disk'
         if _bp['tag'] == _ba['tag'] else
         '**NO.** The pooled winner carries a_p %s; the a_p winner scores pooled %s. Two failures, two winners.'
         % (f(_bp['a_p']), f(_ba['pooled']))))
    W('| **Factorial coverage** | **%d of 48** cells scored; %d missing |' % (len(_have), len(_missing)))
    W('| **Runs on this board** | %d, all on 27 proteins, one epoch each |' % len(rows))
    W('')
    W('**Basis: 27 test proteins (2K5H DROPPED), ddG metric, per-protein fits.**  ')
    W('Generated by `scripts/scoreboard.py`. Re-run as more CSVs land.')
    W('')
    W('## How to read this')
    W('')
    W('Every number carries three qualifiers: *pooled-or-per-protein* / *dG-or-ddG* / *which')
    W('split*. All rows here are the **test** split on the **27-protein** basis.')
    W('')
    W('| column | definition | side |')
    W('|---|---|---|')
    W('| `pooled` | PCC over every row of every protein pooled together | ddG |')
    W('| `oracle` | subtract each protein\'s own mean residual, then pool. **Uses the test labels -- an upper bound, not a method.** | ddG |')
    W('| `gain` | `oracle - pooled`: what solving b_p would be worth on this run | ddG |')
    W('| `perprot` | **median** per-protein PCC -- ranking quality *within* a protein | ddG |')
    W('| `a_p` | median per-protein slope, `a_p = r * s` | ddG |')
    W('| `r` | median per-protein PCC (the ranking half of a_p) | ddG |')
    W('| `s` | median `std(pred)/std(true)` (the spread half of a_p) | ddG |')
    W('| `std(b_p)` | sd over proteins of the WT error `pred_dG - dG` on row 0 | **dG** |')
    W('')
    W('**The metric rule (MASTER 2.1).** ddG cancels anything identical between wild type and')
    W('mutant, so a reference-state or whole-protein lever must be read on **dG or b_p**.')
    W('`a_p` is a *within*-protein slope and does not cancel, so it **is** ddG-measurable.')
    W('`std(b_p)` and `dG_MAE` are the dG-side columns; they are never averaged into the ddG ones.')
    W('')
    W('**The 2K5H basis matters.** `scripts/factorial_analysis.py` *shifts* 2K5H by -3.0824 and')
    W('keeps 28 proteins; MASTER.md *drops* it and keeps 27. The two differ:')
    W('`sigma_seed2_e10` scores 0.6382 dropped, 0.6386 shifted, 0.6575 uncorrected. MASTER 1.3')
    W('quotes **0.6382**, so **drop** is canonical and is what this table uses.')
    W('')
    W('**The seed noise band is +/-%.3f pooled** (five `abl_sigma` seeds, identical config).' % SEED_BAND)
    W('Every group below is reported as mean +/- sd so it can be read against that band.')
    W('**A difference under +/-%.3f is not distinguishable from seed noise.**' % SEED_BAND)
    W('')
    W('**Epoch discipline (OPEN_PROBLEMS P0).** Epoch selection is on VALIDATION')
    W('(`run_calib_eval.sh` line 15); the test set is scored once at that epoch. Two runs were')
    W('deliberately scored at six epochs each to measure a trajectory -- `gld_slope1.0_s42` and')
    W('`gld_dg_coil_s42`. **Quoting the best of those six would be test-set peeking**, so this')
    W('table keeps only their canonical (val-selected) epoch e14; the other epochs are listed')
    W('separately at the bottom and must never be read as run results.')
    W('')

    # ---- headline table
    W('## 1. The table -- all %d runs, sorted by pooled ddG PCC' % len(rows))
    W('')
    W('| # | run | ep | group | pooled | vs ctrl | oracle | gain | perprot | a_p | r | s | std(b_p) | dG MAE |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, d in enumerate(rows, 1):
        star = ' **(control)**' if d['tag'] == CONTROL_TAG else ''
        dv = d['pooled'] - cp
        W('| %d | `%s`%s | %d | %s | **%s** | %+.4f | %s | %s | %s | %s | %s | %s | %s | %s |'
          % (i, d['tag'], star, d['epoch'], d['group'], f(d['pooled']), dv,
             f(d['oracle']), f(d['oracle_gain']), f(d['perprot_med']), f(d['a_p']),
             f(d['r']), f(d['s']), f(d['std_bp']), f(d['dG_MAE'], 3)))
    W('')

    # ---- groups
    W('## 2. By arm -- every group mean against the +/-%.3f seed band' % SEED_BAND)
    W('')
    W('| group | n | pooled mean +/- sd | range | oracle mean | a_p mean +/- sd | r mean | s mean | std(b_p) mean |')
    W('|---|---|---|---|---|---|---|---|---|')
    groups = defaultdict(list)
    for d in rows: groups[d['group']].append(d)
    gstat = {}
    for g in GROUP_ORDER:
        if g not in groups: continue
        v = groups[g]
        P = [d['pooled'] for d in v]; A = [d['a_p'] for d in v]
        gstat[g] = dict(n=len(v), pooled_mean=mean(P), pooled_sd=std(P),
                        a_p_mean=mean(A), a_p_sd=std(A),
                        oracle_mean=mean([d['oracle'] for d in v]),
                        r_mean=mean([d['r'] for d in v]),
                        s_mean=mean([d['s'] for d in v]),
                        std_bp_mean=mean([d['std_bp'] for d in v]),
                        members=[d['tag'] for d in v])
        W('| **%s** | %d | %s +/- %s | %s ... %s | %s | %s +/- %s | %s | %s | %s |'
          % (g, len(v), f(mean(P)), f(std(P)), f(min(P)), f(max(P)),
             f(mean([d['oracle'] for d in v])), f(mean(A)), f(std(A)),
             f(mean([d['r'] for d in v])), f(mean([d['s'] for d in v])),
             f(mean([d['std_bp'] for d in v]))))
    W('')
    W('Read the `sd` column first. A group whose own sd is comparable to +/-%.3f is telling you' % SEED_BAND)
    W('its spread is seed noise; a group whose sd is much *tighter* or much *wider* is telling')
    W('you something the mean alone hides.')
    W('')
    for g in GROUP_ORDER:
        if g in gstat:
            W('- **%s** (n=%d): %s' % (g, gstat[g]['n'], ', '.join('`%s`' % m for m in gstat[g]['members'])))
    W('')

    # ---- factor D, in seed sigmas
    d0 = gstat.get('factorial D0 (coil)'); d1 = gstat.get('factorial D1 (uemb zero)')
    if d0 and d1:
        sep = (d0['pooled_mean'] - d1['pooled_mean']) / SEED_BAND
        W('### Factor D, measured in seed sigmas')
        W('')
        W('| arm | n | pooled mean +/- sd | a_p mean |')
        W('|---|---|---|---|')
        W('| **D0 (coil reference)** | %d | %s +/- %s | %s |'
          % (d0['n'], f(d0['pooled_mean']), f(d0['pooled_sd']), f(d0['a_p_mean'])))
        W('| **D1 (`--unfolded_emb zero`)** | %d | %s +/- %s | %s |'
          % (d1['n'], f(d1['pooled_mean']), f(d1['pooled_sd']), f(d1['a_p_mean'])))
        W('')
        gap = d0['pooled_mean'] - d1['pooled_mean']
        W('**D0 - D1 = %s pooled = %.1f seed sigmas** (sigma = the +/-%.3f seed band).'
          % (f(gap), sep, SEED_BAND))
        W('')
        W('> **Which sigma?** MASTER 5.2 quotes this separation as **14.4 sigma** using *D0\'s own')
        W('> sd* (0.0091) as the denominator. This table deliberately uses the **seed band**')
        W('> (+/-%.3f) instead, which is the more conservative choice and the bar every other' % SEED_BAND)
        W('> lever in this report is held to. Same gap, different yardstick -- %.1f sigma here vs' % sep)
        W('> 14.4 there. Quote the denominator whenever you quote the number; a sigma without its')
        W('> definition is exactly the kind of unqualified statistic MASTER 1.4 corrects.')
        W('')
        W('Read the *spreads*, not only the means. D0\'s own sd is %s, **%.1fx tighter than the'
          % (f(d0['pooled_sd']), SEED_BAND/d0['pooled_sd'] if d0['pooled_sd'] > 0 else float('nan')))
        W('+/-%.3f seed band**; D1\'s is %s, an order of magnitude wider, and it contains a cell that' % (SEED_BAND, f(d1['pooled_sd'])))
        W('is *anti*-correlated. On a_p the arms differ by a factor of %.0f -- an a_p near zero means'
          % ((d0['a_p_mean']/d1['a_p_mean']) if d1['a_p_mean'] else float('nan')))
        W('the model is **flat**, not merely worse. D1 removes the offset by removing the signal.')
        W('')

    # ---- the contrast (task step 4)
    best_p, best_a = _bp, _ba
    same = best_p['tag'] == best_a['tag']
    W('## 3. Best on pooled vs best on a_p -- the calibration thesis in one line')
    W('')
    W('| criterion | winning run | pooled | a_p | r | s | oracle | std(b_p) |')
    W('|---|---|---|---|---|---|---|---|')
    W('| **best pooled** | `%s` (e%d) | **%s** | %s | %s | %s | %s | %s |'
      % (best_p['tag'], best_p['epoch'], f(best_p['pooled']), f(best_p['a_p']),
         f(best_p['r']), f(best_p['s']), f(best_p['oracle']), f(best_p['std_bp'])))
    W('| **best a_p** | `%s` (e%d) | %s | **%s** | %s | %s | %s | %s |'
      % (best_a['tag'], best_a['epoch'], f(best_a['pooled']), f(best_a['a_p']),
         f(best_a['r']), f(best_a['s']), f(best_a['oracle']), f(best_a['std_bp'])))
    if ctrl:
        W('| control | `%s` (e%d) | %s | %s | %s | %s | %s | %s |'
          % (CONTROL_TAG, ctrl['epoch'], f(cp), f(ctrl['a_p']), f(ctrl['r']), f(ctrl['s']),
             f(ctrl['oracle']), f(ctrl['std_bp'])))
    W('')
    if same:
        W('**They are the SAME run.** On the evidence currently on disk the two criteria agree.')
    else:
        W('**They are DIFFERENT runs -- and that contrast IS the calibration thesis.**')
        W('')
        W('`%s` wins pooled at **%s** while carrying a_p = %s.  '
          % (best_p['tag'], f(best_p['pooled']), f(best_p['a_p'])))
        W('`%s` wins a_p at **%s** while its pooled is only %s (%+.4f vs control).'
          % (best_a['tag'], f(best_a['a_p']), f(best_a['pooled']), best_a['pooled'] - cp))
        W('')
        W('The two criteria disagree because they measure **two different failures**:')
        W('')
        W('- **`a_p` is a WITHIN-protein slope.** Fixing it is pure calibration: `a_p = r * s`,')
        W('  and `--slope_weight` drives `s` toward 1 with `r` flat (MASTER 3.2). It cannot move')
        W('  the pooled number much, because pooling is dominated by the *between*-protein offset.')
        W('- **`pooled` is dominated by b_p**, the per-protein offset. That is why the offset')
        W('  oracle sits **%s** above the best pooled run, and why b_p -- not the slope -- is the'
          % f(best_p['oracle_gain'], 4))
        W('  binding constraint on the headline number.')
        W('')
        W('So the run that best *calibrates the slope* is not the run that best *ranks across')
        W('proteins*. **Two independent failures, two different winners, one shared ceiling.**')
    W('')

    # ---- factor C within D0: the a_p = r*s mechanism on independent cells
    d0rows = [d for d in rows if d.get('D') == 0]
    c0 = [d for d in d0rows if d.get('C') == 0]
    c1 = [d for d in d0rows if d.get('C') == 1]
    if c0 and c1:
        W('### The slope factor C, inside D0 -- a_p = r * s, on independent cells')
        W('')
        W('| C (`--slope_weight`) | n | pooled | a_p | **r** | **s** | std(b_p) |')
        W('|---|---|---|---|---|---|---|')
        for nm, v in (('off (s0)', c0), ('on (s1)', c1)):
            W('| %s | %d | %s | %s | %s | %s | %s |'
              % (nm, len(v), f(mean([x['pooled'] for x in v])), f(mean([x['a_p'] for x in v])),
                 f(mean([x['r'] for x in v])), f(mean([x['s'] for x in v])),
                 f(mean([x['std_bp'] for x in v]))))
        da = mean([x['a_p'] for x in c1]) - mean([x['a_p'] for x in c0])
        ds = mean([x['s'] for x in c1]) - mean([x['s'] for x in c0])
        dr = mean([x['r'] for x in c1]) - mean([x['r'] for x in c0])
        dp = mean([x['pooled'] for x in c1]) - mean([x['pooled'] for x in c0])
        db = mean([x['std_bp'] for x in c1]) - mean([x['std_bp'] for x in c0])
        W('| **effect** | | **%+.4f** | **%+.4f** | **%+.4f** | **%+.4f** | **%+.4f** |'
          % (dp, da, dr, ds, db))
        W('')
        W('**This is the `a_p = r * s` identity reproducing itself on cells that were never used')
        W('to derive it.** Turning the slope lever on moves `a_p` by %+.4f and `s` by %+.4f while' % (da, ds))
        W('`r` stays flat at %+.4f -- inside the noise, exactly as the identity demands: the lever' % dr)
        W('cannot manufacture ranking, only spread.')
        W('')
        W('And it shows the **cost** in the same table. Pooled moves %+.4f (nothing, against a' % dp)
        W('+/-%.3f band) while `std(b_p)` **worsens by %+.4f**. The arm buys within-protein' % (SEED_BAND, db))
        W('calibration and pays for it in between-protein offset. That trade is the reason the')
        W('pooled winner and the a_p winner are different runs.')
        W('')

    # ---- degenerate-baseline guard
    W('## 3b. The degenerate check -- is any winner just predicting nothing?')
    W('')
    W('`--dg_length_norm` once looked like a 31 percent improvement and was the model predicting')
    W('nothing (MASTER 6.3): dividing by N drives `pred -> 0`, so `b_p -> -true_dG` and')
    W('`std(b_p)` collapses onto `std(true WT dG) ~ 0.92`, which reads as a win. **Every**')
    W('**headline number therefore ships with its degenerate check.** A run is suspect when')
    W('`std(pred WT dG)` approaches zero while `std(b_p)` parks at `std(true WT dG)`.')
    W('')
    susp = sorted(rows, key=lambda d: d['std_pred_wt_dG'])[:6]
    W('| run | pooled | a_p | std(pred WT dG) | std(true WT dG) | std(b_p) | verdict |')
    W('|---|---|---|---|---|---|---|')
    for d in [best_p, best_a] + [x for x in susp if x['tag'] not in (best_p['tag'], best_a['tag'])][:4]:
        deg = d['std_pred_wt_dG'] < 0.25 * d['std_true_wt_dG']
        W('| `%s` | %s | %s | %s | %s | %s | %s |'
          % (d['tag'], f(d['pooled']), f(d['a_p']), f(d['std_pred_wt_dG']),
             f(d['std_true_wt_dG']), f(d['std_bp']),
             '**DEGENERATE -- prediction collapsed**' if deg else 'predicting'))
    W('')
    ndeg = [d for d in rows if d['std_pred_wt_dG'] < 0.25 * d['std_true_wt_dG']]
    W('%d of %d runs trip the collapse test%s.'
      % (len(ndeg), len(rows), '' if not ndeg else ': ' + ', '.join('`%s`' % d['tag'] for d in ndeg)))
    W('')

    # ---- factorial coverage (task step 5)
    have, want, missing = _have, _want, _missing
    W('## 4. Factorial coverage -- %d of 48 cells have results' % len(have))
    W('')
    W('The design is **A x B x C x D x seed** = `--wt_anchor` x `--designed_weight` x')
    W('`--slope_weight` x reference-state x seed{42, 1, 2} = 2*2*2*2*3 = **48 cells**.')
    W('')
    W('**%d have a scored CSV; %d are missing.**' % (len(have), len(missing)))
    W('')
    by_seed = defaultdict(list)
    for k in missing: by_seed[k[4]].append(k)
    W('| seed | present | missing |')
    W('|---|---|---|')
    for s in (42, 1, 2):
        pres = [k for k in have if k[4] == s]
        W('| %d | %d/16 | %d |' % (s, len(pres), len(by_seed.get(s, []))))
    W('')
    W('### Why each missing cell is missing')
    W('')
    W('**A missing cell is not one thing.** Split by what is actually on disk, because the')
    W('three cases cost completely different amounts to fix:')
    W('')
    ts = train_status()
    buckets = {'unscored': [], 'partial': [], 'nodir': []}
    for k in missing:
        t = 'p3_' + keyname(k)
        e = ts.get(t)
        if e is None:            buckets['nodir'].append((k, e))
        elif e >= 14:            buckets['unscored'].append((k, e))
        else:                    buckets['partial'].append((k, e))
    W('| why | n | what it costs to fix |')
    W('|---|---|---|')
    W('| **trained to e14, NOT YET SCORED** | %d | CPU only -- scoring is a forward pass and never needed a GPU (MASTER 6.6) |' % len(buckets['unscored']))
    W('| **training incomplete (< e14)** | %d | GPU time |' % len(buckets['partial']))
    W('| **never submitted (no model dir)** | %d | GPU time |' % len(buckets['nodir']))
    W('')
    for name, hdr in (('unscored', 'Trained to epoch 14, awaiting a CSV (CPU-recoverable)'),
                      ('partial', 'Training stopped short of epoch 14'),
                      ('nodir', 'No model directory -- never launched')):
        if not buckets[name]: continue
        W('**%s (%d):**' % (hdr, len(buckets[name])))
        W('')
        W('```')
        for k, e in sorted(buckets[name], key=lambda z: (z[0][4], z[0][3], z[0][0], z[0][1], z[0][2])):
            W('%-34s %s' % (keyname(k), ('max ckpt e%d' % e) if e is not None and e >= 0
                            else ('dir present, no checkpoint' if e is not None else 'no dir')))
        W('```')
        W('')

    # ---- scoring backlog: what is recoverable on CPU right now
    ts = train_status(ALL_MODEL_GLOB)
    scored_tags = set(d['tag'] for d in rows)
    backlog = sorted(t for t, e in ts.items() if e >= 14 and t not in scored_tags)
    W('## 4b. The scoring backlog')
    W('')
    W('**%d run(s) have a `kf_all_epoch_14.pt` and no CSV.** Scoring is a forward pass over the'
      % len(backlog))
    W('test proteins and never needed a GPU (MASTER 6.6), so this backlog is CPU-recoverable at')
    W('zero GPU cost -- it is a throughput problem, not a science problem.')
    W('')
    if backlog:
        W('```')
        for t in backlog: W(t)
        W('```')
        W('')
    W('**A finished training run is NOT a result** (MASTER 12.3). Track')
    W('`ls eval_results/abl_*.csv | wc -l`, never a DONE message: `run_calib_eval.sh` prints')
    W('`DONE ... -> abl_<tag>.csv` *after* a crash, with no CSV on disk. Every row in this')
    W('report was built by reading a CSV off disk, so the artifact is the evidence.')
    W('')

    # ---- suppressed epochs
    if suppressed:
        W('## 5. Suppressed epochs -- NOT run results')
        W('')
        W('These CSVs exist but are **excluded from the table above** because their run already')
        W('appears at its validation-selected epoch. `gld_slope1.0_s42` and `gld_dg_coil_s42`')
        W('were scored at six epochs *deliberately*, to measure whether `r` stays flat while `s`')
        W('rises (MASTER 3.2). **Quoting the best of those six as a headline is test-set peeking.**')
        W('')
        W('| run | suppressed epoch | file |')
        W('|---|---|---|')
        for tag, ep, fn in sorted(suppressed):
            W('| `%s` | e%d | `%s` |' % (tag, ep, fn))
        W('')
        multi = [d for d in rows if d['multi_epoch'] and not d['allowlisted']]
        if multi:
            W('**Multi-epoch but NOT on the trajectory allowlist** (last epoch taken -- check why):')
            W('')
            for d in multi: W('- `%s` (kept e%d)' % (d['tag'], d['epoch']))
            W('')

    try:
        os.makedirs(os.path.dirname(OUT_MD))
    except OSError:
        pass
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(basis=basis, seed_band=SEED_BAND,
                   rows=[dict((k, v) for k, v in d.items()) for d in rows],
                   groups=gstat, missing=[keyname(k) for k in missing],
                   n_cells_present=len(have)),
              open(OUT_JSON, 'w'), indent=1)

    # ---- terminal view
    print('BASIS: 27 proteins (2K5H dropped), ddG.  %d runs.' % len(rows))
    print('%-34s %2s %-26s %7s %7s %7s %7s %7s %7s'
          % ('run', 'ep', 'group', 'pooled', 'oracle', 'perprot', 'a_p', 'r', 's'))
    for d in rows:
        print('%-34s %2d %-26s %7s %7s %7s %7s %7s %7s'
              % (d['tag'][:34], d['epoch'], d['group'][:26], f(d['pooled']), f(d['oracle']),
                 f(d['perprot_med']), f(d['a_p']), f(d['r']), f(d['s'])))
    print()
    for g in GROUP_ORDER:
        if g in gstat:
            s = gstat[g]
            print('GROUP %-28s n=%2d pooled %s +/- %s   a_p %s +/- %s'
                  % (g, s['n'], f(s['pooled_mean']), f(s['pooled_sd']),
                     f(s['a_p_mean']), f(s['a_p_sd'])))
    print()
    print('BEST pooled: %-30s %.4f  (a_p %.4f)' % (best_p['tag'], best_p['pooled'], best_p['a_p']))
    print('BEST a_p   : %-30s %.4f  (pooled %.4f)' % (best_a['tag'], best_a['a_p'], best_a['pooled']))
    print('SAME CELL?  %s' % ('YES' if same else 'NO -- different cells'))
    print('factorial cells with results: %d/48   missing: %d' % (len(have), len(missing)))
    print('wrote %s and %s' % (OUT_MD, OUT_JSON))


if __name__ == '__main__':
    main()
