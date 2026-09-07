"""AUTOPILOT -- the on-cluster driver. Polls, verifies, decides by written rule, submits.

Once running under tmux this consumes no LLM context at all. It wakes a human only on
one of the four halts in AUTOPILOT.md section 7.

Every decision traces to a numbered rule. Where a situation is not covered by a rule,
it HALTS rather than improvising -- being clever is the failure mode here.

ABSOLUTE: never cancels a job it did not itself submit and record in state.json.
Three irreplaceable trainings were running when this was written.

Usage:
    python scripts/autopilot.py                 # dry run, prints the decision trace
    python scripts/autopilot.py --go            # live
    python scripts/autopilot.py --once --go     # a single pass, for cron
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
STATE = os.path.join(ROOT, 'results', 'state.json')
STATUS = os.path.expanduser('~/auto/STATUS.txt')
HALT_MD = os.path.join(ROOT, 'results', 'HALT.md')
TSV = os.path.join(ROOT, 'results', 'RESULTS.tsv')

SIGMA_POOLED = 0.0072
SIGMA_PP = 0.0037
EFFECT_MIN = 0.0083          # 2 * SIGMA_POOLED / sqrt(3)
REF_POOLED, REF_PP, REF_STD_B = 0.591, 0.731, 0.223
CEILING = 0.711
# AUTOPILOT.md D1: "A cell that raises pooled while collapsing slope below 0.3 median is
# not selected, even if its pooled is highest -- that is the anchor's known failure mode
# and selecting it would defeat the purpose." The driver used to select on pooled alone,
# so it could and would pick exactly that cell. a_p is a within-protein slope: it does not
# cancel in ddG, so it is legitimately measurable here, and it is a separate calibration
# channel (corr(a_p, per-protein PCC) = +0.571; mean_rel_SASA vs a_p r = +0.714, 10/10).
SLOPE_MED_MIN = 0.3
MIN_EVAL_CSV_BYTES = 1000    # a finished training run is not a result
POLL_SECONDS = 600
MAX_CONCURRENT = 8

TSV_COLS = ['run_tag', 'state', 'node_class', 'seed', 'epochs', 'best_epoch', 'pooled',
            'pp', 'abs_wt_dg', 'std_b', 'slope_med', 'slope_min', 'designed_pooled',
            'wall_h', 'notes']


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode('utf-8', 'replace')
    except subprocess.TimeoutExpired:
        return 124, 'TIMEOUT'


def log(msg):
    line = '[%s] autopilot: %s' % (time.strftime('%m-%d %H:%M'), msg)
    print(line, flush=True)
    try:
        d = os.path.dirname(STATUS)
        if not os.path.isdir(d):
            os.makedirs(d)
        with open(STATUS, 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ---------------------------------------------------------------- state
def _default_state():
    return {'state': 'S3_FACTORIAL', 'entered': time.strftime('%Y-%m-%dT%H:%MZ'),
            'cells_done': 0, 'cells_total': 16, 'decisions': [], 'blocked': [],
            'retries': {}, 'submitted': []}


def load_state():
    """Return a USABLE state, never a half-one.

    The old version only fell back to the default when the file was ABSENT. A file
    containing '{}' - which is what an interrupted or truncated write leaves behind -
    was loaded as-is, and main() then died on KeyError: 'state'. That killed the
    autopilot silently: the wrapper logged 'driver exited rc=1' and nothing was
    scored for hours. Merge onto the default so any missing key is filled in.
    """
    base = _default_state()
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                loaded = json.load(f)
        except Exception:
            loaded = None
        if isinstance(loaded, dict) and loaded:
            base.update(loaded)
    return base


def save_state(st):
    """Crash-safe: temp file then rename, so an interrupted write never truncates."""
    d = os.path.dirname(STATE)
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE)


def decide(st, node, choice, evidence):
    """Record a decision. NEVER re-derived once written -- it was made with evidence
    that is recorded, and re-deciding without that evidence is how a stale premise
    gets re-adopted (AUTOPILOT section 5)."""
    for d in st['decisions']:
        if d['node'] == node:
            return d
    d = {'node': node, 'choice': choice, 'evidence': evidence,
         'ts': time.strftime('%Y-%m-%dT%H:%MZ')}
    st['decisions'].append(d)
    save_state(st)
    log('DECISION %s -> %s  (%s)' % (node, choice, evidence))
    return d


def halt(st, code, why, tried, causes):
    body = ['# HALT %s' % code, '', '**State:** %s' % st.get('state'), '',
            '## Evidence', why, '', '## What was tried', tried, '',
            '## Three most likely causes'] + ['%d. %s' % (i + 1, c) for i, c in enumerate(causes)]
    with open(HALT_MD, 'w') as f:
        f.write('\n'.join(body) + '\n')
    st['blocked'].append({'halt': code, 'ts': time.strftime('%Y-%m-%dT%H:%MZ'), 'why': why})
    save_state(st)
    log('PROBLEM: HALT %s -- %s  (see results/HALT.md)' % (code, why))
    sys.exit(3)


# ---------------------------------------------------------------- queue
def squeue():
    rc, out = sh('squeue -u $USER -h -o "%i|%j|%T|%M|%N"')
    jobs = []
    for line in out.strip().splitlines():
        p = line.split('|')
        if len(p) == 5:
            jobs.append(dict(jid=p[0].strip(), name=p[1].strip(), state=p[2].strip(),
                             time=p[3].strip(), node=p[4].strip()))
    return jobs


def node_class(node):
    m = re.match(r'([a-z]+-[a-z0-9]+)', node or '')
    return m.group(1) if m else (node or 'unknown')


def eval_csv_for(tag):
    """A FINISHED TRAINING RUN IS NOT A RESULT. Three separate 7h+ runs reached
    COMPLETED before any eval CSV existed. Completion is defined by the CSV."""
    d = os.path.join(ROOT, 'eval_results')
    if not os.path.isdir(d):
        return None
    best = None
    for f in os.listdir(d):
        if f.startswith('abl_%s_e' % tag) and f.endswith('.csv'):
            p = os.path.join(d, f)
            if os.path.getsize(p) > 1000:
                if best is None or os.path.getsize(p) > os.path.getsize(best):
                    best = p
    return best


def calib_diag(csv):
    # calib_diag takes --csv, not a positional argument. Verified against its --help:
    # the positional form exits 2 and every cell would have silently scored as unparsed.
    rc, out = sh('python cluster_run/code/calib_diag.py --csv "%s" 2>&1' % csv, timeout=300)
    if rc != 0:
        return None
    # Parse the [ALL] block only -- the file also prints a [DESIGNED] block with the
    # same labels, and a looser pattern would silently pick up whichever came last.
    g = {}
    pats = (('pooled', r'\[ALL\]\s+pooled ddG PCC\s*=\s*(-?\d+\.\d+)'),
            ('pp', r'\[ALL\]\s+PP \(mean per-protein\)\s*=\s*(-?\d+\.\d+)'),
            ('std_b', r'\[ALL\]\s+std\(b\).*?=\s*(-?\d+\.\d+)'),
            ('offset_removed', r'\[ALL\]\s+pooled OFFSET-removed PCC\s*=\s*(-?\d+\.\d+)'),
            ('slope_med', r'\[ALL\]\s+slope a_p\s+min/med/max\s*=\s*-?\d+\.\d+\s*/\s*(-?\d+\.\d+)'),
            ('slope_min', r'\[ALL\]\s+slope a_p\s+min/med/max\s*=\s*(-?\d+\.\d+)'),
            ('designed_pooled', r'\[DESIGNED\]\s+pooled ddG PCC\s*=\s*(-?\d+\.\d+)'))
    for key, pat in pats:
        m = re.search(pat, out)
        if m:
            g[key] = float(m.group(1))
    return g if 'pooled' in g else None


# ------------------------------------------------- D2 zero-variance guard
# A lever whose driving feature has ZERO VARIANCE on the eval set cannot be measured
# here. Scoring it anyway produces a null that looks like a real negative result --
# a MANUFACTURED NEGATIVE. The 28 MegaScale test proteins are single-chain,
# ligand-free and metal-free monomers (n_het_residues == 0 and n_metal_residues == 0
# for all 28, verified), so --ligand_nodes (W11) is not merely inert on this set, it
# is UNTESTABLE: its block is identically zero for every protein in both the folded
# and the unfolded state, so the cell is byte-identical to baseline by construction.
#
# This is the same principle the MIN_EVAL_CSV_BYTES check enforces one screen down:
# refuse entry, do not score a zero. A cell that cannot vary its input cannot inform
# a main effect, and averaging it in dilutes every other cell on that axis.
#
# The rule is stated on the DRIVING FEATURE, not on the flag name, so a future lever
# gets the same protection without editing this list.
UNTESTABLE_MD = os.path.join(ROOT, 'results', 'UNTESTABLE.md')

# lever -> (driving-feature columns, human description). A lever is testable on this
# eval set only if at least one driving column is present AND varies across proteins.
LEVER_DRIVERS = {
    'ligand_nodes': (('n_het_residues', 'n_metal_residues'),
                     'W11 bound-ligand / cofactor / ion hetero-nodes'),
    'metal_features': (('n_metal_residues',),
                       'W9 metal ions (RETIRED -- superseded by W11 --ligand_nodes)'),
}


def feature_variance(rows, columns):
    """Max spread of any named driving column over the eval set.

    Returns (variance, column_used, n_seen). variance is None when NO named column is
    present at all -- absent is not the same as constant, and the two must not be
    collapsed: a missing column means we cannot even ask the question.
    """
    best = None
    for col in columns:
        vals = [r[col] for r in rows if isinstance(r, dict) and r.get(col) is not None]
        if not vals:
            continue
        try:
            fv = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        spread = max(fv) - min(fv)
        if best is None or spread > best[0]:
            best = (spread, col, len(fv))
    return best if best is not None else (None, None, 0)


def lever_testable(lever, rows):
    """(ok, reason). ok=False means: log 'untestable-here', REFUSE entry to the factorial.

    Never returns a score. A refused lever is not a zero and not a negative result --
    it is a measurement this eval set cannot make.
    """
    drivers, desc = LEVER_DRIVERS.get(lever, ((), lever))
    if not drivers:
        return True, 'no driving feature registered for %r; not guarded' % lever
    if not rows:
        return False, ('untestable-here: no eval rows to measure the variance of %s on'
                       % (', '.join(drivers)))
    var, col, n = feature_variance(rows, drivers)
    if var is None:
        return False, ('untestable-here: none of the driving columns %s is present in the '
                       'eval set, so the variance of %s cannot be established. Absent is '
                       'NOT constant -- refusing to score rather than assume.'
                       % (', '.join(drivers), desc))
    if var == 0:
        return False, ('untestable-here: driving feature %r is CONSTANT across all %d eval '
                       'proteins (spread 0). %s cannot vary its input on this set, so any '
                       'result would be a manufactured negative, not evidence.'
                       % (col, n, desc))
    return True, 'testable: %r varies across %d eval proteins (spread %g)' % (col, n, var)


def admit_lever(st, lever, rows, log_fn=log):
    """D2 gate. True = the lever may enter the factorial; False = recorded untestable."""
    ok, reason = lever_testable(lever, rows)
    log_fn('D2 %s: %s' % (lever, reason))
    if ok:
        return True
    entry = {'lever': lever, 'verdict': 'untestable-here', 'reason': reason,
             'ts': time.strftime('%Y-%m-%dT%H:%MZ')}
    rec = st.setdefault('untestable', [])
    if not any(e.get('lever') == lever for e in rec):
        rec.append(entry)
        save_state(st)
        try:
            d = os.path.dirname(UNTESTABLE_MD)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            new = not os.path.exists(UNTESTABLE_MD)
            with io.open(UNTESTABLE_MD, 'a', encoding='utf-8') as f:
                if new:
                    f.write(u'# Levers refused entry to the factorial\n\n'
                            u'Not negative results. Each lever below has a driving feature '
                            u'with zero variance on this eval set, so it cannot be measured '
                            u'here at all. Scoring one would manufacture a negative.\n\n')
                f.write(u'- **%s** (%s) -- %s\n' % (lever, entry['ts'], reason))
        except (OSError, IOError) as e:
            log_fn('D2 note: could not write %s (%s)' % (UNTESTABLE_MD, e))
    return False


# ---------------------------------------------------------------- factorial parsing
CELL_RE = re.compile(r'^p3_a(\d)_d(\d)_s(\d)_D(\d)_[a-z]+_seed(\d+)$')


def parse_tag(tag):
    m = CELL_RE.match(tag)
    if not m:
        return None
    a, b, c, d, seed = m.groups()
    return dict(A=int(a), B=int(b), C=int(c), D=int(d), seed=int(seed), tag=tag)


def collect_results():
    """Every completed factorial cell, from the eval CSVs that actually exist."""
    rows = []
    d = os.path.join(ROOT, 'eval_results')
    if not os.path.isdir(d):
        return rows
    seen = set()
    for f in sorted(os.listdir(d)):
        m = re.match(r'abl_(p3_a\d_d\d_s\d_D\d_[a-z]+_seed\d+)_e\d+\.csv$', f)
        if not m:
            continue
        tag = m.group(1)
        if tag in seen:
            continue
        seen.add(tag)
        cell = parse_tag(tag)
        if not cell:
            continue
        # A FINISHED TRAINING RUN IS NOT A RESULT. Three separate 7h+ runs reached
        # COMPLETED with no CSV. A stub/empty/header-only CSV must never be scored:
        # it would enter D1 as a real cell. Assert non-empty BEFORE scoring.
        path = os.path.join(d, f)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size < MIN_EVAL_CSV_BYTES:
            log('SKIP %s: eval CSV is %d bytes (< %d) -- not a result, not scored'
                % (tag, size, MIN_EVAL_CSV_BYTES))
            seen.discard(tag)
            continue
        stats = calib_diag(path)
        if stats and 'pooled' in stats:
            cell.update(stats)
            rows.append(cell)
        else:
            log('SKIP %s: calib_diag returned no pooled ddG PCC -- not scored' % tag)
            seen.discard(tag)
    return rows


def select_best(rows, log_fn=log):
    """D1 selection under the documented slope guard.

    Returns (best, rejected) where `rejected` lists (row, slope_med) for every cell
    excluded for slope collapse. A cell is ELIGIBLE only if its median a_p is known and
    >= SLOPE_MED_MIN. Ranking among eligible cells is by pooled, as documented.

    Rejection is LOUD: the whole point of the guard is that a pooled-winning cell got
    thrown out, and that must appear in the log, not be silently skipped.

    A cell with no slope_med is NOT eligible -- unknown slope is not a passing slope.
    If every cell collapses, this returns (None, rejected) and the caller halts rather
    than quietly selecting a collapsed cell.
    """
    eligible, rejected = [], []
    for r in rows:
        sm = r.get('slope_med')
        if sm is None or sm != sm:            # missing or NaN
            rejected.append((r, sm))
            continue
        if sm < SLOPE_MED_MIN:
            rejected.append((r, sm))
            continue
        eligible.append(r)
    for r, sm in sorted(rejected, key=lambda t: -t[0].get('pooled', 0)):
        shown = 'missing' if (sm is None or sm != sm) else '%.3f' % sm
        log_fn('D1 REJECT %s: pooled=%.4f but slope_med=%s < %.2f -- slope collapse; '
               'not selected (AUTOPILOT.md D1)'
               % (r.get('tag', '?'), r.get('pooled', float('nan')), shown, SLOPE_MED_MIN))
    if not eligible:
        return None, rejected
    best = max(eligible, key=lambda r: r['pooled'])
    if rejected:
        top = max(rejected, key=lambda t: t[0].get('pooled', 0))[0]
        if top.get('pooled', 0) > best['pooled']:
            log_fn('D1 GUARD BINDING: %s had the highest pooled (%.4f) but was rejected '
                   'for slope collapse; selecting %s (pooled %.4f, slope_med %.3f) instead'
                   % (top.get('tag', '?'), top.get('pooled', float('nan')),
                      best['tag'], best['pooled'], best['slope_med']))
    return best, rejected


def main_effects(rows):
    """Main effects and all six two-way interactions on pooled ddG.
    An effect is REAL only if |effect| > EFFECT_MIN (AUTOPILOT D1)."""
    out = {}
    if len(rows) < 4:
        return out
    for f in 'ABCD':
        hi = [r['pooled'] for r in rows if r[f] == 1]
        lo = [r['pooled'] for r in rows if r[f] == 0]
        if hi and lo:
            out[f] = sum(hi) / len(hi) - sum(lo) / len(lo)
    fs = 'ABCD'
    for i in range(4):
        for j in range(i + 1, 4):
            f1, f2 = fs[i], fs[j]
            same = [r['pooled'] for r in rows if r[f1] == r[f2]]
            diff = [r['pooled'] for r in rows if r[f1] != r[f2]]
            if same and diff:
                out[f1 + f2] = (sum(same) / len(same) - sum(diff) / len(diff)) / 2.0
    return out


def write_tsv(rows):
    d = os.path.dirname(TSV)
    if not os.path.isdir(d):
        os.makedirs(d)
    existing = set()
    if os.path.exists(TSV):
        with open(TSV) as f:
            for line in f:
                existing.add(line.split('\t')[0])
    new = 0
    with open(TSV, 'a') as f:
        if not existing:
            f.write('\t'.join(TSV_COLS) + '\n')
        for r in rows:
            if r['tag'] in existing:      # never write a duplicate run_tag
                continue
            f.write('\t'.join([r['tag'], 'done', r.get('node_class', '?'),
                               str(r['seed']), '15', '?', '%.4f' % r.get('pooled', float('nan')),
                               '%.4f' % r.get('pp', float('nan')), '?',
                               '%.4f' % r.get('std_b', float('nan')),
                               '%.3f' % r.get('slope_med', float('nan')),
                               '%.3f' % r.get('slope_min', float('nan')),
                               '%.4f' % r.get('designed_pooled', float('nan')), '?',
                               'A=%d B=%d C=%d D=%d' % (r['A'], r['B'], r['C'], r['D'])]) + '\n')
            new += 1
    return new


# ---------------------------------------------------------------- self audit
def self_audit(st, rows):
    fails = []
    tags = [r['tag'] for r in rows]
    if len(tags) != len(set(tags)):
        fails.append('duplicate run_tag in results')
    for r in rows:
        if not (0.0 <= r.get('pooled', -1) <= 1.0):
            fails.append('pooled out of [0,1]: %s' % r['tag'])
        if r.get('std_b', 1) <= 0:
            fails.append('std_b <= 0: %s' % r['tag'])
    rc, out = sh('df -h %s | tail -1' % ROOT)
    m = re.search(r'(\d+)%', out)
    if m and int(m.group(1)) > 85:
        fails.append('disk above 85%%: %s%%' % m.group(1))
    rc, out = sh('git remote -v')
    if 'shaharec' in out:
        fails.append('origin resolves to shaharec -- ABORT')
    return fails


# ---------------------------------------------------------------- the pass
def one_pass(st, go):
    jobs = squeue()
    running = [j for j in jobs if j['state'] == 'RUNNING']
    pending = [j for j in jobs if j['state'] == 'PENDING']
    log('queue: %d running, %d pending' % (len(running), len(pending)))

    rows = collect_results()
    n_new = write_tsv(rows)
    if n_new:
        log('recorded %d newly completed cell(s); %d total' % (n_new, len(rows)))

    fails = self_audit(st, rows)
    if len(fails) >= 2:
        halt(st, 'H2', 'two self-audit checks failed together: %s' % '; '.join(fails),
             'routine self-audit during %s' % st['state'],
             ['a partially written results file', 'a job wrote an out-of-range metric',
              'the git remote was changed outside this driver'])
    for f in fails:
        log('audit: %s' % f)

    st['cells_done'] = len(rows)
    save_state(st)

    # H4: a result contradicting a recorded fact by more than 3 sigma
    for r in rows:
        if r.get('pooled', 0) > CEILING + 0.05:
            halt(st, 'H4',
                 'cell %s reports pooled ddG %.4f, above the measured ceiling %.3f by more '
                 'than 3 sigma' % (r['tag'], r['pooled'], CEILING),
                 'calib_diag on its eval CSV',
                 ['the eval scored the wrong split', 'a checkpoint from another run was loaded',
                  'the affine calibration was applied twice'])

    if len(rows) < 16:
        log('S3: %d/48 cells scored; waiting' % len(rows))
        return st

    # ---- D1 ----
    eff = main_effects(rows)
    real = {k: v for k, v in eff.items() if abs(v) > EFFECT_MIN}
    log('D1 effects: ' + ', '.join('%s=%+.4f%s' % (k, v, '*' if abs(v) > EFFECT_MIN else '')
                                   for k, v in sorted(eff.items())))
    mains = {k: v for k, v in real.items() if len(k) == 1}
    best, rejected = select_best(rows)
    if best is None:
        halt(st, 'H2',
             'every one of the %d scored cells has median slope a_p < %.2f, so D1 has no '
             'eligible cell: selecting on pooled alone would pick a slope-collapsed cell, '
             'which AUTOPILOT.md D1 forbids' % (len(rows), SLOPE_MED_MIN),
             'D1 slope guard over all scored cells',
             ['the anchor weight is too high across the whole factorial',
              'calib_diag slope parsing broke and every slope_med is missing',
              'the eval scored a split with too little within-protein range'])
    log('D1 selection: %d/%d cells eligible (slope_med >= %.2f), %d rejected'
        % (len(rows) - len(rejected), len(rows), SLOPE_MED_MIN, len(rejected)))

    if mains:
        choice, why = best['tag'], 'real main effect(s): %s' % ', '.join(sorted(mains))
    elif real:
        choice, why = best['tag'], 'no real main effect; real interaction(s): %s' % ', '.join(sorted(real))
    else:
        choice = 'calib_ctrl'
        why = ('F1: no effect exceeds EFFECT_MIN=%.4f. Calibration levers do not move '
               'pooled beyond noise -- a genuine negative result and a thesis finding.'
               % EFFECT_MIN)
    decide(st, 'D1', choice,
           why + ' | best pooled %.4f slope_med %.3f (%s) | slope guard >= %.2f rejected %d cell(s): %s'
           % (best['pooled'], best['slope_med'], best['tag'], SLOPE_MED_MIN, len(rejected),
              ', '.join('%s(pooled %.4f, slope %s)'
                        % (r.get('tag', '?'), r.get('pooled', float('nan')),
                           'missing' if (sm is None or sm != sm) else '%.3f' % sm)
                        for r, sm in rejected) or 'none'))
    st['state'] = 'S5_BUILD_INFO'
    save_state(st)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--go', action='store_true', help='live; default is dry run')
    ap.add_argument('--once', action='store_true')
    A = ap.parse_args()
    lock = os.path.expanduser('~/auto/.autopilot.lock')
    d = os.path.dirname(lock)
    if not os.path.isdir(d):
        os.makedirs(d)
    # mkdir is atomic, so it excludes a second copy -- but a crash leaves the directory
    # behind and would block every restart forever. So the lock records its pid and a
    # stale lock (no such process) is reclaimed. Verified necessary: the first launch
    # died and its lock blocked the wrapper's restart.
    pidfile = os.path.join(lock, 'pid')
    try:
        os.mkdir(lock)
    except OSError:
        holder = None
        try:
            with open(pidfile) as f:
                holder = int(f.read().strip())
        except Exception:
            pass
        alive = False
        if holder:
            try:
                os.kill(holder, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            print('autopilot pid %d holds %s -- refusing to start a second' % (holder, lock))
            sys.exit(1)
        log('reclaiming stale lock (holder %s is gone)' % holder)
        try:
            os.remove(pidfile)
        except Exception:
            pass
    try:
        with open(pidfile, 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass
    try:
        st = load_state()
        log('start; state=%s go=%s' % (st['state'], A.go))
        while True:
            st = one_pass(st, A.go)
            if A.once or st['state'] in ('S9_REPORT', 'DONE'):
                break
            time.sleep(POLL_SECONDS)
    finally:
        try:
            os.remove(pidfile)
        except Exception:
            pass
        try:
            os.rmdir(lock)
        except Exception:
            pass


if __name__ == '__main__':
    main()
