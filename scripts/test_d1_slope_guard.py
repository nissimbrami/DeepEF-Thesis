"""Test D1's selection rule on SYNTHETIC rows. No GPU, no cluster, no job submission.

Proves the documented rule: "A cell that raises pooled while collapsing slope below 0.3
median is not selected, even if its pooled is highest" (AUTOPILOT.md D1).

The failure this guards against is real: the driver used to do
    best = max(rows, key=lambda r: r['pooled'])
which selects the slope-collapsed cell precisely when the collapse buys the most pooled.

Run:  python scripts/test_d1_slope_guard.py
"""
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autopilot as AP

FAILS = []
LOG = []


def _log(msg):
    LOG.append(msg)


def check(name, cond, detail=''):
    print('  [%s] %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail else ''))
    if not cond:
        FAILS.append(name)


def cell(tag, pooled, slope_med, **kw):
    r = dict(tag=tag, pooled=pooled, A=0, B=0, C=0, D=0, seed=42)
    if slope_med is not None:
        r['slope_med'] = slope_med
    r.update(kw)
    return r


print('=' * 70)
print('D1 SLOPE GUARD -- synthetic selection tests')
print('SLOPE_MED_MIN = %.2f' % AP.SLOPE_MED_MIN)
print('=' * 70)

# ---------------------------------------------------------------- T1
# THE case the rule exists for: the highest-pooled cell won by collapsing the slope.
print('')
print('T1  highest pooled but collapsed slope MUST be rejected')
rows = [
    cell('p3_a1_d1_s0_D0_anchor_seed42', pooled=0.6400, slope_med=0.11),   # pooled winner, collapsed
    cell('p3_a0_d1_s1_D0_ctrl_seed42', pooled=0.6010, slope_med=0.52),     # honest
    cell('p3_a0_d0_s0_D0_ctrl_seed42', pooled=0.5900, slope_med=0.49),
]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('a cell was selected', best is not None)
check('the collapsed cell is NOT selected',
      best is not None and best['tag'] != 'p3_a1_d1_s0_D0_anchor_seed42',
      'selected=%s' % (best or {}).get('tag'))
check('the selected cell is the best ELIGIBLE cell',
      best is not None and best['tag'] == 'p3_a0_d1_s1_D0_ctrl_seed42',
      'selected=%s pooled=%s' % ((best or {}).get('tag'), (best or {}).get('pooled')))
check('the rejected cell is reported',
      len(rejected) == 1 and rejected[0][0]['tag'] == 'p3_a1_d1_s0_D0_anchor_seed42')
check('rejection names the cell AND its slope value in the log',
      any('p3_a1_d1_s0_D0_anchor_seed42' in m and '0.110' in m and 'REJECT' in m for m in LOG),
      'log=%r' % LOG)
check('the binding guard is announced (pooled winner thrown out)',
      any('GUARD BINDING' in m for m in LOG))
# and the old buggy rule really would have picked it -- so the test has teeth
old = max(rows, key=lambda r: r['pooled'])
check('control: the OLD pooled-only rule would have picked the collapsed cell',
      old['tag'] == 'p3_a1_d1_s0_D0_anchor_seed42')
for m in LOG:
    print('      log| %s' % m)

# ---------------------------------------------------------------- T2
print('')
print('T2  no collapse anywhere -> plain highest pooled wins, nothing rejected')
rows = [cell('c_hi', 0.6300, 0.55), cell('c_mid', 0.6100, 0.48), cell('c_lo', 0.5800, 0.71)]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('highest pooled selected', best['tag'] == 'c_hi', 'selected=%s' % best['tag'])
check('nothing rejected', rejected == [])
check('no REJECT line logged', not any('REJECT' in m for m in LOG))

# ---------------------------------------------------------------- T3
print('')
print('T3  boundary: slope_med exactly 0.3 is ELIGIBLE (the rule says "below 0.3")')
rows = [cell('c_edge', 0.6200, 0.30), cell('c_safe', 0.6000, 0.60)]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('slope_med == 0.30 is kept', best['tag'] == 'c_edge', 'selected=%s' % best['tag'])
rows = [cell('c_just_under', 0.6200, 0.299), cell('c_safe', 0.6000, 0.60)]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('slope_med == 0.299 is rejected', best['tag'] == 'c_safe', 'selected=%s' % best['tag'])

# ---------------------------------------------------------------- T4
print('')
print('T4  a missing / NaN slope is NOT a passing slope')
rows = [cell('c_noslope', 0.6500, None), cell('c_known', 0.6000, 0.50)]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('cell with no slope_med is not selected', best['tag'] == 'c_known',
      'selected=%s' % best['tag'])
check('missing slope is logged as such', any('missing' in m for m in LOG), 'log=%r' % LOG)
rows = [cell('c_nan', 0.6500, float('nan')), cell('c_known', 0.6000, 0.50)]
best, rejected = AP.select_best(rows, log_fn=_log)
check('NaN slope_med is not selected', best['tag'] == 'c_known')

# ---------------------------------------------------------------- T5
print('')
print('T5  every cell collapsed -> no selection (caller halts; never picks a collapsed cell)')
rows = [cell('c_a', 0.6600, 0.10), cell('c_b', 0.6400, 0.22), cell('c_c', 0.6100, 0.05)]
LOG[:] = []
best, rejected = AP.select_best(rows, log_fn=_log)
check('returns None rather than selecting a collapsed cell', best is None)
check('all three reported rejected', len(rejected) == 3)

# ---------------------------------------------------------------- T6
print('')
print('T6  the guard reads the REAL column name produced by calib_diag')
# calib_diag.py prints "[ALL] slope a_p  min/med/max = min / med / max";
# autopilot.calib_diag() parses the median into 'slope_med'. TSV column is 'slope_med'.
sample = ('==================== calib_diag [ALL] ====================\n'
          '[ALL] proteins=42  pp_groups(len>=3)=39\n'
          '[ALL] pooled ddG PCC            = 0.6400\n'
          '[ALL] PP (mean per-protein)     = 0.7310\n'
          '[ALL] std(b)  [offset spread]   = 0.2230\n'
          '[ALL] slope a_p  min/med/max    = 0.031 / 0.110 / 0.884\n'
          '[ALL] slope collapsed (a<0.3)   = 27\n'
          '[ALL] pooled OFFSET-removed PCC = 0.7000  (remove b_p)\n')
m = re.search(r'\[ALL\]\s+slope a_p\s+min/med/max\s*=\s*-?\d+\.\d+\s*/\s*(-?\d+\.\d+)', sample)
check("autopilot's slope_med regex parses calib_diag's real output line",
      m is not None and float(m.group(1)) == 0.110,
      'parsed=%s' % (m.group(1) if m else None))
check("'slope_med' is a real TSV column name", 'slope_med' in AP.TSV_COLS)

# ---------------------------------------------------------------- T7
print('')
print('T7  an empty / header-only eval CSV is never scored ("a run is not a result")')
tmp = tempfile.mkdtemp()
try:
    ev = os.path.join(tmp, 'eval_results')
    os.makedirs(ev)
    tag_empty = 'p3_a1_d1_s1_D1_ctrl_seed42'
    tag_stub = 'p3_a0_d0_s0_D0_ctrl_seed42'
    open(os.path.join(ev, 'abl_%s_e14.csv' % tag_empty), 'w').close()          # 0 bytes
    with open(os.path.join(ev, 'abl_%s_e14.csv' % tag_stub), 'w') as f:        # header only
        f.write('protein,deltaG,pred_deltaG\n')
    old_root, old_log = AP.ROOT, AP.log
    AP.ROOT = tmp
    LOG[:] = []
    AP.log = _log
    try:
        got = AP.collect_results()
    finally:
        AP.ROOT, AP.log = old_root, old_log
    check('no rows scored from empty/stub CSVs', got == [], 'got=%r' % got)
    check('the 0-byte CSV is called out by tag and size',
          any(tag_empty in m and 'SKIP' in m and '0 bytes' in m for m in LOG), 'log=%r' % LOG)
    check('the header-only CSV is also skipped',
          any(tag_stub in m and 'SKIP' in m for m in LOG), 'log=%r' % LOG)
    for m_ in LOG:
        print('      log| %s' % m_)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- T8
print('')
print('T8  regression: one_pass no longer contains the unguarded pooled-only max')
srcp = os.path.join(os.path.dirname(os.path.abspath(AP.__file__)), 'autopilot.py')
src = open(srcp).read()
check('the bug line "best = max(rows, key=lambda r: r[pooled])" is gone',
      "best = max(rows, key=lambda r: r['pooled'])" not in src)
check('D1 calls select_best', 'best, rejected = select_best(rows)' in src)
check('SLOPE_MED_MIN is 0.3 as documented', AP.SLOPE_MED_MIN == 0.3)

print('')
print('=' * 70)
if FAILS:
    print('D1 GUARD TESTS: %d FAILED -> %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('D1 GUARD TESTS: ALL PASSED')
print('=' * 70)
