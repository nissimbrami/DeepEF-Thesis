"""D2 zero-variance guard -- test.

A lever whose driving feature has zero variance on the eval set must be logged
'untestable-here' and REFUSED entry to the factorial. It must never be scored and
given a zero: that is a manufactured negative.

T1  synthetic lever, driving feature CONSTANT  -> refused
T2  synthetic lever, driving feature VARYING   -> admitted
T3  the real case: 28 ligand-free/metal-free proteins -> --ligand_nodes refused
T4  driving column ABSENT entirely             -> refused (absent != constant)
T5  refusal is recorded in state + UNTESTABLE.md, and is idempotent
T6  a refused lever contributes NO row to the factorial (no zero is averaged in)

Run:  python scripts/gate_d2_untestable.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autopilot as AP

FAILED = []


def check(name, cond, detail=''):
    if cond:
        print('[PASS] %s' % name)
    else:
        print('[FAIL] %s  %s' % (name, detail))
        FAILED.append(name)


def _rows(vals, col='n_het_residues'):
    out = []
    for i, v in enumerate(vals):
        r = {'tag': 'p3_a0_d0_s0_D0_x_seed%d' % i, 'pooled': 0.59}
        if v is not None:
            r[col] = v
        out.append(r)
    return out


def main():
    AP.LEVER_DRIVERS['synthetic_zero'] = (('synth_feature',), 'synthetic test lever')

    # T1 -- constant driving feature
    rows = _rows([0.0] * 28, col='synth_feature')
    ok, why = AP.lever_testable('synthetic_zero', rows)
    check('T1 constant driving feature -> refused', ok is False, why)
    check('T1 reason says untestable-here', 'untestable-here' in why, why)

    # T2 -- varying driving feature
    rows_v = _rows([0.0] * 27 + [3.0], col='synth_feature')
    ok_v, why_v = AP.lever_testable('synthetic_zero', rows_v)
    check('T2 varying driving feature -> admitted', ok_v is True, why_v)

    # T3 -- the real case
    real = []
    for i in range(28):
        real.append({'tag': 't%d' % i, 'n_het_residues': 0, 'n_metal_residues': 0})
    ok_l, why_l = AP.lever_testable('ligand_nodes', real)
    check('T3 ligand_nodes on 28 ligand-free proteins -> refused', ok_l is False, why_l)

    # T4 -- column absent entirely (absent != constant)
    ok_a, why_a = AP.lever_testable('synthetic_zero', _rows([None] * 28))
    check('T4 driving column absent -> refused', ok_a is False, why_a)
    check('T4 distinguishes absent from constant', 'NOT constant' in why_a, why_a)

    # T5 -- recorded, and idempotent
    tmp = tempfile.mkdtemp()
    AP.UNTESTABLE_MD = os.path.join(tmp, 'UNTESTABLE.md')
    AP.STATE = os.path.join(tmp, 'state.json')
    st = {'decisions': [], 'blocked': []}
    logged = []
    a1 = AP.admit_lever(st, 'synthetic_zero', rows, log_fn=logged.append)
    a2 = AP.admit_lever(st, 'synthetic_zero', rows, log_fn=logged.append)
    check('T5 admit_lever refuses', a1 is False and a2 is False)
    check('T5 recorded once in state (idempotent)',
          len([e for e in st.get('untestable', []) if e['lever'] == 'synthetic_zero']) == 1,
          str(st.get('untestable')))
    body = open(AP.UNTESTABLE_MD).read() if os.path.exists(AP.UNTESTABLE_MD) else ''
    check('T5 written to UNTESTABLE.md', 'synthetic_zero' in body, body[:200])
    check('T5 file states these are not negatives', 'manufactured' in body, body[:200])

    # T6 -- a refused lever must contribute NO scored row
    admitted = [r for r in rows if AP.admit_lever(st, 'synthetic_zero', rows,
                                                  log_fn=lambda m: None)]
    check('T6 refused lever contributes no rows to the factorial', admitted == [],
          str(admitted[:2]))

    # T7 -- an unregistered lever is not silently blocked
    ok_u, why_u = AP.lever_testable('some_other_lever', rows)
    check('T7 unregistered lever passes through', ok_u is True, why_u)

    print('')
    if FAILED:
        print('D2 GUARD GATE FAILED: %s' % ', '.join(FAILED))
        return 1
    print('=' * 62)
    print('D2 ZERO-VARIANCE GUARD PASSED. A lever with no variance on the eval set')
    print('is refused entry and logged untestable-here, never scored as a zero.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
