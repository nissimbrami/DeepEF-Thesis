"""Patch autopilot.py: D1 slope guard + non-empty-CSV assertion.

VERBATIM ANCHORS ONLY. Every anchor is asserted to match EXACTLY ONCE before any write.
Refuses to write if the file is already patched or if any anchor count != 1.
"""
import io, os, sys

P = '/home/nissimb/DeepPEF/scripts/autopilot.py'
src = io.open(P, encoding='utf-8').read()
orig = src

if 'SLOPE_MED_MIN' in src:
    print('ALREADY PATCHED -- no write')
    sys.exit(0)

# ---------------------------------------------------------------- anchors
A1 = """REF_POOLED, REF_PP, REF_STD_B = 0.591, 0.731, 0.223
CEILING = 0.711
"""
R1 = """REF_POOLED, REF_PP, REF_STD_B = 0.591, 0.731, 0.223
CEILING = 0.711
# AUTOPILOT.md D1: "A cell that raises pooled while collapsing slope below 0.3 median is
# not selected, even if its pooled is highest -- that is the anchor's known failure mode
# and selecting it would defeat the purpose." The driver used to select on pooled alone,
# so it could and would pick exactly that cell. a_p is a within-protein slope: it does not
# cancel in ddG, so it is legitimately measurable here, and it is a separate calibration
# channel (corr(a_p, per-protein PCC) = +0.571; mean_rel_SASA vs a_p r = +0.714, 10/10).
SLOPE_MED_MIN = 0.3
MIN_EVAL_CSV_BYTES = 1000    # a finished training run is not a result
"""

# 2) empty-CSV assertion in collect_results
A2 = """        cell = parse_tag(tag)
        if not cell:
            continue
        stats = calib_diag(os.path.join(d, f))
        if stats and 'pooled' in stats:
            cell.update(stats)
            rows.append(cell)
    return rows
"""
R2 = """        cell = parse_tag(tag)
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
    \"\"\"D1 selection under the documented slope guard.

    Returns (best, rejected) where `rejected` lists (row, slope_med) for every cell
    excluded for slope collapse. A cell is ELIGIBLE only if its median a_p is known and
    >= SLOPE_MED_MIN. Ranking among eligible cells is by pooled, as documented.

    Rejection is LOUD: the whole point of the guard is that a pooled-winning cell got
    thrown out, and that must appear in the log, not be silently skipped.

    A cell with no slope_med is NOT eligible -- unknown slope is not a passing slope.
    If every cell collapses, this returns (None, rejected) and the caller halts rather
    than quietly selecting a collapsed cell.
    \"\"\"
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
"""

# 3) D1 selection site
A3 = """    mains = {k: v for k, v in real.items() if len(k) == 1}
    best = max(rows, key=lambda r: r['pooled'])

    if mains:"""
R3 = """    mains = {k: v for k, v in real.items() if len(k) == 1}
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

    if mains:"""

# 4) evidence string must carry the guard
A4 = """    decide(st, 'D1', choice, why + ' | best pooled %.4f (%s)' % (best['pooled'], best['tag']))"""
R4 = """    decide(st, 'D1', choice,
           why + ' | best pooled %.4f slope_med %.3f (%s) | slope guard >= %.2f rejected %d cell(s): %s'
           % (best['pooled'], best['slope_med'], best['tag'], SLOPE_MED_MIN, len(rejected),
              ', '.join('%s(pooled %.4f, slope %s)'
                        % (r.get('tag', '?'), r.get('pooled', float('nan')),
                           'missing' if (sm is None or sm != sm) else '%.3f' % sm)
                        for r, sm in rejected) or 'none'))"""

for name, a in (('A1', A1), ('A2', A2), ('A3', A3), ('A4', A4)):
    n = src.count(a)
    print('anchor %s matches %d time(s)' % (name, n))
    assert n == 1, 'ANCHOR %s MATCHED %d TIMES -- REFUSING TO WRITE' % (name, n)

for a, r in ((A1, R1), (A2, R2), (A3, R3), (A4, R4)):
    src = src.replace(a, r, 1)

assert src != orig
io.open(P + '.prepatch_d1', 'w', encoding='utf-8').write(orig)
io.open(P, 'w', encoding='utf-8').write(src)
print('WROTE %s (backup %s.prepatch_d1)' % (P, P))
