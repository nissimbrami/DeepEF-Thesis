"""Read the nu x b sweep JSON and print the grid on the metric that matters."""
import json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'results/08_data/nu_sweep.json'
d = json.load(open(path))
rows = d['per_protein']
true = np.array([r['true'] for r in rows])
STD_TRUE = float(true.std())
NUS = [0.5, 0.55, 0.588, 0.62, 0.65]
BS = [4.97, 5.82]

print('file        : %s' % path)
print('checkpoint  : %s' % d['checkpoint'])
print('n proteins  : %d   ref_fix=%s' % (d['n'], d.get('ref_fix')))
print('std(true WT dG) = %.4f  <-- DEGENERATE ATTRACTOR' % STD_TRUE)

base = d['results']['base'] if 'base' in d['results'] else None
if base:
    print('\nBASELINE (no coil)      std(b_p)=%.4f  MAE=%.4f  std(predWT)=%.4f  n_under=%d/%d'
          % (base['std_err'], base['mae'], base['std_pred'], base['n_under'], base['n']))
cf = d['results'].get('coil_fixed_b')
if cf:
    print('coil_fixed_b (nu=0.5)   std(b_p)=%.4f  MAE=%.4f  std(predWT)=%.4f'
          % (cf['std_err'], cf['mae'], cf['std_pred']))

print('\n' + '=' * 78)
print('THE GRID -- std(b_p), the DISPERSION metric.  Success = below %.4f'
      % (base['std_err'] if base else float('nan')))
print('=' * 78)
hdr = '%-8s' % 'nu'
for b in BS:
    hdr += '   b=%.2f: std(b_p)  std(predWT)      MAE' % b
print(hdr)
grid = {}
for nu in NUS:
    line = '%-8.3f' % nu
    for b in BS:
        k = 'nu%.3f_b%.2f' % (nu, b)
        r = d['results'].get(k)
        if r is None:
            line += '  %-38s' % 'MISSING'
            continue
        grid[(nu, b)] = r
        line += '         %8.4f     %8.4f %8.4f' % (r['std_err'], r['std_pred'], r['mae'])
    print(line)

print('\n--- delta std(b_p) vs the NO-COIL baseline (negative = better dispersion) ---')
for nu in NUS:
    line = '%-8.3f' % nu
    for b in BS:
        r = grid.get((nu, b))
        line += '   b=%.2f %+8.4f' % (b, r['std_err'] - base['std_err']) if r else '   ---'
    print(line)

print('\n--- degeneracy check: std(predWT) relative to baseline %.4f ---' % base['std_pred'])
for (nu, b), r in sorted(grid.items()):
    frac = r['std_pred'] / base['std_pred']
    flag = 'COLLAPSING' if frac < 0.5 else ('shrinking' if frac < 0.85 else 'ok')
    print('  nu=%.3f b=%.2f  std(predWT)=%.4f  (%.0f%% of baseline)  %s'
          % (nu, b, r['std_pred'], 100 * frac, flag))

best = min(grid.items(), key=lambda kv: kv[1]['std_err'])
print('\nBEST CELL on dispersion: nu=%.3f b=%.2f  std(b_p)=%.4f  (baseline %.4f, delta %+.4f)'
      % (best[0][0], best[0][1], best[1]['std_err'], base['std_err'],
         best[1]['std_err'] - base['std_err']))
print('VERDICT: %s' % ('BEATS baseline' if best[1]['std_err'] < base['std_err']
                       else 'NO cell beats baseline'))
print('\nsign check (n_under / n) -- MAE == |mean(b_p)| only when all share a sign:')
for k in ['base', 'coil_fixed_b'] + ['nu%.3f_b%.2f' % (nu, b) for nu in NUS for b in BS]:
    r = d['results'].get(k)
    if r:
        print('   %-18s %d/%d under-predicted   MAE=%.4f |mean(b_p)|=%.4f'
              % (k, r['n_under'], r['n'], r['mae'], abs(r['bias'])))
