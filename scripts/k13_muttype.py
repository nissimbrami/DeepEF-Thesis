"""K13: is a_p driven by a chemical CLASS of mutation?

An agent reported that mutations TO hydrophobic residues are compressed ~49% harder than to
polar ones (within-protein slope 0.334 vs 0.497, corr with Kyte-Doolittle -0.833, 10/10
checkpoints), but that rank accuracy degrades in lockstep, making it LOST INFORMATION rather
than a rescalable calibration error. That distinction decides whether a lever could fix it,
so it is worth verifying.

The eval CSV carries no mutation code, so the destination residue is recovered by joining on
deltaG against the mutation_datasets file, whose `name` encodes the mutation (e.g. 1PSE.pdb_A42W).
Row counts differ between the two (the eval uses a filtered subset), so the join is on VALUE,
keyed to 6 decimals, and any deltaG appearing more than once on either side is DROPPED rather
than guessed at.
"""
import csv, io, math, os, glob, collections, json, re

KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',
              [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
MUT = re.compile(r'_([A-Z])(\d+)([A-Z])$')
DIRS = ('data_fixed/mutation_datasets', 'data/Processed_K50_dG_datasets/mutation_datasets')

def dest_map(protein):
    """deltaG (6dp) -> destination residue, for unambiguous rows only."""
    for d in DIRS:
        f = os.path.join(d, protein + '.csv')
        if not os.path.exists(f):
            continue
        seen = collections.defaultdict(set)
        for r in csv.DictReader(io.open(f, encoding='utf-8', errors='replace')):
            m = MUT.search(r['name'].strip())
            if not m or not r.get('deltaG'):
                continue
            try: k = round(float(r['deltaG']), 6)
            except ValueError: continue
            seen[k].add(m.group(3))
        return {k: list(v)[0] for k, v in seen.items() if len(v) == 1}
    return {}

def slope(xs, ys):
    n = len(xs)
    if n < 8: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((a-mx)**2 for a in xs)
    if sxx <= 0: return None
    return sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sxx

def spear(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        for pos, i in enumerate(s): r[i] = pos
        return r
    a, b = rank(xs), rank(ys); n = len(a)
    ma, mb = sum(a)/n, sum(b)/n
    sab = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    saa = sum((x-ma)**2 for x in a); sbb = sum((y-mb)**2 for y in b)
    return sab/math.sqrt(saa*sbb) if saa > 0 and sbb > 0 else float('nan')

def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else float('nan')

out = {}
for f in sorted(glob.glob('eval_results/abl_*.csv'))[:10]:
    by = collections.defaultdict(list)      # dest residue -> (true, pred)
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        per[r['protein']].append(r)
    matched = 0
    for prot, rows in per.items():
        dm = dest_map(prot)
        if not dm: continue
        for r in rows:
            k = round(float(r['deltaG']), 6)
            d = dm.get(k)
            if not d or d not in KD: continue
            dd = float(r['ddG'])
            if prot == '2K5H': dd -= 3.0824
            by[d].append((dd, float(r['pred_ddG'])))
            matched += 1
    if matched < 500: continue
    res = {}
    for d, v in by.items():
        s = slope([a for a, _ in v], [b for _, b in v])
        if s is None: continue
        res[d] = dict(n=len(v), slope=s, spearman=spear([a for a,_ in v],[b for _,b in v]))
    if len(res) >= 10:
        ds = sorted(res)
        out[os.path.basename(f)] = dict(matched=matched, per_residue=res,
            corr_slope_kd=pear([KD[d] for d in ds], [res[d]['slope'] for d in ds]),
            corr_rho_kd=pear([KD[d] for d in ds], [res[d]['spearman'] for d in ds]))

if out:
    ks = list(out)
    print('checkpoints analysed: %d' % len(ks))
    print('mean corr(slope, Kyte-Doolittle) = %+.4f' % (sum(out[k]['corr_slope_kd'] for k in ks)/len(ks)))
    print('mean corr(spearman, KD)          = %+.4f' % (sum(out[k]['corr_rho_kd'] for k in ks)/len(ks)))
    print('sign-consistent (slope<0): %d/%d' % (sum(1 for k in ks if out[k]['corr_slope_kd'] < 0), len(ks)))
    ref = out[ks[0]]['per_residue']
    print('\n%-4s %6s %8s %8s %8s' % ('res', 'n', 'KD', 'slope', 'spearman'))
    for d in sorted(ref, key=lambda x: KD[x]):
        print('%-4s %6d %8.1f %8.4f %8.4f' % (d, ref[d]['n'], KD[d], ref[d]['slope'], ref[d]['spearman']))
    json.dump(out, open('results/k13_muttype.json', 'w'), indent=1)
    print('\nwrote results/k13_muttype.json')
else:
    print('join failed - too few matched rows')
