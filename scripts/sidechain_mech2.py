# -*- coding: utf-8 -*-
"""
sidechain_mech2.py -- follow-ups forced by sidechain_mech.py.

Three problems the first pass exposed:

 P1  The within-burial-bucket slopes (0.17-0.23) are FAR below the global per-destination
     slopes (0.27-0.57).  That is because the global fit pools mutations across proteins
     and so absorbs the between-protein offset b_p as signal, while the bucket fit does
     not.  Fix: subtract each protein's own mean from both true and pred BEFORE fitting
     (a within-protein centring), so every slope reported is a genuine within-protein
     slope and the buckets are comparable with each other AND with the global number.

 P2  var(true ddG) is 1.49 buried vs 0.43 exposed -- a 3.4x spread.  A slope is not
     range-restricted (OLS slope is scale-free in y-variance), but the SPEARMAN is,
     so the burial comparison of spearman must be read against that.  Report var and
     also report the CORRELATION (which IS attenuated by restriction) alongside the
     slope so the reader can tell them apart.

 P3  The volume test at n=20 destinations is confounded by KD (r=+0.047 overall, so
     nominally orthogonal) BUT the hydrophobic set spans a huge volume range while the
     polar set does too -- so a cleaner test is available: hold CHEMICAL CLASS fixed
     and ask whether volume still predicts slope WITHIN class.  If bulk is the
     mechanism, W(227) should be far worse than A(88) among hydrophobics, and
     R(173)/K(168) far worse than S(89)/N(114) among polars.

 Plus: the DELTA-volume and DELTA-hydropathy of the mutation (dst minus src) rather than
 the destination alone.  A packing failure is about the CHANGE in bulk at that site.
"""
import csv, io, os, re, glob, json, math, collections, random

KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = {'G':60.1,'A':88.6,'S':89.0,'C':108.5,'D':111.1,'P':112.7,'N':114.1,'T':116.1,
       'E':138.4,'V':140.0,'Q':143.8,'H':153.2,'M':162.9,'I':166.7,'L':166.7,'K':168.6,
       'R':173.4,'F':189.9,'Y':193.6,'W':227.8}
MAXSASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
           'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
           'Y':263,'V':174}
THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
             'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
             'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
PHOBIC = set('AVILMFWCY')
POLAR = set('DEKRNQSTHG')

MUT = re.compile(r'_([A-Z])(\d+)([A-Z])(?:_|$)')
DIRS = ('data_fixed/mutation_datasets', 'data/Processed_K50_dG_datasets/mutation_datasets')
PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'


def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else float('nan')

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0]*len(v); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
        avg = (i+j)/2.0
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def spear(x, y): return pear(rank(x), rank(y))

def partial(x, y, z):
    rxy, rxz, ryz = pear(x, y), pear(x, z), pear(y, z)
    den = math.sqrt(max(1e-12, (1-rxz**2)*(1-ryz**2)))
    return (rxy - rxz*ryz)/den

def slope(xs, ys):
    n = len(xs)
    if n < 8: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((a-mx)**2 for a in xs)
    if sxx <= 0: return None
    return sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sxx

def var(v):
    n = len(v)
    if n < 2: return 0.0
    m = sum(v)/n
    return sum((a-m)**2 for a in v)/(n-1)

_SC = {}
def rel_sasa(protein):
    if protein in _SC: return _SC[protein]
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    p = os.path.join(PDB_DIR, protein + '.pdb')
    if not os.path.exists(p):
        _SC[protein] = {}; return {}
    st = PDBParser(QUIET=True).get_structure(protein, p)
    model = st[0]
    ShrakeRupley().compute(model, level='R')
    out = {}
    for r in model.get_residues():
        if r.get_id()[0] != ' ': continue
        one = THREE2ONE.get(r.get_resname().strip())
        if not one or not MAXSASA.get(one): continue
        out[r.get_id()[1]] = (one, r.sasa/MAXSASA[one])
    _SC[protein] = out
    return out

def mut_map(protein):
    for d in DIRS:
        f = os.path.join(d, protein + '.csv')
        if not os.path.exists(f): continue
        seen = collections.defaultdict(set)
        for r in csv.DictReader(io.open(f, encoding='utf-8', errors='replace')):
            m = MUT.search(r['name'].strip())
            if not m or not r.get('deltaG'): continue
            try: k = round(float(r['deltaG']), 6)
            except ValueError: continue
            seen[k].add((m.group(1), int(m.group(2)), m.group(3)))
        return dict((k, list(v)[0]) for k, v in seen.items() if len(v) == 1)
    return {}


def load(path):
    """Return list of records with per-protein CENTRED true/pred (P1 fix)."""
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        per[r['protein']].append(r)
    recs = []
    for prot, rows in per.items():
        mm = mut_map(prot)
        if not mm: continue
        sm = rel_sasa(prot)
        raw = []
        for r in rows:
            k = round(float(r['deltaG']), 6)
            hit = mm.get(k)
            if not hit: continue
            src, pos, dst = hit
            if dst not in KD or src not in KD: continue
            dd = float(r['ddG'])
            if prot == '2K5H': dd -= 3.0824
            ent = sm.get(pos)
            if ent is None or ent[0] != src: continue
            raw.append((prot, src, pos, dst, dd, float(r['pred_ddG']), ent[1]))
        if len(raw) < 20: continue
        mt = sum(a[4] for a in raw)/len(raw)
        mp = sum(a[5] for a in raw)/len(raw)
        for prot_, src, pos, dst, t, p, rs in raw:
            recs.append(dict(prot=prot_, src=src, pos=pos, dst=dst,
                             t=t-mt, p=p-mp, rel=rs,
                             bucket='buried' if rs < 0.25 else ('exposed' if rs > 0.50 else 'mid')))
    return recs


def fit(recs):
    if len(recs) < 8: return None
    t = [r['t'] for r in recs]; p = [r['p'] for r in recs]
    s = slope(t, p)
    if s is None: return None
    return dict(n=len(recs), slope=s, pearson=pear(t, p), spearman=spear(t, p),
                var_true=var(t))


def main():
    files = sorted(glob.glob('eval_results/abl_*.csv'))[:10]
    per_ck = {}
    for f in files:
        recs = load(f)
        if len(recs) < 500: continue
        per_ck[os.path.basename(f)] = recs
    tags = sorted(per_ck)
    print('checkpoints: %d   mean matched (SASA-verified) rows: %d'
          % (len(tags), sum(len(per_ck[t]) for t in tags)//len(tags)))
    print('ALL slopes below are WITHIN-PROTEIN (per-protein mean removed from t and p).')
    print()

    print('=' * 78)
    print('P1  GLOBAL AND PER-DESTINATION SLOPE, WITHIN-PROTEIN CENTRED')
    print('=' * 78)
    g = [fit(per_ck[t]) for t in tags]
    print('global within-protein: slope %.4f  pearson %.4f  spearman %.4f'
          % (sum(x['slope'] for x in g)/len(g), sum(x['pearson'] for x in g)/len(g),
             sum(x['spearman'] for x in g)/len(g)))
    print()

    # per-destination, centred
    dest = {}
    for t in tags:
        by = collections.defaultdict(list)
        for r in per_ck[t]: by[r['dst']].append(r)
        dest[t] = dict((d, fit(v)) for d, v in by.items() if fit(v))
    ds = sorted(dest[tags[0]])
    print('%-4s %7s %7s %8s %8s %9s %9s' % ('res','n','KD','vol','slope','pearson','spearman'))
    for d in sorted(ds, key=lambda x: -VOL[x]):
        vals = [dest[t][d] for t in tags if d in dest[t]]
        print('%-4s %7d %7.1f %8.1f %8.4f %9.4f %9.4f'
              % (d, sum(v['n'] for v in vals)//len(vals), KD[d], VOL[d],
                 sum(v['slope'] for v in vals)/len(vals),
                 sum(v['pearson'] for v in vals)/len(vals),
                 sum(v['spearman'] for v in vals)/len(vals)))
    print()
    for tgt in ('slope', 'spearman'):
        rk = [pear([KD[d] for d in ds], [dest[t][d][tgt] for d in ds]) for t in tags]
        rv = [pear([VOL[d] for d in ds], [dest[t][d][tgt] for d in ds]) for t in tags]
        pv = [partial([VOL[d] for d in ds], [dest[t][d][tgt] for d in ds],
                      [KD[d] for d in ds]) for t in tags]
        pk = [partial([KD[d] for d in ds], [dest[t][d][tgt] for d in ds],
                      [VOL[d] for d in ds]) for t in tags]
        print('centred %-9s r(KD)=%+.4f  r(vol)=%+.4f  part(vol|KD)=%+.4f  part(KD|vol)=%+.4f'
              % (tgt, sum(rk)/len(rk), sum(rv)/len(rv), sum(pv)/len(pv), sum(pk)/len(pk)))
    print()

    print('=' * 78)
    print('P3  VOLUME WITHIN CHEMICAL CLASS -- the clean bulk test')
    print('=' * 78)
    print('If BULK is the mechanism, volume must predict slope INSIDE each class.')
    for cname, cset in (('hydrophobic AVILMFWCY', PHOBIC), ('polar DEKRNQSTHG', POLAR)):
        sub = [d for d in ds if d in cset]
        rows_s, rows_r = [], []
        for t in tags:
            rows_s.append(pear([VOL[d] for d in sub], [dest[t][d]['slope'] for d in sub]))
            rows_r.append(pear([VOL[d] for d in sub], [dest[t][d]['spearman'] for d in sub]))
        neg = sum(1 for v in rows_s if v < 0)
        print('  %-22s n_res=%2d  r(slope,vol)=%+.4f [%.4f,%.4f]  r(spearman,vol)=%+.4f  sign<0: %d/%d'
              % (cname, len(sub), sum(rows_s)/len(rows_s), min(rows_s), max(rows_s),
                 sum(rows_r)/len(rows_r), neg, len(rows_s)))
        # print the members so the reader can see the spread
        print('     ' + '  '.join('%s(%.0f,%.3f)' % (d, VOL[d],
              sum(dest[t][d]['slope'] for t in tags)/len(tags))
              for d in sorted(sub, key=lambda x: VOL[x])))
    print()
    print('  class means:')
    for cname, cset in (('hydrophobic', PHOBIC), ('polar', POLAR)):
        sub = [d for d in ds if d in cset]
        v = [sum(dest[t][d]['slope'] for t in tags)/len(tags) for d in sub]
        print('    %-12s mean slope %.4f  mean vol %.1f' %
              (cname, sum(v)/len(v), sum(VOL[d] for d in sub)/len(sub)))
    print()

    print('=' * 78)
    print('P2  BURIAL, WITH THE RANGE-RESTRICTION CONTROL VISIBLE')
    print('=' * 78)
    print('%-9s %8s %8s %9s %9s %9s' % ('bucket','n','slope','pearson','spearman','var_true'))
    for b in ('buried','mid','exposed'):
        vals = [fit([r for r in per_ck[t] if r['bucket'] == b]) for t in tags]
        vals = [v for v in vals if v]
        print('%-9s %8d %8.4f %9.4f %9.4f %9.4f'
              % (b, sum(v['n'] for v in vals)//len(vals),
                 sum(v['slope'] for v in vals)/len(vals),
                 sum(v['pearson'] for v in vals)/len(vals),
                 sum(v['spearman'] for v in vals)/len(vals),
                 sum(v['var_true'] for v in vals)/len(vals)))
    print()
    print('--- burial x class, WITHIN-PROTEIN centred (the packing prediction) ---')
    print('the packing story predicts: buried BIG-HYDROPHOBIC is the WORST cell.')
    print('%-9s %-14s %8s %8s %9s %9s' % ('bucket','class','n','slope','pearson','var_true'))
    grid = {}
    for b in ('buried','mid','exposed'):
        for cname, cset in (('big-phobic FILMWY', set('FILMWY')),
                            ('small AGSVTCP', set('AGSVTCP')),
                            ('polar DEKRNQH', set('DEKRNQH'))):
            vals = [fit([r for r in per_ck[t] if r['bucket'] == b and r['dst'] in cset])
                    for t in tags]
            vals = [v for v in vals if v]
            if not vals: continue
            grid[(b, cname)] = (sum(v['slope'] for v in vals)/len(vals),
                                sum(v['pearson'] for v in vals)/len(vals))
            print('%-9s %-14s %8d %8.4f %9.4f %9.4f'
                  % (b, cname, sum(v['n'] for v in vals)//len(vals),
                     sum(v['slope'] for v in vals)/len(vals),
                     sum(v['pearson'] for v in vals)/len(vals),
                     sum(v['var_true'] for v in vals)/len(vals)))
    print()
    print('  INTERACTION: (polar - big-phobic) slope gap, per bucket:')
    for b in ('buried','mid','exposed'):
        if (b,'polar DEKRNQH') in grid and (b,'big-phobic FILMWY') in grid:
            print('    %-9s gap = %.4f  (polar %.4f - phobic %.4f)'
                  % (b, grid[(b,'polar DEKRNQH')][0]-grid[(b,'big-phobic FILMWY')][0],
                     grid[(b,'polar DEKRNQH')][0], grid[(b,'big-phobic FILMWY')][0]))
    print()

    print('=' * 78)
    print('P4  DELTA features: does the CHANGE in bulk/hydropathy explain it better?')
    print('=' * 78)
    print('bin mutations by d(vol) = vol[dst]-vol[src] and by d(KD), fit slope per bin')
    for name, keyf, edges in (
            ('d_volume', lambda r: VOL[r['dst']]-VOL[r['src']], (-120,-60,-20,20,60,120)),
            ('d_KD', lambda r: KD[r['dst']]-KD[r['src']], (-6,-3,-1,1,3,6))):
        print('  --- %s ---' % name)
        print('  %-16s %8s %8s %9s' % ('bin','n','slope','spearman'))
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            vals = [fit([r for r in per_ck[t] if lo <= keyf(r) < hi]) for t in tags]
            vals = [v for v in vals if v]
            if not vals: continue
            print('  [%6.1f,%6.1f) %8d %8.4f %9.4f'
                  % (lo, hi, sum(v['n'] for v in vals)//len(vals),
                     sum(v['slope'] for v in vals)/len(vals),
                     sum(v['spearman'] for v in vals)/len(vals)))
    print()
    # correlation of |residual| with d_vol and d_KD at the mutation level
    print('  mutation-level: corr(signed pred error, d_vol) and (d_KD), pooled centred')
    for t in tags[:3]:
        R = per_ck[t]
        err = [r['p']-r['t'] for r in R]
        dv = [VOL[r['dst']]-VOL[r['src']] for r in R]
        dk = [KD[r['dst']]-KD[r['src']] for r in R]
        print('    %-38s r(err,dvol)=%+.4f  r(err,dKD)=%+.4f  n=%d'
              % (t[:38], pear(err, dv), pear(err, dk), len(R)))

    json.dump(dict(dest=dict((t, dict((d, dest[t][d]) for d in dest[t])) for t in tags),
                   grid=dict(('%s|%s' % k, v) for k, v in grid.items())),
              open('results/sidechain_mech2.json', 'w'), indent=1)
    print('\nwrote results/sidechain_mech2.json')

main()
