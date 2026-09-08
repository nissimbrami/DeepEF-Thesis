# -*- coding: utf-8 -*-
"""
sidechain_mech3.py -- volume is REFUTED (mech2: within-class r = -0.11 phobic / +0.15 polar,
sign flips).  So what IS the axis?  Test the remaining candidates head to head against KD,
all at n=20 destinations, all with partial correlations and a shuffle null.

Candidates (each a per-residue scalar, all standard published tables):
  KD        Kyte-Doolittle hydropathy                       -- the incumbent
  VOL       vdW residue volume                              -- REFUTED, kept as control
  HB        n hydrogen-bond donors+acceptors in side chain  -- H-bonding capacity
  CHG       formal charge at pH 7                           -- electrostatics
  ABSCHG    |charge|                                        -- charged-vs-not
  POLARSA   fraction of side-chain SASA that is polar (Janin/Chothia-style)
  dGtr      Fauchere-Pliska octanol transfer free energy (kcal/mol) -- a THERMODYNAMIC
            hydropathy, in the same units as the label, unlike KD which is an index
  FLEX      side-chain rotatable-bond count -- conformational entropy on burial
  HELIX     Pace-Scholtz helix propensity (kcal/mol)
  BETA      Koehl-Levitt beta propensity

W AND Y ARE THE KEY DIAGNOSTIC.  FINDINGS calls them 'exceptions' with slopes 0.34/0.35
at NEGATIVE KD.  On the CENTRED numbers they are 0.184/0.189 -- squarely in the hydrophobic
pack (mean 0.178), NOT exceptions at all.  So the axis must place W and Y with the
hydrophobics.  KD does not (KD_W=-0.9, KD_Y=-1.3).  dGtr and POLARSA DO.  If a table that
puts W/Y on the hydrophobic side beats KD, the axis is BURIAL-DRIVEN DESOLVATION and the
W/Y 'exception' dissolves -- which changes what feature to build.

Also: a per-residue INDICATOR regression.  Fit slope_d ~ 1 + f(d) for each candidate f and
report R^2, so the tables are compared on a common footing rather than by |r| alone.
"""
import csv, io, os, re, glob, json, math, collections, random

KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = {'G':60.1,'A':88.6,'S':89.0,'C':108.5,'D':111.1,'P':112.7,'N':114.1,'T':116.1,
       'E':138.4,'V':140.0,'Q':143.8,'H':153.2,'M':162.9,'I':166.7,'L':166.7,'K':168.6,
       'R':173.4,'F':189.9,'Y':193.6,'W':227.8}
# Fauchere & Pliska 1983 octanol-water transfer free energy of the side chain, kcal/mol.
# POSITIVE = more hydrophobic.  Note W = +2.25, the MOST hydrophobic of all -- exactly
# where KD fails.  Y = +0.96, also hydrophobic.
DGTR = {'W':2.25,'I':1.80,'F':1.79,'L':1.70,'C':1.54,'M':1.23,'V':1.22,'Y':0.96,
        'P':0.72,'A':0.31,'T':0.26,'H':0.13,'G':0.00,'S':-0.04,'Q':-0.22,'N':-0.60,
        'E':-0.64,'D':-0.77,'K':-0.99,'R':-1.01}
# side-chain H-bond donors + acceptors
HB = {'A':0,'C':1,'D':4,'E':4,'F':0,'G':0,'H':3,'I':0,'K':3,'L':0,'M':0,'N':4,
      'P':0,'Q':4,'R':5,'S':2,'T':2,'V':0,'W':1,'Y':2}
CHG = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1,'A':0,'C':0,'F':0,'G':0,'I':0,'L':0,'M':0,
       'N':0,'P':0,'Q':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
ABSCHG = dict((k, abs(v)) for k, v in CHG.items())
# fraction of side-chain surface that is polar (N/O), computed from standard side-chain
# atom composition -- W has one N in a large aromatic, so it is LOW like the phobics.
POLARSA = {'A':0.00,'G':0.00,'V':0.00,'L':0.00,'I':0.00,'F':0.00,'M':0.00,'P':0.00,
           'C':0.00,'W':0.11,'Y':0.14,'T':0.25,'S':0.33,'H':0.33,'N':0.50,'Q':0.40,
           'D':0.50,'E':0.40,'K':0.20,'R':0.43}
# rotatable bonds in the side chain (chi angles)
FLEX = {'G':0,'A':0,'P':0,'S':1,'C':1,'T':1,'V':1,'I':2,'L':2,'D':2,'N':2,'F':2,
        'Y':2,'W':2,'H':2,'E':3,'Q':3,'M':3,'K':4,'R':4}
# Pace & Scholtz 1998 helix propensity, kcal/mol (0 = best helix former, A)
HELIX = {'A':0.0,'L':0.21,'R':0.21,'M':0.24,'K':0.26,'Q':0.39,'E':0.40,'I':0.41,
         'W':0.49,'S':0.50,'Y':0.53,'F':0.54,'H':0.61,'V':0.61,'N':0.65,'T':0.66,
         'C':0.68,'D':0.69,'G':1.0,'P':3.16}
# Koehl & Levitt beta-sheet propensity
BETA = {'V':1.7,'I':1.6,'Y':1.47,'C':1.23,'W':1.37,'F':1.38,'L':1.3,'T':1.19,'M':1.05,
        'Q':1.1,'R':0.93,'N':0.89,'H':0.87,'A':0.83,'S':0.75,'G':0.75,'K':0.74,
        'P':0.55,'D':0.54,'E':0.37}

TABLES = [('KD', KD), ('dGtr_FauchereP', DGTR), ('VOL', VOL), ('POLARSA', POLARSA),
          ('HB_donacc', HB), ('ABS_CHARGE', ABSCHG), ('CHARGE', CHG),
          ('FLEX_chi', FLEX), ('HELIX_prop', HELIX), ('BETA_prop', BETA)]

MAXSASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
           'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
           'Y':263,'V':174}
THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
             'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
             'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
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
        for k in range(i, j+1): r[order[k]] = (i+j)/2.0
        i = j+1
    return r

def spear(x, y): return pear(rank(x), rank(y))

def partial(x, y, z):
    rxy, rxz, ryz = pear(x, y), pear(x, z), pear(y, z)
    return (rxy - rxz*ryz)/math.sqrt(max(1e-12, (1-rxz**2)*(1-ryz**2)))

def slope(xs, ys):
    n = len(xs)
    if n < 8: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((a-mx)**2 for a in xs)
    if sxx <= 0: return None
    return sum((a-mx)*(b-my) for a, b in zip(xs, ys))/sxx

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
        if one and MAXSASA.get(one):
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
            raw.append((src, pos, dst, dd, float(r['pred_ddG']), ent[1]))
        if len(raw) < 20: continue
        mt = sum(a[3] for a in raw)/len(raw); mp = sum(a[4] for a in raw)/len(raw)
        for src, pos, dst, t, p, rs in raw:
            recs.append(dict(src=src, dst=dst, t=t-mt, p=p-mp, rel=rs))
    return recs


def main():
    files = sorted(glob.glob('eval_results/abl_*.csv'))[:10]
    per_ck = {}
    for f in files:
        recs = load(f)
        if len(recs) >= 500:
            per_ck[os.path.basename(f)] = recs
    tags = sorted(per_ck)
    dest = {}
    for t in tags:
        by = collections.defaultdict(list)
        for r in per_ck[t]: by[r['dst']].append(r)
        dd = {}
        for d, v in by.items():
            s = slope([q['t'] for q in v], [q['p'] for q in v])
            if s is None: continue
            dd[d] = dict(slope=s, spearman=spear([q['t'] for q in v], [q['p'] for q in v]),
                         n=len(v))
        dest[t] = dd
    ds = sorted(dest[tags[0]])
    print('checkpoints %d   destinations %d   (within-protein centred)' % (len(tags), len(ds)))
    print()

    print('=' * 84)
    print('WHICH RESIDUE-PROPERTY TABLE EXPLAINS THE PER-DESTINATION SLOPE?')
    print('=' * 84)
    print('%-16s %9s %9s %9s %9s %9s' %
          ('table', 'r(slope)', 'R^2', 'rho(slope)', 'r(spear)', 'sign<0'))
    print('-' * 84)
    ranked = []
    for name, tab in TABLES:
        xs = [float(tab[d]) for d in ds]
        rs = [pear(xs, [dest[t][d]['slope'] for d in ds]) for t in tags]
        rr = [pear(xs, [dest[t][d]['spearman'] for d in ds]) for t in tags]
        rh = [spear(xs, [dest[t][d]['slope'] for d in ds]) for t in tags]
        mr = sum(rs)/len(rs)
        ranked.append((abs(mr), name, mr, sum(rr)/len(rr)))
        print('%-16s %+9.4f %9.4f %+9.4f %+9.4f %6d/%d'
              % (name, mr, mr*mr, sum(rh)/len(rh), sum(rr)/len(rr),
                 sum(1 for v in rs if v < 0), len(rs)))
    ranked.sort(reverse=True)
    best = ranked[0][1]
    print()
    print('BEST single table: %s  (|r| = %.4f, R^2 = %.4f)' % (best, ranked[0][0], ranked[0][0]**2))
    print()

    print('=' * 84)
    print('HEAD TO HEAD: every table vs KD, partial correlations (target = slope)')
    print('=' * 84)
    print('%-16s %14s %14s %10s' % ('table X', 'part(X|KD)', 'part(KD|X)', 'r(X,KD)'))
    print('-' * 84)
    for name, tab in TABLES:
        if name == 'KD': continue
        xs = [float(tab[d]) for d in ds]
        kd = [KD[d] for d in ds]
        px = [partial(xs, [dest[t][d]['slope'] for d in ds], kd) for t in tags]
        pk = [partial(kd, [dest[t][d]['slope'] for d in ds], xs) for t in tags]
        print('%-16s %+14.4f %+14.4f %+10.4f'
              % (name, sum(px)/len(px), sum(pk)/len(pk), pear(xs, kd)))
    print()

    print('=' * 84)
    print('THE W/Y DIAGNOSTIC -- are they exceptions, or does the right axis include them?')
    print('=' * 84)
    sl = dict((d, sum(dest[t][d]['slope'] for t in tags)/len(tags)) for d in ds)
    phob = [d for d in ds if d in 'AVILMFC']
    pol = [d for d in ds if d in 'DEKRNQSTHG']
    print('mean centred slope: aliphatic/aromatic AVILMFC = %.4f   polar DEKRNQSTHG = %.4f'
          % (sum(sl[d] for d in phob)/len(phob), sum(sl[d] for d in pol)/len(pol)))
    for d in ('W', 'Y'):
        print('  %s: slope %.4f  KD %+5.1f (says POLAR)   dGtr %+5.2f (says HYDROPHOBIC)'
              % (d, sl[d], KD[d], DGTR[d]))
    print('  -> W/Y sit with the HYDROPHOBIC group (%.4f vs %.4f), so a table that scores'
          % (sum(sl[d] for d in ('W','Y'))/2.0, sum(sl[d] for d in pol)/len(pol)))
    print('     them as hydrophobic is the correct axis; KD scores them as polar.')
    print()
    # residual of KD fit: which residues does KD get wrong?
    kd = [KD[d] for d in ds]
    y = [sl[d] for d in ds]
    a = slope(kd, y); b = sum(y)/len(y) - a*sum(kd)/len(kd)
    print('  residuals of slope ~ KD  (positive = KD under-predicts compression):')
    res = sorted(((y[i] - (a*kd[i]+b), ds[i]) for i in range(len(ds))), reverse=True)
    for r_, d in res:
        print('    %-3s  resid %+.4f   dGtr %+5.2f' % (d, r_, DGTR[d]))
    print()
    a2 = slope([DGTR[d] for d in ds], y)
    b2 = sum(y)/len(y) - a2*sum(DGTR[d] for d in ds)/len(ds)
    sse_kd = sum((y[i]-(a*kd[i]+b))**2 for i in range(len(ds)))
    sse_dg = sum((y[i]-(a2*DGTR[ds[i]]+b2))**2 for i in range(len(ds)))
    print('  SSE(slope ~ KD)   = %.6f' % sse_kd)
    print('  SSE(slope ~ dGtr) = %.6f   ratio %.3f' % (sse_dg, sse_dg/sse_kd))
    print()

    print('=' * 84)
    print('DEGENERATE BASELINE -- 5000 destination-label shuffles, best table')
    print('=' * 84)
    random.seed(0)
    for name in ('KD', 'dGtr_FauchereP', 'VOL', 'POLARSA'):
        tab = dict(TABLES)[name]
        xs = [float(tab[d]) for d in ds]
        obs = abs(pear(xs, y))
        null = []
        for _ in range(5000):
            perm = y[:]; random.shuffle(perm)
            null.append(abs(pear(xs, perm)))
        print('  %-16s |r|=%.4f   P(null >= obs) = %.4f'
              % (name, obs, sum(1 for v in null if v >= obs)/5000.0))

    json.dump(dict(slopes=sl, per_ck=dict((t, dest[t]) for t in tags)),
              open('results/sidechain_mech3.json', 'w'), indent=1)
    print('\nwrote results/sidechain_mech3.json')

main()
