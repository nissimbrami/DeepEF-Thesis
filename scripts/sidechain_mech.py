# -*- coding: utf-8 -*-
"""
sidechain_mech.py -- TEST THE MECHANISM behind the hydrophobic ranking failure.

FINDINGS 3.4: per-destination-residue slope correlates with Kyte-Doolittle at -0.734 and
Spearman at -0.740.  The STATED mechanism is that only 4 backbone atoms (N,CA,C,CB) are
stored so side-chain PACKING is invisible -- W and Y being slope outliers (0.34/0.35 at
negative KD) is the tell that the pattern tracks BULK, not hydropathy.

That is an assumption.  This script tests it:

  Q1  Does per-destination slope correlate with van der Waals VOLUME at least as well as
      with hydropathy?  Partial correlations decide which survives controlling for the
      other.  n = 20 destination residues (not 28 proteins) -- the |r|<0.374 rule does not
      apply here; the n=20 two-sided 5% threshold is |r|=0.444.
  Q2  Does the compression depend on whether the mutated POSITION is BURIED?  Split
      mutations by relative SASA of the WT residue at that position (Shrake-Rupley on the
      AlphaFold model, Tien 2013 max-SASA normalisation -- the same machinery as
      scripts/catalogue_vs_bp.py) and refit the slope within (destination x burial) cells.

DEGENERATE BASELINES built in: shuffled destination labels; within-cell n and true-ddG
variance reported so a "slope" that is really range restriction is visible.
"""
import csv, io, os, re, glob, json, math, collections

# ---------------------------------------------------------------- residue property tables
KD = dict(zip('ACDEFGHIKLMNPQRSTVWY',
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))

# van der Waals residue volume, A^3 (Richards 1974 / Creighton) -- standard packing table.
VOL = {'G':60.1,'A':88.6,'S':89.0,'C':108.5,'D':111.1,'P':112.7,'N':114.1,'T':116.1,
       'E':138.4,'V':140.0,'Q':143.8,'H':153.2,'M':162.9,'I':166.7,'L':166.7,'K':168.6,
       'R':173.4,'F':189.9,'Y':193.6,'W':227.8}

# SIDE-CHAIN-only volume = residue volume minus glycine volume.
SCVOL = dict((k, v - VOL['G']) for k, v in VOL.items())

# heavy atoms in the side chain BEYOND CB -- exactly what the dataset does NOT store.
NBEYOND_CB = {'G':0,'A':0,'S':1,'C':1,'T':2,'V':2,'P':2,'D':3,'N':3,'I':3,'L':3,
              'M':3,'E':4,'Q':4,'K':4,'H':5,'F':6,'R':5,'Y':7,'W':9}

MAXSASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,
           'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
           'Y':263,'V':174}
THREE2ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
             'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
             'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

MUT = re.compile(r'_([A-Z])(\d+)([A-Z])(?:_|$)')
DIRS = ('data_fixed/mutation_datasets', 'data/Processed_K50_dG_datasets/mutation_datasets')
PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'

# ---------------------------------------------------------------- stats helpers
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

def spear(x, y):
    return pear(rank(x), rank(y))

def partial(x, y, z):
    rxy, rxz, ryz = pear(x, y), pear(x, z), pear(y, z)
    den = math.sqrt(max(1e-12, (1-rxz**2)*(1-ryz**2)))
    return (rxy - rxz*ryz)/den

def _betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta)/a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 400):
        m = i//2
        if i == 0: num = 1.0
        elif i % 2 == 0: num = (m*(b-m)*x)/((a+2*m-1)*(a+2*m))
        else: num = -((a+m)*(a+b+m)*x)/((a+2*m)*(a+2*m+1))
        d = 1.0 + num*d
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0/d
        c = 1.0 + num/c
        if abs(c) < 1e-30: c = 1e-30
        f *= c*d
        if abs(1.0-c*d) < 1e-12: break
    return front*(f-1.0)

def t_p(r, n, k=0):
    df = n - 2 - k
    if df <= 0 or not (abs(r) < 1): return float('nan')
    t = abs(r)*math.sqrt(df/(1-r*r))
    return _betainc(df/2.0, 0.5, df/(df + t*t))

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

# ---------------------------------------------------------------- per-protein SASA
_SASA_CACHE = {}
def rel_sasa(protein):
    if protein in _SASA_CACHE: return _SASA_CACHE[protein]
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    p = os.path.join(PDB_DIR, protein + '.pdb')
    if not os.path.exists(p):
        _SASA_CACHE[protein] = {}
        return {}
    st = PDBParser(QUIET=True).get_structure(protein, p)
    model = st[0]
    ShrakeRupley().compute(model, level='R')
    out = {}
    for r in model.get_residues():
        if r.get_id()[0] != ' ': continue
        one = THREE2ONE.get(r.get_resname().strip())
        if not one or not MAXSASA.get(one): continue
        out[r.get_id()[1]] = (one, r.sasa/MAXSASA[one])
    _SASA_CACHE[protein] = out
    return out

# ---------------------------------------------------------------- deltaG -> (src,pos,dst)
def mut_map(protein):
    """Same join discipline as scripts/k13_muttype.py: key on deltaG to 6dp, DROP any
    value that maps to more than one distinct mutation rather than guessing."""
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

# ---------------------------------------------------------------- main
def analyse(path, sasa_ok):
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        per[r['protein']].append(r)
    by_dst = collections.defaultdict(list)
    by_dst_burial = collections.defaultdict(list)
    by_burial = collections.defaultdict(list)
    matched = 0
    pos_missing = 0
    for prot, rows in per.items():
        mm = mut_map(prot)
        if not mm: continue
        sm = rel_sasa(prot) if sasa_ok else {}
        for r in rows:
            k = round(float(r['deltaG']), 6)
            hit = mm.get(k)
            if not hit: continue
            src, pos, dst = hit
            if dst not in KD: continue
            dd = float(r['ddG'])
            if prot == '2K5H': dd -= 3.0824
            pp = float(r['pred_ddG'])
            by_dst[dst].append((dd, pp))
            matched += 1
            if sm:
                ent = sm.get(pos)
                if ent is None or ent[0] != src:
                    pos_missing += 1
                    continue
                rs = ent[1]
                bucket = 'buried' if rs < 0.25 else ('exposed' if rs > 0.50 else 'mid')
                by_dst_burial[(dst, bucket)].append((dd, pp))
                by_burial[bucket].append((dd, pp))
    return by_dst, by_dst_burial, by_burial, matched, pos_missing


def fit_table(by_dst):
    res = {}
    for d, v in by_dst.items():
        t = [a for a, _ in v]; p = [b for _, b in v]
        s = slope(t, p)
        if s is None: continue
        res[d] = dict(n=len(v), slope=s, spearman=spear(t, p),
                      var_true=var(t), var_pred=var(p))
    return res


def corr_block(res, label):
    ds = sorted(res)
    if len(ds) < 10: return None
    sl = [res[d]['slope'] for d in ds]
    rho = [res[d]['spearman'] for d in ds]
    kd = [KD[d] for d in ds]
    vo = [VOL[d] for d in ds]
    sv = [SCVOL[d] for d in ds]
    nb = [float(NBEYOND_CB[d]) for d in ds]
    n = len(ds)
    out = dict(label=label, n_res=n, residues=ds)
    for tgt_name, tgt in (('slope', sl), ('spearman', rho)):
        out[tgt_name] = dict(
            r_kd=pear(kd, tgt), rho_kd=spear(kd, tgt),
            r_vol=pear(vo, tgt), rho_vol=spear(vo, tgt),
            r_scvol=pear(sv, tgt), rho_scvol=spear(sv, tgt),
            r_nbcb=pear(nb, tgt), rho_nbcb=spear(nb, tgt),
            partial_vol_given_kd=partial(vo, tgt, kd),
            partial_kd_given_vol=partial(kd, tgt, vo),
            partial_scvol_given_kd=partial(sv, tgt, kd),
            partial_kd_given_scvol=partial(kd, tgt, sv),
            p_kd=t_p(pear(kd, tgt), n), p_vol=t_p(pear(vo, tgt), n),
            p_partial_vol=t_p(partial(vo, tgt, kd), n, 1),
            p_partial_kd=t_p(partial(kd, tgt, vo), n, 1))
    out['r_kd_vol'] = pear(kd, vo)
    return out


def main():
    files = sorted(glob.glob('eval_results/abl_*.csv'))[:10]
    sasa_ok = True
    try:
        import Bio  # noqa
    except Exception:
        sasa_ok = False
        print('!! Biopython unavailable -- burial analysis SKIPPED')

    allres, allburial, allbybur = {}, {}, {}
    for f in files:
        bd, bdb, bb, matched, missing = analyse(f, sasa_ok)
        if matched < 500:
            print('skip %s (matched=%d)' % (os.path.basename(f), matched)); continue
        res = fit_table(bd)
        if len(res) < 10: continue
        tag = os.path.basename(f)
        allres[tag] = dict(matched=matched, pos_dropped=missing, per_residue=res,
                           corr=corr_block(res, tag))
        bres = {}
        for key, v in bdb.items():
            d, b = key
            t = [a for a, _ in v]; p = [q for _, q in v]
            s = slope(t, p)
            if s is None: continue
            bres['%s|%s' % (d, b)] = dict(n=len(v), slope=s, spearman=spear(t, p),
                                          var_true=var(t))
        allburial[tag] = bres
        gb = {}
        for b, v in bb.items():
            t = [a for a, _ in v]; p = [q for _, q in v]
            s = slope(t, p)
            gb[b] = dict(n=len(v), slope=s, spearman=spear(t, p), var_true=var(t))
        allbybur[tag] = gb

    if not allres:
        print('JOIN FAILED -- no checkpoint produced enough matched rows'); return

    tags = sorted(allres)
    print('=' * 78)
    print('Q1  WHAT EXPLAINS THE PER-DESTINATION SLOPE: HYDROPATHY OR SIDE-CHAIN BULK?')
    print('=' * 78)
    print('checkpoints: %d   mean matched rows: %d' %
          (len(tags), sum(allres[t]['matched'] for t in tags)//len(tags)))
    n_res = allres[tags[0]]['corr']['n_res']
    print('n destination residues = %d' % n_res)
    print('collinearity: corr(KD, volume) = %+.4f' % allres[tags[0]]['corr']['r_kd_vol'])
    print()
    for tgt in ('slope', 'spearman'):
        print('--- target: %s ---' % tgt)
        hdr = '%-34s %9s %9s %9s %9s' % ('quantity', 'mean r', 'min', 'max', 'p(ck0)')
        print(hdr); print('-'*len(hdr))
        keys = [('r_kd', 'corr(%s, KD hydropathy)' % tgt, 'p_kd'),
                ('r_vol', 'corr(%s, residue volume)' % tgt, 'p_vol'),
                ('r_scvol', 'corr(%s, SIDE-CHAIN volume)' % tgt, None),
                ('r_nbcb', 'corr(%s, atoms beyond CB)' % tgt, None),
                ('partial_vol_given_kd', 'PARTIAL vol | KD', 'p_partial_vol'),
                ('partial_kd_given_vol', 'PARTIAL KD  | vol', 'p_partial_kd'),
                ('partial_scvol_given_kd', 'PARTIAL scvol | KD', None),
                ('partial_kd_given_scvol', 'PARTIAL KD    | scvol', None)]
        for k, lab, pk in keys:
            vals = [allres[t]['corr'][tgt][k] for t in tags]
            ps = ('%9.2e' % allres[tags[0]]['corr'][tgt][pk]) if pk else ' ' * 9
            print('%-34s %+9.4f %+9.4f %+9.4f %s'
                  % (lab, sum(vals)/len(vals), min(vals), max(vals), ps))
        for k, lab in (('r_kd','KD'), ('r_vol','vol'), ('partial_vol_given_kd','vol|KD'),
                       ('partial_kd_given_vol','KD|vol')):
            vals = [allres[t]['corr'][tgt][k] for t in tags]
            neg = sum(1 for v in vals if v < 0)
            print('   sign-consistent(%s<0): %d/%d' % (lab, neg, len(vals)))
        print()

    ref = allres[tags[0]]['per_residue']
    print('--- per-destination table (%s) sorted by VOLUME ---' % tags[0])
    print('%-4s %7s %8s %8s %8s %9s' % ('res','n','KD','vol A^3','slope','spearman'))
    for d in sorted(ref, key=lambda x: VOL[x]):
        print('%-4s %7d %8.1f %8.1f %8.4f %9.4f'
              % (d, ref[d]['n'], KD[d], VOL[d], ref[d]['slope'], ref[d]['spearman']))
    print()

    import random
    random.seed(0)
    ds = sorted(ref)
    sl = [ref[d]['slope'] for d in ds]
    null_kd, null_vol = [], []
    for _ in range(2000):
        perm = sl[:]; random.shuffle(perm)
        null_kd.append(abs(pear([KD[d] for d in ds], perm)))
        null_vol.append(abs(pear([VOL[d] for d in ds], perm)))
    obs_kd = abs(allres[tags[0]]['corr']['slope']['r_kd'])
    obs_vol = abs(allres[tags[0]]['corr']['slope']['r_vol'])
    print('DEGENERATE BASELINE (2000 destination-label shuffles, %s):' % tags[0])
    print('  P(|r_KD_null|  >= %.4f) = %.4f' % (obs_kd, sum(1 for v in null_kd if v >= obs_kd)/2000.0))
    print('  P(|r_vol_null| >= %.4f) = %.4f' % (obs_vol, sum(1 for v in null_vol if v >= obs_vol)/2000.0))
    print()

    if allbybur:
        print('=' * 78)
        print('Q2  DOES THE COMPRESSION DEPEND ON BURIAL OF THE MUTATED POSITION?')
        print('=' * 78)
        print('burial = Shrake-Rupley rel.SASA of the WT residue (Tien 2013 max SASA)')
        print('  buried rel<0.25 | mid 0.25-0.50 | exposed rel>0.50')
        print('rows dropped for PDB-numbering mismatch: %d (mean per checkpoint)' %
              (sum(allres[t]['pos_dropped'] for t in tags)//len(tags)))
        print()
        print('--- global slope by burial bucket (mean over %d checkpoints) ---' % len(tags))
        print('%-9s %9s %9s %9s %9s' % ('bucket','n','slope','spearman','var_true'))
        for b in ('buried','mid','exposed'):
            vals = [allbybur[t][b] for t in tags if b in allbybur[t]]
            if not vals: continue
            print('%-9s %9d %9.4f %9.4f %9.4f'
                  % (b, sum(v['n'] for v in vals)//len(vals),
                     sum(v['slope'] for v in vals)/len(vals),
                     sum(v['spearman'] for v in vals)/len(vals),
                     sum(v['var_true'] for v in vals)/len(vals)))
        print()
        print('--- KEY TEST: corr(slope, volume) computed SEPARATELY within each bucket ---')
        print('if packing is the mechanism, |r| must be LARGER at buried positions')
        print('%-9s %6s %6s %11s %11s %11s %11s' %
              ('bucket','n_ck','n_res','r_slope_vol','r_slope_KD','part vol|KD','part KD|vol'))
        bsum = {}
        for b in ('buried','mid','exposed'):
            rv, rk, pv, pk, nres = [], [], [], [], []
            for t in tags:
                sub = dict((k.split('|')[0], v) for k, v in allburial[t].items()
                           if k.endswith('|'+b) and v['n'] >= 30)
                if len(sub) < 10: continue
                dd = sorted(sub)
                s = [sub[d]['slope'] for d in dd]
                rv.append(pear([VOL[d] for d in dd], s))
                rk.append(pear([KD[d] for d in dd], s))
                pv.append(partial([VOL[d] for d in dd], s, [KD[d] for d in dd]))
                pk.append(partial([KD[d] for d in dd], s, [VOL[d] for d in dd]))
                nres.append(len(dd))
            if not rv:
                print('%-9s %6s  too few destination residues with n>=30' % (b, '-')); continue
            bsum[b] = dict(n_res=sum(nres)/float(len(nres)), r_vol=sum(rv)/len(rv),
                           r_kd=sum(rk)/len(rk), p_vol=sum(pv)/len(pv),
                           p_kd=sum(pk)/len(pk), n_ck=len(rv))
            print('%-9s %6d %6.1f %+11.4f %+11.4f %+11.4f %+11.4f'
                  % (b, bsum[b]['n_ck'], bsum[b]['n_res'], bsum[b]['r_vol'],
                     bsum[b]['r_kd'], bsum[b]['p_vol'], bsum[b]['p_kd']))
        print()
        print('--- slope of the 5 LARGEST hydrophobics (F,I,L,M,W) by burial ---')
        print('%-9s %8s %8s %8s %6s' % ('bucket','n','slope','spearman','cells'))
        for label, group in (('BIG-PHOBIC', ('F','I','L','M','W')),
                             ('POLAR', ('D','E','K','N','R')),
                             ('SMALL', ('A','G','S'))):
            print('  [%s]' % label)
            for b in ('buried','mid','exposed'):
                tot_n, sl_, sp_ = 0, [], []
                for t in tags:
                    for d in group:
                        v = allburial[t].get('%s|%s' % (d, b))
                        if v and v['n'] >= 30:
                            tot_n += v['n']; sl_.append(v['slope']); sp_.append(v['spearman'])
                if sl_:
                    print('%-9s %8d %8.4f %8.4f %6d'
                          % (b, tot_n//max(1, len(tags)), sum(sl_)/len(sl_),
                             sum(sp_)/len(sp_), len(sl_)))

    json.dump(dict(per_checkpoint=allres, burial=allburial, by_burial=allbybur,
                   tables=dict(KD=KD, VOL=VOL, SCVOL=SCVOL, NBEYOND_CB=NBEYOND_CB)),
              open('results/sidechain_mech.json', 'w'), indent=1)
    print('\nwrote results/sidechain_mech.json')

main()
