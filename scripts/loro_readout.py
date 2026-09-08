"""LORO readout: does the descriptor arm beat one-hot on mutations TO TRYPTOPHAN?

W was held out of TRAINING only, in both arms. The arms differ in exactly one flag:
--aa_descriptors none vs mordred_pca16. Prediction on record (CONTEXT.md CHECKPOINT 26):
the descriptor arm will NOT beat one-hot by much, because ProtT5 already predicts
held-out-residue hydropathy at R^2 0.704 and the embedding is in BOTH arms.

Destination residue is recovered by joining on deltaG against the mutation files,
exactly as scripts/k13_muttype.py does: join on VALUE keyed to 6 decimals, and DROP any
deltaG that is ambiguous (appears more than once) rather than guessing.
"""
import csv, io, math, os, glob, collections, json, re, sys

MUT = re.compile(r'_([A-Z])(\d+)([A-Z])$')
DIRS = ('data_fixed/mutation_datasets', 'data/Processed_K50_dG_datasets/mutation_datasets')

def dest_map(protein):
    """deltaG (6dp) -> (source, destination) residue, unambiguous rows only."""
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
            seen[k].add((m.group(1), m.group(3)))
        return {k: list(v)[0] for k, v in seen.items() if len(v) == 1}
    return {}

def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0]*len(v); i = 0
    while i < len(order):                      # average ties, else rho is biased
        j = i
        while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
        avg = (i+j)/2.0
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else float('nan')

def spear(x, y):
    return pear(rank(x), rank(y))

def load(f):
    """-> per-protein rows, with 2K5H's known mutant-reference shift corrected."""
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        per[r['protein']].append(r)
    return per

def analyse(f):
    per = load(f)
    toW, fromW, other, allrows = [], [], [], []
    matched = unmatched = 0
    perprot_W = collections.defaultdict(list)
    for prot, rows in per.items():
        dm = dest_map(prot)
        for r in rows:
            dd = float(r['ddG']); pp = float(r['pred_ddG'])
            if prot == '2K5H': dd -= 3.0824
            allrows.append((dd, pp))
            k = round(float(r['deltaG']), 6)
            sd = dm.get(k) if dm else None
            if not sd:
                unmatched += 1; continue
            matched += 1
            src, dst = sd
            if dst == 'W':
                toW.append((dd, pp)); perprot_W[prot].append((dd, pp))
            elif src == 'W':
                fromW.append((dd, pp))
            else:
                other.append((dd, pp))
    def blk(v):
        if len(v) < 3: return dict(n=len(v))
        t = [a for a, _ in v]; p = [b for _, b in v]
        return dict(n=len(v), rho=spear(t, p), pcc=pear(t, p),
                    std_true=(sum((a-sum(t)/len(t))**2 for a in t)/len(t))**.5,
                    std_pred=(sum((b-sum(p)/len(p))**2 for b in p)/len(p))**.5)
    pp = {k: spear([a for a,_ in v],[b for _,b in v])
          for k, v in perprot_W.items() if len(v) >= 5}
    return dict(file=os.path.basename(f), matched=matched, unmatched=unmatched,
                toW=blk(toW), fromW=blk(fromW), other=blk(other), pooled=blk(allrows),
                perprot_W_rho=pp,
                perprot_W_mean=(sum(pp.values())/len(pp) if pp else float('nan')),
                n_prot_W=len(pp))

if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('eval_results/abl_loroW_*.csv'))
    out = {}
    for f in files:
        a = analyse(f); out[a['file']] = a
        print('\n=== %s ===' % a['file'])
        print('  join: matched %d, unmatched %d' % (a['matched'], a['unmatched']))
        for key in ('toW', 'fromW', 'other', 'pooled'):
            b = a[key]
            if b.get('n', 0) >= 3:
                print('  %-7s n=%-6d rho=%+.4f  pcc=%+.4f  std_true=%.3f std_pred=%.3f'
                      % (key, b['n'], b['rho'], b['pcc'], b['std_true'], b['std_pred']))
            else:
                print('  %-7s n=%d (too few)' % (key, b.get('n', 0)))
        print('  per-protein W rho: mean=%+.4f over %d proteins' % (a['perprot_W_mean'], a['n_prot_W']))
    if len(out) == 2:
        k = list(out)
        oh = [x for x in k if 'onehot' in x]; de = [x for x in k if 'desc' in x]
        if oh and de:
            o, d = out[oh[0]], out[de[0]]
            print('\n===== COMPARISON (desc - onehot) =====')
            print('  toW    rho: %+.4f -> %+.4f   delta %+.4f' % (o['toW']['rho'], d['toW']['rho'], d['toW']['rho']-o['toW']['rho']))
            print('  pooled rho: %+.4f -> %+.4f   delta %+.4f' % (o['pooled']['rho'], d['pooled']['rho'], d['pooled']['rho']-o['pooled']['rho']))
            print('  pooled pcc: %+.4f -> %+.4f   delta %+.4f' % (o['pooled']['pcc'], d['pooled']['pcc'], d['pooled']['pcc']-o['pooled']['pcc']))
            print('  perprot W : %+.4f -> %+.4f   delta %+.4f' % (o['perprot_W_mean'], d['perprot_W_mean'], d['perprot_W_mean']-o['perprot_W_mean']))
    json.dump(out, open('results/loro_readout.json','w'), indent=1)
    print('\nwrote results/loro_readout.json')
