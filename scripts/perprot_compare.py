"""Per-protein paired comparison: does a lever move per-protein PCC on the SAME proteins?

Paired test is the right one here -- the 28 proteins are shared between arms, so the
per-protein PCC differences are paired, not two independent samples.
"""
import sys, csv, math
from collections import defaultdict, OrderedDict

def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['protein']].append((float(r['ddG']), float(r['pred_ddG']),
                                       float(r['deltaG']), float(r['pred_deltaG'])))
    return rows

def pear(x, y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<=0 or syy<=0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def stats(path):
    d=OrderedDict()
    for p,v in sorted(load(path).items()):
        ddg=[t[0] for t in v]; pddg=[t[1] for t in v]
        r=pear(ddg,pddg)
        sd_t=math.sqrt(sum((a-sum(ddg)/len(ddg))**2 for a in ddg)/len(ddg))
        sd_p=math.sqrt(sum((a-sum(pddg)/len(pddg))**2 for a in pddg)/len(pddg))
        s=sd_p/sd_t if sd_t>0 else float('nan')
        wt=[i for i,t in enumerate(v) if t[0]==0.0]
        bp=(v[wt[0]][3]-v[wt[0]][2]) if wt else float('nan')
        d[p]=(r,s,r*s,bp)
    return d

ctrl=stats(sys.argv[1]); arm=stats(sys.argv[2])
common=[p for p in ctrl if p in arm]
print('paired over %d proteins' % len(common))
for name,idx in (('r',0),('s',1),('a_p',2),('b_p',3)):
    dif=[arm[p][idx]-ctrl[p][idx] for p in common if not math.isnan(arm[p][idx]) and not math.isnan(ctrl[p][idx])]
    n=len(dif); m=sum(dif)/n
    sd=math.sqrt(sum((x-m)**2 for x in dif)/(n-1)) if n>1 else float('nan')
    se=sd/math.sqrt(n) if n>1 else float('nan')
    t=m/se if se and se>0 else float('nan')
    win=sum(1 for x in dif if x>0)
    print('  d%-4s mean=%+.4f sd=%.4f  t=%+.2f (df=%d)  arm>ctrl in %d/%d' % (name,m,sd,t,n-1,win,n))
