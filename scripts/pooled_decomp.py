"""Why did pooled ddG PCC move?

Pooled PCC mixes (a) within-protein ranking and (b) between-protein scale/offset agreement.
A lever that only SHRINKS predictions changes pooled PCC without adding information.

Test: recompute pooled PCC after per-protein z-scoring of the PREDICTIONS ONLY
(removes each protein's predicted scale+offset, keeps the ranking). If a lever's pooled
gain survives, it is informational; if it evaporates, it was rescaling.
"""
import sys, csv, math
from collections import defaultdict

def pear(x,y):
    n=len(x); mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else float('nan')

for path in sys.argv[1:]:
    rows=defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['protein']].append((float(r['ddG']),float(r['pred_ddG'])))
    t_all,p_all,pz_all=[],[],[]
    for p,v in rows.items():
        t=[a for a,_ in v]; q=[b for _,b in v]
        m=sum(q)/len(q); sd=math.sqrt(sum((a-m)**2 for a in q)/len(q))
        z=[(a-m)/sd for a in q] if sd>0 else [0.0]*len(q)
        t_all+=t; p_all+=q; pz_all+=z
    print('%-30s pooled=%.4f   pooled_after_perprot_zscore(pred)=%.4f' %
          (path.split('/')[-1][:30], pear(t_all,p_all), pear(t_all,pz_all)))
