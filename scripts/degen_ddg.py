"""Degenerate baseline on the ddG side: what does a NON-model predictor score?

If a lever's pooled ddG PCC is near what a trivial predictor achieves, the number is not
evidence of learned structure. The honest reference for 'ranking mutations' is the
hydrophobicity/charge of the substitution alone -- but the cheapest one is: predict the
protein's MEAN ddG for every mutation (pure between-protein signal, zero within-protein).
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
    t,p,pm=[],[],[]
    for _,v in rows.items():
        tt=[a for a,_ in v]; m=sum(tt)/len(tt)
        t+=tt; p+=[b for _,b in v]; pm+=[m]*len(v)
    print('%-30s model=%.4f  protein-mean-only(degenerate)=%.4f' % (path.split('/')[-1][:30], pear(t,p), pear(t,pm)))
