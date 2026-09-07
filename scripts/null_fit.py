"""Can ANY pure-random-offset null match all three of our numbers at once?

Grid-search the null over (noise_sd, slope) and score each cell by how far it is from
our measured triple (pooled 0.59, offset-removed 0.71, per-protein 0.798). If no cell
matches, the lift is NOT merely an estimator artifact and the thesis survives the
critic's strongest attack.
"""
import csv, math, random
from collections import defaultdict
SRC='eval_results/abl_calib_ctrl_repro2_e14.csv'
prot=defaultdict(list)
with open(SRC) as f:
    for r in csv.DictReader(f): prot[r['protein']].append(float(r['ddG']))
def pear(x,y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<=0 or syy<=0: return float('nan')
    return sxy/math.sqrt(sxx*syy)
STD_B=1.5741
TARGET=(0.59,0.71,0.798)
def trial(sd,a,seed):
    random.seed(seed)
    X=[];Y=[];per=[];groups=[]
    for p,d in prot.items():
        b=random.gauss(0,STD_B); xs=[];ys=[]
        for t in d:
            xs.append(t); ys.append(a*t+b+random.gauss(0,sd))
        per.append(pear(xs,ys)); groups.append((xs,ys)); X+=xs; Y+=ys
    pooled=pear(X,Y)
    Xc=[];Yc=[]
    for xs,ys in groups:
        off=sum(q-p for p,q in zip(xs,ys))/len(xs)
        Xc+=xs; Yc+=[q-off for q in ys]
    pv=[q for q in per if q==q]
    return pooled,pear(Xc,Yc),sum(pv)/len(pv)
best=None
print('%6s %6s | %7s %7s %7s | %7s'%('noise','slope','pooled','removed','perprot','dist'))
print('-'*60)
for sd in [x/10 for x in range(1,26)]:
    for a in [x/100 for x in range(30,131,5)]:
        r=[trial(sd,a,s) for s in range(3)]
        m=[sum(c[i] for c in r)/3 for i in range(3)]
        dist=math.sqrt(sum((m[i]-TARGET[i])**2 for i in range(3)))
        if best is None or dist<best[0]: best=(dist,sd,a,m)
d,sd,a,m=best
print('%6.2f %6.2f | %7.4f %7.4f %7.4f | %7.4f  <- BEST NULL'%(sd,a,m[0],m[1],m[2],d))
print('%6s %6s | %7.4f %7.4f %7.4f |'%('--','--',TARGET[0],TARGET[1],TARGET[2]))
print()
print('Closest the pure-random-offset null can get to our triple: L2 distance %.4f'%d)
print('Per-axis error: pooled %+.4f  removed %+.4f  perprot %+.4f'%(m[0]-TARGET[0],m[1]-TARGET[1],m[2]-TARGET[2]))
