"""Robustness of the W8 pLDDT gradient: outlier leave-one-out + Spearman + confound."""
import json, csv, math
from collections import defaultdict
SRC='eval_results/abl_calib_ctrl_repro2_e14.csv'
mn=defaultdict(float); mc=defaultdict(int); nres=defaultdict(set)
with open(SRC) as f:
    for r in csv.DictReader(f):
        p=r['protein']; mn[p]+=abs(float(r['pred_deltaG'])-float(r['deltaG'])); mc[p]+=1
mae={p:mn[p]/mc[p] for p in mn}
calib=json.load(open('results/calib_per_protein.json'))
pl=defaultdict(list)
with open('data/Processed_K50_dG_datasets/plddt.csv') as f:
    for r in csv.DictReader(f): pl[r['protein']].append(float(r['plddt']))
pm={p:sum(v)/len(v) for p,v in pl.items()}
L={p:len(v) for p,v in pl.items()}
have=sorted([p for p in calib if p in pm])

def pear(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else float('nan')
def rank(v):
    s=sorted(range(len(v)), key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
        avg=(i+j)/2.0+1
        for k in range(i,j+1): r[s[k]]=avg
        i=j+1
    return r
def spear(x,y): return pear(rank(x),rank(y))

series={'pcc_ddG':lambda p:calib[p]['pcc_ddg'],'a_p':lambda p:calib[p]['a_p'],
        'dG_MAE':lambda p:mae[p],'|b_p|':lambda p:abs(calib[p]['b_p_wt_error'])}
x=[pm[p] for p in have]
print('=== Spearman (rank, outlier-insensitive), n=24 ===')
for k,f in series.items():
    print('  plddt vs %-8s  pearson %+.4f   spearman %+.4f'%(k,pear(x,[f(p) for p in have]),spear(x,[f(p) for p in have])))

print()
print('=== leave-one-out on pearson: most influential protein per series ===')
for k,f in series.items():
    base=pear(x,[f(p) for p in have]); worst=None
    for p in have:
        g=[q for q in have if q!=p]
        r=pear([pm[q] for q in g],[f(q) for q in g])
        if worst is None or abs(r-base)>abs(worst[1]-base): worst=(p,r)
    print('  %-8s base r=%+.4f   drop %-20s -> r=%+.4f   (LOO min/max %+.4f/%+.4f)'
          %(k,base,worst[0],worst[1],
            min(pear([pm[q] for q in have if q!=p],[f(q) for q in have if q!=p]) for p in have),
            max(pear([pm[q] for q in have if q!=p],[f(q) for q in have if q!=p]) for p in have)))

print()
print('=== confound: is pLDDT a proxy for length? ===')
print('  n_residues range over the 24: %d-%d'%(min(L[p] for p in have),max(L[p] for p in have)))
print('  plddt vs length      r=%+.4f'%pear(x,[L[p] for p in have]))
print('  length vs pcc_ddG    r=%+.4f'%pear([L[p] for p in have],[calib[p]['pcc_ddg'] for p in have]))
print('  length vs a_p        r=%+.4f'%pear([L[p] for p in have],[calib[p]['a_p'] for p in have]))
print()
print('=== designed vs natural (the 4 no-pLDDT are all designed) ===')
des=[p for p in calib if p.startswith(('HEEH','HHH','r11','r12','r18'))]
nat=[p for p in calib if p not in des]
for lab,g in (('designed',des),('natural',nat)):
    print('  %-9s n=%2d  pcc_mean %.4f  a_p_mean %.4f  MAE_mean %.4f'
          %(lab,len(g),sum(calib[p]['pcc_ddg'] for p in g)/len(g),
            sum(calib[p]['a_p'] for p in g)/len(g),sum(mae[p] for p in g)/len(g)))
