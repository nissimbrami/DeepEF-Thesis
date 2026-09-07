"""Partial correlations: does pLDDT survive controlling for length, or vice versa?"""
import json, csv, math
from collections import defaultdict
mn=defaultdict(float); mc=defaultdict(int)
with open('eval_results/abl_calib_ctrl_repro2_e14.csv') as f:
    for r in csv.DictReader(f):
        p=r['protein']; mn[p]+=abs(float(r['pred_deltaG'])-float(r['deltaG'])); mc[p]+=1
mae={p:mn[p]/mc[p] for p in mn}
calib=json.load(open('results/calib_per_protein.json'))
pl=defaultdict(list)
with open('data/Processed_K50_dG_datasets/plddt.csv') as f:
    for r in csv.DictReader(f): pl[r['protein']].append(float(r['plddt']))
pm={p:sum(v)/len(v) for p,v in pl.items()}; L={p:len(v) for p,v in pl.items()}
have=sorted([p for p in calib if p in pm])
def pear(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy) if sxx>0 and syy>0 else float('nan')
def partial(xy,xz,yz):  # corr(x,y | z)
    d=math.sqrt(max(1e-12,(1-xz*xz)*(1-yz*yz)))
    return (xy-xz*yz)/d
def pval(r,n):
    df=n-2; t=abs(r)*math.sqrt(df/max(1e-12,1-r*r))
    def bcf(a,b,x):
        c=1.0; d=1-(a+b)*x/(a+1); d=1/d if abs(d)>1e-300 else 1e300; h=d
        for m in range(1,300):
            m2=2*m; aa=m*(b-m)*x/((a-1+m2)*(a+m2))
            d=1+aa*d; c=1+aa/c; d=1/d if abs(d)>1e-300 else 1e300; h*=d*c
            aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+1+m2))
            d=1+aa*d; c=1+aa/c; d=1/d if abs(d)>1e-300 else 1e300; de=d*c; h*=de
            if abs(de-1)<3e-16: break
        return h
    def bi(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
        return bt*bcf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*bcf(b,a,1-x)/b
    return bi(df/2,0.5,df/(df+t*t))
P=[pm[p] for p in have]; N=[L[p] for p in have]
r_PN=pear(P,N); n=len(have)
print('n=%d   corr(plddt, length) = %+.4f  (p=%.3g)'%(n,r_PN,pval(r_PN,n)))
print()
print('%-9s %10s %10s %14s %14s'%('target','r_plddt','r_length','plddt|length','length|plddt'))
for lab,y in [('pcc_ddG',[calib[p]['pcc_ddg'] for p in have]),
              ('a_p',    [calib[p]['a_p'] for p in have]),
              ('dG_MAE', [mae[p] for p in have]),
              ('|b_p|',  [abs(calib[p]['b_p_wt_error']) for p in have])]:
    rp=pear(P,y); rn=pear(N,y)
    pp=partial(rp,r_PN,rn); pn=partial(rn,r_PN,rp)
    # partial corr df = n-3
    def pv3(r):
        df=n-3; t=abs(r)*math.sqrt(df/max(1e-12,1-r*r)); return pval(r,df+2)
    print('%-9s %+10.4f %+10.4f %+9.4f(p=%.2g) %+9.4f(p=%.2g)'%(lab,rp,rn,pp,pv3(pp),pn,pv3(pn)))
print()
print('Bonferroni bar over 90 tests: 5.6e-4')
