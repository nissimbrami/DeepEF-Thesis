"""Was the missing 9 a BIASED subset? Test on b_p, |b_p|, length, burial, designed-vs-natural."""
import csv, json, math, os, sys
sys.path.insert(0,'/home/nissimb/DeepPEF')
import torch, train_utils as tu
EV='eval_results/abl_calib_ctrl_repro2_e14.csv'
OLD='/groups/keasar_group/casp15/meytav/protein_tensors'
NEW='/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors'
rows={}
for r in csv.DictReader(open(EV)):
    rows.setdefault(r['protein'],[]).append((float(r['deltaG']),float(r['pred_deltaG']),float(r['ddG'])))
bp={}; ap={}; pcc={}
def pear(x,y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=sum(x)/n,sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    if sxx<=0 or syy<=0: return float('nan')
    return sxy/math.sqrt(sxx*syy)
for p,v in rows.items():
    wt=[t for t in v if t[2]==0.0]
    if wt: bp[p]=wt[0][1]-wt[0][0]
    # ddG slope a_p and per-protein ddG PCC
    tv=[t[2] for t in v]
    # predicted ddG = pred - pred_WT
    if wt:
        pw=wt[0][1]
        pv=[t[1]-pw for t in v]
        pcc[p]=pear(tv,pv)
        n=len(tv); mx=sum(tv)/n; my=sum(pv)/n
        sxx=sum((a-mx)**2 for a in tv)
        ap[p]=(sum((a-mx)*(b-my) for a,b in zip(tv,pv))/sxx) if sxx>0 else float('nan')
feat={}
for p in bp:
    root=NEW if os.path.exists(os.path.join(NEW,p,'coords_tensor.pt')) else OLD
    x=torch.as_tensor(torch.load(os.path.join(root,p,'coords_tensor.pt'),map_location='cpu',weights_only=False)).float()
    m=torch.as_tensor(torch.load(os.path.join(root,p,'mask_tensor.pt'),map_location='cpu',weights_only=False)).float()
    b=tu.compute_burial(x,m); v=b[m>0].flatten()
    feat[p]=dict(mean_burial=float(v.mean()), length=float(int(m.sum())))
MISS=set(['1W4H','2K1B','2K28','2KXD','2L33','6EWT','r11_1081_TrROS_Hall','r12_757_TrROS_Hall','2K5H'])
present=[p for p in bp if p not in MISS]; miss=[p for p in bp if p in MISS]
print('present n=%d  missing n=%d'%(len(present),len(miss)))
def mw(a,b):
    # Mann-Whitney U with normal approx (ties-corrected)
    comb=sorted([(v,0) for v in a]+[(v,1) for v in b])
    ranks={}; i=0; rk=[0.0]*len(comb)
    while i<len(comb):
        j=i
        while j+1<len(comb) and comb[j+1][0]==comb[i][0]: j+=1
        avg=(i+j)/2.0+1
        for k in range(i,j+1): rk[k]=avg
        i=j+1
    ra=sum(rk[i] for i in range(len(comb)) if comb[i][1]==0)
    na,nb=len(a),len(b)
    U=ra-na*(na+1)/2.0
    mu=na*nb/2.0; sd=math.sqrt(na*nb*(na+nb+1)/12.0)
    if sd==0: return float('nan')
    z=(U-mu)/sd
    return math.erfc(abs(z)/math.sqrt(2))
def rep(name, d):
    a=[d[p] for p in present if p in d and d[p]==d[p]]
    b=[d[p] for p in miss if p in d and d[p]==d[p]]
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    print('%-14s present %8.4f (n=%d)   missing %8.4f (n=%d)   MW p=%.3f'%(name,ma,len(a),mb,len(b),mw(a,b)))
    return dict(present_mean=ma,missing_mean=mb,p=mw(a,b))
out={}
out['b_p']=rep('b_p',bp)
out['abs_b_p']=rep('|b_p|',{p:abs(v) for p,v in bp.items()})
out['a_p']=rep('a_p',ap)
out['pcc']=rep('per-prot PCC',pcc)
out['length']=rep('length',{p:feat[p]['length'] for p in feat})
out['mean_burial']=rep('mean_burial',{p:feat[p]['mean_burial'] for p in feat})
des=lambda p: (('_rd' in p) or ('TrROS' in p) or p.startswith(('HEEH','HHH','EEHEE','EHEE')))
dp=sum(des(p) for p in present); dm=sum(des(p) for p in miss)
print('\ndesigned: present %d/%d (%.0f%%)  missing %d/%d (%.0f%%)'%(dp,len(present),100*dp/len(present),dm,len(miss),100*dm/len(miss)))
out['designed']={'present':[dp,len(present)],'missing':[dm,len(miss)]}
# rank of missing by |b_p|
order=sorted(bp,key=lambda p:-abs(bp[p]))
print('\ntop-6 |b_p| (the oracle-gain carriers):')
for p in order[:6]: print('   %-22s |b_p|=%6.3f  %s'%(p,abs(bp[p]),'WAS-MISSING' if p in MISS else 'present'))
out['top6']=[[p,abs(bp[p]),p in MISS] for p in order[:6]]
json.dump(out,open('results/missing_bias.json','w'),indent=1)
print('\nwrote results/missing_bias.json')
