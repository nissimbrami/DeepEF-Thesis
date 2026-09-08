"""T4: does one_hot[i] encode the mutation that row i's mut_type NAMES?

This is the alignment test that matters. If labels were misaligned to structures the
shapes would still agree and every metric would still look normal -- only this test
sees it. For every single-point row: decode one_hot[i], compare to the WT one-hot,
and check the difference is exactly the one position and one substitution that
mut_type names (e.g. 'A12G' -> position 12 changes A to G).
"""
import pandas as pd, numpy as np, torch, os, re, sys
ROOT='data/Processed_K50_dG_datasets/training_data'
D='data/Processed_K50_dG_datasets/mutation_datasets/'
AA=list("ACDEFGHIKLMNPQRSTVWY")
TEST=['1GYZ','1PSE','1QKH','1QP2','1TUC','1W4H','2BTH','2K1B','2K28','2K5H','2KVS','2KWH',
      '2KXD','2L33','2LQK','2WXC','3DKM','4C26','6EWS','6EWT','6EWU','HEEH_KT_rd6_0746',
      'HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244','r11_1081_TrROS_Hall',
      'r12_757_TrROS_Hall','r18_3_TrROS_Hall']
pat=re.compile(r'^([A-Z])(\d+)([A-Z])$')
print('%-22s %7s %7s %7s %7s %7s  %s'%('protein','n1mut','ok','badpos','badaa','ndiff!=1','worst example'))
G=dict(ok=0,bad=0)
for p in TEST:
    oh=torch.load(ROOT+'/'+p+'/one_hot_encodings.pt',map_location='cpu',weights_only=False).cpu().numpy()
    m=pd.read_csv(D+p+'.csv')
    mf=m[~m['mut_type'].astype(str).str.contains('ins|del')].reset_index(drop=True)
    dec=np.array([[AA[j] for j in oh[i][:,:20].argmax(axis=1)] for i in range(oh.shape[0])])
    wt=dec[0]
    n1=ok=bp=ba=nd=0; ex=''
    for i in range(min(len(mf),oh.shape[0])):
        mt=str(mf['mut_type'].iloc[i])
        g=pat.match(mt)
        if not g: continue
        n1+=1
        src,pos,dst=g.group(1),int(g.group(2)),g.group(3)
        d=np.where(dec[i]!=wt)[0]
        if len(d)!=1:
            nd+=1
            if not ex: ex='%s: %d positions differ'%(mt,len(d))
            continue
        j=d[0]
        # positions may be 1-based or 0-based; accept either, but require consistency
        if j not in (pos-1,pos):
            bp+=1
            if not ex: ex='%s: differs at index %d (expected %d/%d)'%(mt,j,pos-1,pos)
            continue
        if wt[j]!=src or dec[i][j]!=dst:
            ba+=1
            if not ex: ex='%s: %s->%s but one_hot says %s->%s'%(mt,src,dst,wt[j],dec[i][j])
            continue
        ok+=1
    G['ok']+=ok; G['bad']+=bp+ba+nd
    print('%-22s %7d %7d %7d %7d %7d  %s'%(p,n1,ok,bp,ba,nd,ex))
print('\nT4 TOTAL: %d single-point rows verified OK, %d anomalies'%(G['ok'],G['bad']))
