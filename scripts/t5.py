"""T5: (a) finiteness of every test tensor; (b) 2K5H-style multi-background files in the TRAIN set."""
import pandas as pd, numpy as np, torch, os, glob, csv, collections
ROOT='data/Processed_K50_dG_datasets/training_data'
D='data/Processed_K50_dG_datasets/mutation_datasets/'
TM='data/ThermoMPNN/mega_test.csv'
PN='data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv'
TEST=['1GYZ','1PSE','1QKH','1QP2','1TUC','1W4H','2BTH','2K1B','2K28','2K5H','2KVS','2KWH',
      '2KXD','2L33','2LQK','2WXC','3DKM','4C26','6EWS','6EWT','6EWU','HEEH_KT_rd6_0746',
      'HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244','r11_1081_TrROS_Hall',
      'r12_757_TrROS_Hall','r18_3_TrROS_Hall']
print('=== (a) finiteness / degenerate values, all test tensors ===')
bad=0
for p in TEST:
    d=os.path.join(ROOT,p); r={}
    for f,k in (('coords_tensor.pt','co'),('mask_tensor.pt','mk'),
                ('one_hot_encodings.pt','oh'),('deltaG.pt','dg')):
        t=torch.load(os.path.join(d,f),map_location='cpu',weights_only=False).cpu().numpy()
        r[k]=(int(np.sum(~np.isfinite(t))), float(np.nanmin(t)), float(np.nanmax(t)))
    eb=torch.vstack([torch.load(x,map_location='cpu',weights_only=False) for x in
         sorted(glob.glob(os.path.join(d,'prott5_embeddings','prott5_embedding_*.pt')),
                key=lambda x:int(os.path.splitext(x)[0].split('_')[-1]))]).cpu().numpy()
    ne=int(np.sum(~np.isfinite(eb)))
    zr=int(np.sum(np.all(eb==0,axis=2)))   # all-zero embedding rows = silent failure
    nb=r['co'][0]+r['mk'][0]+r['oh'][0]+r['dg'][0]+ne+zr
    bad+=nb
    print('%-22s coords[nf=%d %.1f..%.1f] mask[nf=%d %.0f..%.0f] oh[nf=%d] dG[nf=%d %.2f..%.2f] emb[nf=%d zerorows=%d] %s'
          %(p,r['co'][0],r['co'][1],r['co'][2],r['mk'][0],r['mk'][1],r['mk'][2],
            r['oh'][0],r['dg'][0],r['dg'][1],r['dg'][2],ne,zr,'OK' if nb==0 else '<<< ANOMALY'))
print('total anomalies:',bad)

print('\n=== (b) multi-background mutation files (the 2K5H defect) across ALL 368 dirs ===')
dirs=sorted(os.listdir(ROOT))
tm=pd.read_csv(TM); tmset=set(tm['name'].apply(lambda x:x.split('.')[0]))
pn=set(pd.read_csv(PN)['protein_name'].tolist())
multi=[]
for p in dirs:
    f=D+p+'.csv'
    if not os.path.exists(f): continue
    names=set()
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if str(r.get('mut_type','')).strip().lower()=='wt':
                names.add(r['name'].split('_wt')[0].strip())
    if len(names)>1:
        role = 'TEST' if p in tmset else ('TRAIN' if p in pn else 'unused')
        multi.append((p,len(names),role,sorted(names)[:4]))
print('files with >1 WT background: %d of %d'%(len(multi),len(dirs)))
for p,n,role,ex in multi: print('   %-24s %d backgrounds  [%s]  %s'%(p,n,role,ex))
if not multi: print('   NONE')
