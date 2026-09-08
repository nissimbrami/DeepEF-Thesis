"""INTEGRITY AUDIT -- production root, real split, leakage.

Uses the SAME roots and the SAME split logic as Megascale-fineTuning/evaluate.py,
so what it audits is what the model actually trained and evaluated on.
"""
import os, sys, glob, json
import numpy as np, pandas as pd, torch

ROOT   = 'data/Processed_K50_dG_datasets/training_data'
MUT    = 'data/Processed_K50_dG_datasets/mutation_datasets'
FIX    = 'data_fixed/mutation_datasets'
TM     = 'data/ThermoMPNN/mega_test.csv'
PNASP  = 'data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv'
AA20   = list("ACDEFGHIKLMNPQRSTVWY")

# ---- reproduce the split exactly ----
dirs = [p for p in os.listdir(ROOT)]
tm = pd.read_csv(TM)
tmset = tm['name'].apply(lambda x: x.split('.')[0]).unique().tolist()
test  = [p for p in dirs if p in tmset]
rest  = [p for p in dirs if p not in tmset]
pnas  = pd.read_csv(PNASP)['protein_name'].tolist()
train = [p for p in rest if p in pnas]
print('dirs=%d  test=%d  rest=%d  train(after PNAS filter)=%d  dropped=%d'
      % (len(dirs), len(test), len(rest), len(train), len(rest)-len(train)))
print('test:', sorted(test))
ov = sorted(set(train) & set(test))
print('S1 train/test NAME overlap:', ov if ov else 'NONE')

def mutpath(p):
    f = os.path.join(FIX, p+'.csv')
    return (f,'data_fixed') if os.path.exists(f) else (os.path.join(MUT,p+'.csv'),'data')

def wtseq(p):
    f,_ = mutpath(p)
    if not os.path.exists(f): return None
    try:
        m = pd.read_csv(f, usecols=['name','aa_seq','mut_type'], nrows=5)
        return str(m['aa_seq'].iloc[0])
    except Exception:
        return None

# ---- S2: sequence-level leakage ----
tseq = {p: wtseq(p) for p in test}
rseq = {}
for p in train:
    s = wtseq(p)
    if s: rseq[p] = s
print('train seqs recovered: %d/%d ; test seqs: %d/%d'
      % (len(rseq), len(train), sum(1 for v in tseq.values() if v), len(test)))

exact = []
for tp, ts in tseq.items():
    if not ts: continue
    for rp, rs in rseq.items():
        if rs == ts: exact.append((tp, rp))
print('S2 EXACT sequence matches train<->test:', exact if exact else 'NONE')

# near-identical: global identity over aligned prefix + containment
def ident(a,b):
    if not a or not b: return 0.0
    n=min(len(a),len(b))
    if n==0: return 0.0
    return sum(1 for i in range(n) if a[i]==b[i])/max(len(a),len(b))
near=[]
for tp, ts in tseq.items():
    if not ts: continue
    best=(0.0,None)
    for rp, rs in rseq.items():
        v=ident(ts,rs)
        if v>best[0]: best=(v,rp)
        if (ts in rs or rs in ts) and abs(len(ts)-len(rs))<=5 and ts!=rs:
            near.append((tp,rp,'containment',round(v,3)))
    if best[0]>=0.60: near.append((tp,best[1],'identity',round(best[0],3)))
print('S3 near-identical (>=0.60 ungapped identity, or containment):', near if near else 'NONE')
# report the max identity per test protein regardless, so "nothing" is quantified
mx = sorted(((max((ident(ts,rs),rp) for rp,rs in rseq.items()), tp) for tp,ts in tseq.items() if ts), reverse=True)
print('\nS3b top-10 max ungapped identity of each test protein to ANY training protein:')
for (v,rp),tp in mx[:10]: print('   %-22s best=%-22s ident=%.3f' % (tp, rp, v))
print('   ... median over %d test proteins = %.3f' % (len(mx), np.median([v for (v,_),_ in mx])))

# ---- also: do any TEST MUTANT sequences appear as training WT sequences? ----
allrs = set(rseq.values())
hits=[]
for tp in test:
    f,_ = mutpath(tp)
    if not os.path.exists(f): continue
    m = pd.read_csv(f, usecols=['aa_seq'])
    s = set(m['aa_seq'].astype(str))
    k = s & allrs
    if k: hits.append((tp, len(k)))
print('S4 test-set VARIANT sequences that equal a training WT sequence:', hits if hits else 'NONE')

# ---- tensors on the production root ----
rows=[]
for p in sorted(test):
    d = os.path.join(ROOT,p); rec={'protein':p}
    f,src = mutpath(p); rec['mutsrc']=src
    m = pd.read_csv(f)
    mf = m[~m['mut_type'].astype(str).str.contains('ins|del')].reset_index(drop=True)
    rec['raw']=len(m); rec['filt']=len(mf)
    try:
        co=torch.load(os.path.join(d,'coords_tensor.pt'),map_location='cpu',weights_only=False)
        mk=torch.load(os.path.join(d,'mask_tensor.pt'),map_location='cpu',weights_only=False)
        oh=torch.load(os.path.join(d,'one_hot_encodings.pt'),map_location='cpu',weights_only=False)
        gg=torch.load(os.path.join(d,'deltaG.pt'),map_location='cpu',weights_only=False)
        eb=torch.vstack([torch.load(x,map_location='cpu',weights_only=False) for x in
             sorted(glob.glob(os.path.join(d,'prott5_embeddings','prott5_embedding_*.pt')),
                    key=lambda x:int(os.path.splitext(x)[0].split('_')[-1]))])
        rec['coords']=tuple(co.shape); rec['mask']=tuple(mk.shape)
        rec['one_hot']=tuple(oh.shape); rec['emb']=tuple(eb.shape); rec['dG']=tuple(gg.shape)
        rec['T1_nvar_agree'] = bool(oh.shape[0]==eb.shape[0]==gg.shape[0])
        rec['T1_len_agree']  = bool(co.shape[0]==oh.shape[1]==eb.shape[1]==mk.shape[0])
        rec['T2_dG_minus_filt'] = int(gg.shape[0]-len(mf))
        rec['T2_dG_minus_raw']  = int(gg.shape[0]-len(m))
        n=min(gg.shape[0],len(mf))
        g=gg.cpu().numpy().reshape(-1)[:n]
        rec['T3_maxdiff_filt']=float(np.nanmax(np.abs(g-pd.to_numeric(mf['deltaG'],errors='coerce').values[:n])))
        rec['T3_nonfinite_dG']=int(np.sum(~np.isfinite(gg.cpu().numpy().reshape(-1))))
        rec['T3_coord_nonfinite']=int(np.sum(~np.isfinite(co.cpu().numpy())))
        # T4 alignment: decode one_hot per row, compare to that row's aa_seq
        ohn=oh.cpu().numpy(); k=min(200,ohn.shape[0]); mism=0
        for i in range(k):
            dec=''.join(AA20[j] for j in ohn[i][:,:20].argmax(axis=1))
            if dec!=str(mf['aa_seq'].iloc[i]): mism+=1
        rec['T4_seq_mismatch_in_first%d'%k]=mism; rec['T4_checked']=k
        rec['T4_mask_allone']=bool(np.all(mk.cpu().numpy()==1))
    except Exception as e:
        rec['err']=repr(e)[:120]
    rows.append(rec)
df=pd.DataFrame(rows)
pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
print('\n=== TENSOR AUDIT on the PRODUCTION root %s ===' % ROOT)
print(df.to_string(index=False))
df.to_csv('/home/nissimb/audit_tensors_prod.csv',index=False)
