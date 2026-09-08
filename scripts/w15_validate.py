#!/usr/bin/env python3
"""Validate W15 against the burial the model already has, and against residue identity.

TWO CHECKS THAT DECIDE WHETHER W15 IS WORTH GPU TIME:
  A. corr(our SASA, existing burial) should be STRONG and NEGATIVE. If it is near zero the
     reconstruction or the SASA is wrong. If it is -1.0 the feature is redundant.
  B. R^2 of predicting each column from ONE-HOT ALONE. This is the decisive test: if one-hot
     explains ~all of a column, that column IS a descriptor table (one_hot @ T) and cannot add
     information -- the exact arithmetic that killed LORO, W6 and half of W12.
"""
import glob, os, sys, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/nissimb/DeepPEF')
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT='/home/nissimb/DeepPEF'
from train_utils import compute_burial, compute_hse

rows_f, rows_b, rows_oh = [], [], []
n=0
for f in sorted(glob.glob('/home/nissimb/tools/w15_features/*.npy')):
    name=os.path.basename(f)[:-4]
    d=glob.glob(f'{ROOT}/data/Processed_K50_dG_datasets/training_data/{name}*/')
    if not d: continue
    try:
        c=torch.load(os.path.join(d[0],'coords_tensor.pt'),map_location='cpu',weights_only=False).float()
        m=torch.load(os.path.join(d[0],'mask_tensor.pt'),map_location='cpu',weights_only=False)
        oh=torch.load(os.path.join(d[0],'one_hot_encodings.pt'),map_location='cpu',weights_only=False).float()
    except Exception: continue
    if oh.dim()==3: oh=oh[0]
    F=np.load(f)
    bur=compute_burial(c,m).squeeze().numpy()
    keep=(m.numpy()>0)
    L=min(len(F), keep.sum(), len(oh))
    if L<10: continue
    rows_f.append(F[:L]); rows_b.append(bur[keep][:L]); rows_oh.append(oh[keep][:L,:20].numpy())
    n+=1
F=np.concatenate(rows_f); B=np.concatenate(rows_b); OH=np.concatenate(rows_oh)
print(f"proteins {n}   residues {len(F)}\n")
names=['sc_sasa','buried_frac','contacts','clash']
print("=== CHECK A: correlation with the burial the model ALREADY has ===")
for j,nm in enumerate(names):
    r=np.corrcoef(F[:,j],B)[0,1]
    flag='  <-- expected strong negative' if j==0 else ''
    print(f"  corr({nm:<12}, burial) = {r:+.4f}{flag}")
print("\n=== CHECK B: can ONE-HOT alone explain each column? ===")
print("   (high R2 => the column is one_hot @ T and adds NOTHING new)")
for j,nm in enumerate(names):
    y=F[:,j]; pr=np.zeros(len(y))
    for tr,te in KFold(5,shuffle=True,random_state=0).split(OH):
        pr[te]=RidgeCV(alphas=np.logspace(-2,3,15)).fit(OH[tr],y[tr]).predict(OH[te])
    r2=1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
    verdict='REDUNDANT (descriptor table)' if r2>0.85 else ('partly new' if r2>0.5 else 'GENUINELY NEW')
    print(f"  {nm:<12} R2 from one-hot = {r2:+.4f}   {verdict}")
print("\n=== residual variance not explained by residue identity ===")
for j,nm in enumerate(names):
    y=F[:,j]; pr=np.zeros(len(y))
    for tr,te in KFold(5,shuffle=True,random_state=0).split(OH):
        pr[te]=RidgeCV(alphas=np.logspace(-2,3,15)).fit(OH[tr],y[tr]).predict(OH[te])
    print(f"  {nm:<12} residual sd = {(y-pr).std():.4f}  (raw sd {y.std():.4f})")
