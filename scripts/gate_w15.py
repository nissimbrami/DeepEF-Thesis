#!/usr/bin/env python3
"""Gate for W15. The decisive check is #4: does the block DIFFER between WT and mutant?
A block that is identical across variants cancels exactly in ddG and is inert -- the W5 bug."""
import sys, glob, os, torch
sys.path.insert(0,'/home/nissimb/DeepPEF')
sys.path.insert(0,'/home/nissimb/DeepPEF/scripts')
from w15_block import w15_block

P=F=0
def chk(name, cond, extra=''):
    global P,F
    if cond: P+=1; print(f"PASS  {name}  {extra}")
    else:    F+=1; print(f"FAIL  {name}  {extra}")

d=glob.glob('/home/nissimb/DeepPEF/data/Processed_K50_dG_datasets/training_data/2PTL*/')[0]
x=torch.load(os.path.join(d,'coords_tensor.pt'),map_location='cpu',weights_only=False).float()
oh=torch.load(os.path.join(d,'one_hot_encodings.pt'),map_location='cpu',weights_only=False).float()
m=torch.load(os.path.join(d,'mask_tensor.pt'),map_location='cpu',weights_only=False).float()
wt=oh[0]
print(f"protein 2PTL  N={x.shape[0]}  variants={oh.shape[0]}\n")

B=w15_block(x,wt,m,folded=True)
chk("1 shape is [N,4]", tuple(B.shape)==(x.shape[0],4), str(tuple(B.shape)))
chk("2 all finite", bool(torch.isfinite(B).all()))
U=w15_block(x,wt,m,folded=False)
chk("3 unfolded is EXACTLY zero", bool((U==0).all()), "-> folded-minus-unfolded IS the packing term")

# THE DECISIVE TEST: build a real mutant one_hot and compare
mut=wt.clone()
pos=None
for i in range(len(wt)):
    if wt[i,:20].argmax().item()!=0:   # not already ALA
        pos=i; break
mut[pos,:20]=0; mut[pos,0]=1           # mutate to ALA (smallest reach)
Bm=w15_block(x,mut,m,folded=True)
diff=(B-Bm).abs()
chk("4 block DIFFERS between WT and mutant", float(diff.max())>1e-6,
    f"max|delta|={float(diff.max()):.4f} at position {pos}")
chk("5 the difference is LOCAL to the mutated position",
    int((diff.sum(1)>1e-6).sum())==1, f"{int((diff.sum(1)>1e-6).sum())} position(s) changed")
chk("6 env_density (col1) is UNCHANGED by mutation", float((B[:,1]-Bm[:,1]).abs().max())<1e-6,
    "-> it is geometry only, as designed")
chk("7 interaction col2 DOES change", float((B[:,2]-Bm[:,2]).abs().max())>1e-6,
    f"max|delta|={float((B[:,2]-Bm[:,2]).abs().max()):.4f}")

# a bulky residue in a crowded pocket must score higher than in open space
env=B[:,1]
big=wt.clone(); big[:, :20]=0; big[:,18]=1     # all TRP
Bb=w15_block(x,big,m,folded=True)
hi=env.argmax().item(); lo=env.argmin().item()
chk("8 same residue, crowded vs open, gives DIFFERENT values",
    abs(float(Bb[hi,2]-Bb[lo,2]))>1e-3,
    f"TRP in crowd {float(Bb[hi,2]):.4f} vs open {float(Bb[lo,2]):.4f}")
chk("9 masked positions are zero",
    bool((B[m.reshape(-1)==0].abs().sum()==0)) if (m.reshape(-1)==0).any() else True)
chk("10 no column has zero variance", all(float(B[:,j].std())>1e-8 for j in range(4)),
    " sd=" + " ".join(f"{float(B[:,j].std()):.4f}" for j in range(4)))
print(f"\nW15 GATE: {P} passed, {F} failed")
sys.exit(1 if F else 0)
