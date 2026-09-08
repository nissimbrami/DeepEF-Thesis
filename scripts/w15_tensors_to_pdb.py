#!/usr/bin/env python3
"""W15 step 1 -- write real PDB backbones from our tensors, so a side-chain packer can run.

THE BLOCKER THIS SOLVES. data/MsDs/training_data/<X>.pdb is a DIRECTORY of tensors
(coords.pt [N,4,3], mask.pt [N], aa_seq.pt), not a PDB file. FASPR/SCWRL take PDB text.
So W15 needs a writer before it needs a packer.

UNITS -- READ THIS. train.normalize_batch multiplies coords by NANO_TO_ANGSTROM = 0.1 BEFORE
the model sees them, but the stored coords.pt are the RAW values. This script checks the CA-CA
neighbour distance and only rescales if the geometry is not already in Angstrom (a real
protein has consecutive CA-CA ~3.8 A). Getting this backwards is the w5_dg.py bug: there I
scaled tensors that were already in Angstrom and every neighbour count saturated.

GLYCINE. Atom order is (N, CA, C, CB) and glycine has no CB; the loader stores something there.
Glycine CB rows are DROPPED, not written as a fake atom -- a packer given a glycine with a CB
will either error or silently place a side chain that does not exist.
"""
import argparse, glob, os, sys
import numpy as np, torch

AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
       'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
       'T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
ATOMS = ['N','CA','C','CB']

def load_seq(d, n):
    """Prefer one_hot_encodings.pt (K50 tree); then aa_seq.pt; then the mutation CSV."""
    p = os.path.join(d,'one_hot_encodings.pt')
    if os.path.exists(p):
        try:
            t = torch.load(p, map_location='cpu', weights_only=False).float()
            if t.dim()==3: t=t[0]
            AA20='ACDEFGHIKLMNPQRSTVWY'
            idx=t[:, :20].argmax(-1).tolist()
            seq=''.join(AA20[i] for i in idx)
            if len(seq)>=n: return seq[:n]
        except Exception: pass
    p = os.path.join(d,'aa_seq.pt')
    if os.path.exists(p):
        try:
            s = torch.load(p, map_location='cpu', weights_only=False)
            if isinstance(s,str) and len(s)==n: return s
            if hasattr(s,'__len__') and len(s)==n: return ''.join(list(s))
        except Exception: pass
    base = os.path.basename(d).replace('.pdb','')
    for csv in [f'/home/nissimb/DeepPEF/data_fixed/mutation_datasets/{base}.csv',
                f'/home/nissimb/DeepPEF/data/mutation_datasets/{base}.csv']:
        if os.path.exists(csv):
            try:
                import pandas as pd
                df = pd.read_csv(csv)
                col = [c for c in df.columns if 'seq' in c.lower()]
                if col:
                    wt = df[df.get('mut_type','wt').astype(str)=='wt']
                    seq = (wt if len(wt) else df)[col[0]].iloc[0]
                    if isinstance(seq,str) and len(seq)==n: return seq
            except Exception: pass
    return None

def detect_scale(ca):
    """Return the factor that puts coordinates in Angstrom, decided by MEASUREMENT."""
    if len(ca) < 2: return 1.0
    step = np.linalg.norm(np.diff(ca,axis=0),axis=1)
    med = float(np.median(step))
    if 3.0 < med < 4.6: return 1.0        # already Angstrom
    if 0.30 < med < 0.46: return 10.0     # nanometre -> Angstrom
    return 1.0                            # unknown: do not guess, leave it and report

def write_pdb(coords, mask, seq, out):
    lines=[]; serial=1; nres=0
    for i in range(coords.shape[0]):
        if mask is not None and float(mask[i]) <= 0: continue
        aa = AA3.get(seq[i] if seq else 'A','ALA')
        nres += 1
        for a_i,a in enumerate(ATOMS):
            if a=='CB' and aa=='GLY': continue          # glycine has no CB
            x,y,z = coords[i,a_i]
            if not np.isfinite([x,y,z]).all(): continue
            lines.append(f"ATOM  {serial:5d}  {a:<3s}{aa:>4s} A{nres:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {a[0]}")
            serial += 1
    lines.append("TER"); lines.append("END")
    open(out,'w').write('\n'.join(lines)+'\n')
    return nres, serial-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/nissimb/DeepPEF/data/Processed_K50_dG_datasets/training_data')
    ap.add_argument('--out', default='/home/nissimb/tools/w15_pdb')
    ap.add_argument('--limit', type=int, default=0)
    A=ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    dirs=sorted(d for d in glob.glob(A.src+'/*') if os.path.isdir(d))
    if A.limit: dirs=dirs[:A.limit]
    ok=fail=noseq=0; scales={}
    for d in dirs:
        try:
            cf=os.path.join(d,'coords_tensor.pt');
            cf=cf if os.path.exists(cf) else os.path.join(d,'coords.pt');
            c=torch.load(cf,map_location='cpu',weights_only=False).float().numpy()
            mp=os.path.join(d,'mask_tensor.pt');
            mp=mp if os.path.exists(mp) else os.path.join(d,'mask.pt')
            m=torch.load(mp,map_location='cpu',weights_only=False).numpy() if os.path.exists(mp) else None
            s=load_seq(d,c.shape[0])
            if s is None: noseq+=1
            f=detect_scale(c[:,1,:]); scales[f]=scales.get(f,0)+1
            if f!=1.0: c=c*f
            name=os.path.basename(d).replace('.pdb','')
            nres,nat=write_pdb(c,m,s,os.path.join(A.out,name+'.pdb'))
            if nres>0: ok+=1
            else: fail+=1
        except Exception as e:
            fail+=1
            if fail<=3: print('  FAIL',os.path.basename(d),type(e).__name__,e)
    print(f"wrote {ok} PDBs, {fail} failed, {noseq} without a real sequence (written as poly-ALA)")
    print(f"scale factors applied: {scales}   (1.0 = already Angstrom)")
    print(f"output: {A.out}")
    return 0

if __name__=='__main__': sys.exit(main())
