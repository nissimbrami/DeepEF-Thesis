"""How much residue-level chemistry does ProtT5 already carry?

This decides whether W6 descriptors can add INFORMATION or only regularisation.
A descriptor block is a deterministic function of residue identity, so if identity and its
chemical properties are already recoverable from the embedding, the block adds nothing the
network could not extract itself.

RESIDUE-DISJOINT SPLIT, deliberately: fit on 19 residue types and predict the 20th. A
row-random split is contaminated for the descriptor question, because every occurrence of a
residue shares one descriptor vector, so a random split leaks the answer.
"""
import os, sys, json, math
sys.path.insert(0, '/home/nissimb/DeepPEF')
import torch

D = '/groups/keasar_group/casp15/meytav/protein_tensors'
AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy, formal charge at pH 7, and van der Waals volume.
KD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
CH = dict(zip(AA, [0,0,-1,-1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0]))
VOL= dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))

X, Y = [], []
n_prot = 0
for p in sorted(os.listdir(D)):
    d = os.path.join(D, p)
    ep = os.path.join(d, 'prott5_embeddings')
    op = os.path.join(d, 'one_hot_encodings.pt')
    if not (os.path.isdir(ep) and os.path.exists(op)):
        continue
    fs = sorted(os.listdir(ep))
    if not fs:
        continue
    try:
        emb = torch.load(os.path.join(ep, fs[0]), map_location='cpu', weights_only=False)
        oh = torch.load(op, map_location='cpu', weights_only=False)
    except Exception:
        continue
    if emb.dim() != 3 or oh.dim() != 3:
        continue
    v = emb[0].float()                 # [L, 1024] one variant (the WT row)
    ids = oh[0][:, :20].argmax(-1)     # drop the 21st column, as train.py:317 does
    L = min(v.shape[0], ids.shape[0])
    X.append(v[:L]); Y.append(ids[:L])
    n_prot += 1
    if n_prot >= 40:
        break

X = torch.cat(X); Y = torch.cat(Y)
print('proteins=%d  residues=%d  emb_dim=%d' % (n_prot, X.shape[0], X.shape[1]))

def r2(pred, true):
    ss = float(((true - pred) ** 2).sum()); tt = float(((true - true.mean()) ** 2).sum())
    return 1 - ss / tt if tt > 0 else float('nan')

# LEAVE-ONE-RESIDUE-TYPE-OUT: fit on 19 types, predict the held-out type's property.
for name, table in (('hydropathy', KD), ('charge', CH), ('volume', VOL)):
    t = torch.tensor([table[a] for a in AA], dtype=torch.float32)
    target = t[Y]
    preds, trues = [], []
    for held in range(20):
        tr = Y != held; te = Y == held
        if int(te.sum()) == 0 or int(tr.sum()) < 50:
            continue
        A = X[tr]; b = target[tr]
        A1 = torch.cat([A, torch.ones(A.shape[0], 1)], 1)
        # ridge, because 1024 dims on a few thousand rows is ill-conditioned
        lam = 10.0
        W = torch.linalg.solve(A1.T @ A1 + lam * torch.eye(A1.shape[1]), A1.T @ b)
        A1te = torch.cat([X[te], torch.ones(int(te.sum()), 1)], 1)
        preds.append(A1te @ W); trues.append(target[te])
    pr = torch.cat(preds); tv = torch.cat(trues)
    print('%-11s leave-one-residue-type-out R^2 = %+.3f' % (name, r2(pr, tv)))

# identity, row-random (an UPPER bound: contaminated for the descriptor question)
idx = torch.randperm(X.shape[0]); k = int(0.7 * len(idx))
tr, te = idx[:k], idx[k:]
Wc = torch.linalg.lstsq(X[tr], torch.nn.functional.one_hot(Y[tr], 20).float()).solution
acc = float(((X[te] @ Wc).argmax(-1) == Y[te]).float().mean())
print('identity   row-random held-out accuracy = %.3f  (chance %.3f)' % (acc, 1 / 20))
json.dump({'n_proteins': n_prot, 'n_residues': int(X.shape[0]), 'identity_acc': acc},
          open('/home/nissimb/DeepPEF/results/emb_probe.json', 'w'), indent=1)
print('wrote results/emb_probe.json')
