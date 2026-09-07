"""LINE 8 / structural lemma: which input dimensions can carry ddG signal AT ALL?

ddG = pred(mutant) - pred(WT). Any input dimension that is IDENTICAL for the wild type
and the mutant contributes nothing that survives the subtraction, for any network.
So before asking what the model uses, measure what actually MOVES when you mutate.

This needs no model forward pass: it is a property of get_graph / get_unfolded_graph.

Reported per protein, over the real evaluated mutation set:
  * per-block variance across mutants, folded graph and unfolded graph
  * max |folded - unfolded| per block, which says which blocks can carry dG at all

Two separate cancellations are at stake and they must not be confused:
  WT-vs-mutant identity  -> the block cannot contribute to ddG
  folded-vs-unfolded identity -> the block cannot contribute to dG = E_u - E_f directly
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.getcwd())
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph

TR = './data/Processed_K50_dG_datasets/training_data'
MR = './data/Processed_K50_dG_datasets/mutation_datasets'
TM = './data/ThermoMPNN/mega_test.csv'
NANO = 0.1
D_W, FB_W, OH_W = 16, 32, 20
EMB_W = int(CFG.emb_input_dim)
BLK = [('D', 0, D_W), ('Fb', D_W, D_W + FB_W),
       ('emb', D_W + FB_W, D_W + FB_W + EMB_W),
       ('oh', D_W + FB_W + EMB_W, D_W + FB_W + EMB_W + OH_W)]
MAXMUT = int(sys.argv[1]) if len(sys.argv) > 1 else 40

tl = lambda p: torch.load(p, map_location='cpu', weights_only=False)
tm_df = pd.read_csv(TM)
tm_names = tm_df['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
proteins = sorted([p for p in os.listdir(TR) if p in tm_names])

out = []
for name in proteins:
    d = os.path.join(TR, name)
    try:
        mut = pd.read_csv(os.path.join(MR, name + '.csv'))
        mut = mut[~mut['mut_type'].str.contains('ins|del')].reset_index(drop=True)
        coords = tl(os.path.join(d, 'coords_tensor.pt')).squeeze() * NANO
        mask = tl(os.path.join(d, 'mask_tensor.pt')).squeeze()
        oh = tl(os.path.join(d, 'one_hot_encodings.pt'))
        fs = sorted(glob.glob(os.path.join(d, 'prott5_embeddings', 'prott5_embedding_*.pt')),
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        emb = torch.vstack([tl(f) for f in fs])
    except Exception as e:
        print('skip %s: %s' % (name, e)); continue

    idx = set(mut.index)
    idx -= set(mut[mut['ddG_ML'] == '-'].index)
    idx -= set(mut[~mut['name'].isin(tm_df['name'])].index)
    idx = sorted(idx)
    if len(idx) < 5:
        print('skip %s: %d rows' % (name, len(idx))); continue
    if len(idx) > MAXMUT:
        step = len(idx) / float(MAXMUT)
        idx = [idx[int(k * step)] for k in range(MAXMUT)]
    if oh.shape[-1] == OH_W + 1:
        oh = oh[..., :-1]

    Gf, Gu = [], []
    for k in idx:
        o = oh[k].squeeze().float(); e = emb[k].squeeze().float()
        Gf.append(get_graph(coords, o, e, mask))
        Gu.append(get_unfolded_graph(coords, o, e, mask))
    Gf = torch.stack(Gf); Gu = torch.stack(Gu)

    r = dict(protein=name, n_mut=len(idx), n_res=int((mask > 0).sum()))
    vf = Gf.var(dim=0).mean(dim=0)     # variance across mutants, mean over residues
    vu = Gu.var(dim=0).mean(dim=0)
    fu = (Gf - Gu).abs().amax(dim=(0, 1))   # max |folded - unfolded| per feature dim
    for nm, lo, hi in BLK:
        r['varmut_f_' + nm] = float(vf[lo:hi].sum())
        r['varmut_u_' + nm] = float(vu[lo:hi].sum())
        r['maxFU_' + nm] = float(fu[lo:hi].max())
    tot_f = float(vf.sum())
    for nm, lo, hi in BLK:
        r['fracvar_f_' + nm] = float(vf[lo:hi].sum()) / tot_f if tot_f > 0 else float('nan')
    out.append(r)
    print('%-20s n=%3d  varmut(D)=%.3e varmut(Fb)=%.3e varmut(emb)=%.3e varmut(oh)=%.3e | maxFU D=%.4f Fb=%.4g emb=%.4g oh=%.4g'
          % (name, len(idx), r['varmut_f_D'], r['varmut_f_Fb'], r['varmut_f_emb'], r['varmut_f_oh'],
             r['maxFU_D'], r['maxFU_Fb'], r['maxFU_emb'], r['maxFU_oh']))
    sys.stdout.flush()

df = pd.DataFrame(out)
df.to_csv('results/line8/varcheck.csv', index=False)
print()
print('=== SUMMARY over %d proteins ===' % len(df))
print('MUTANT-TO-MUTANT variance in the FOLDED graph (0 => block cannot carry ddG):')
for nm, _, _ in BLK:
    v = df['varmut_f_' + nm]
    print('  %-4s  max over proteins = %.6e   #proteins with exactly 0 = %d/%d'
          % (nm, v.max(), int((v == 0).sum()), len(df)))
print()
print('FOLDED-vs-UNFOLDED max abs difference (0 => block identical in both passes):')
for nm, _, _ in BLK:
    v = df['maxFU_' + nm]
    print('  %-4s  max over proteins = %.6e   #proteins with exactly 0 = %d/%d'
          % (nm, v.max(), int((v == 0).sum()), len(df)))
print()
print('share of mutant-to-mutant variance carried by each block (folded), median over proteins:')
for nm, _, _ in BLK:
    print('  %-4s  %.4f' % (nm, df['fracvar_f_' + nm].median()))
print()
print('wrote results/line8/varcheck.csv')
