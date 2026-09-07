"""LINE 8 -- what does the trained network actually USE?

Block ablation at INFERENCE on a frozen checkpoint. The 1092-dim node feature is
    [ D(16) | Fb(32) | emb(1024) | one_hot(20) ]
built by train_utils.get_graph (folded) and get_unfolded_graph (unfolded).

For every test protein and every mutation we build the folded and unfolded graphs
ONCE, then zero a block slice post-assembly and re-run the readout. Zeroing after
assembly, not before, is deliberate: get_graph applies F.normalize(emb, p=2, dim=0),
and normalising zeros is not the same operation as normalising then zeroing.

Each block is ablated in three PLACES:
    _f   folded pass only
    _u   unfolded pass only
    _b   both passes
A block that only matters in the unfolded pass is a REFERENCE-STATE channel and
cancels out of ddG; a block that matters in both is a within-protein channel.

We also log E_folded and E_unfolded per row so cov(E_u, E_f) can be computed.
"""
import os, sys, json, argparse, gc, glob
os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.getcwd())
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph, load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', default='results/line8_blocks.csv')
ap.add_argument('--device', default='cpu')
ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
ap.add_argument('--mut_root', default='./data/Processed_K50_dG_datasets/mutation_datasets')
ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
ap.add_argument('--mb', type=int, default=32)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
A = ap.parse_args()
dev = torch.device(A.device)
NANO_TO_ANGSTROM = 0.1

D_W, FB_W, OH_W = 16, 32, 20
EMB_W = int(CFG.emb_input_dim)
SL = {
    'D':   (0, D_W),
    'Fb':  (D_W, D_W + FB_W),
    'emb': (D_W + FB_W, D_W + FB_W + EMB_W),
    'oh':  (D_W + FB_W + EMB_W, D_W + FB_W + EMB_W + OH_W),
}
WIDTH = D_W + FB_W + EMB_W + OH_W
print('block layout %s  width=%d' % (SL, WIDTH))

CONDS = [('base', None, None)]
for blk in ('D', 'Fb', 'emb', 'oh'):
    for place in ('f', 'u', 'b'):
        CONDS.append(('%s_%s' % (blk, place), blk, place))
CONDS.append(('geom_b', ('D', 'Fb'), 'b'))
CONDS.append(('geom_f', ('D', 'Fb'), 'f'))
CONDS.append(('seq_b', ('emb', 'oh'), 'b'))

tm_df = pd.read_csv(A.tm_path)
tm_names = tm_df['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
proteins = sorted([p for p in os.listdir(A.tensor_root) if p in tm_names])
proteins = [p for i, p in enumerate(proteins) if i % A.nshard == A.shard]
print('proteins this shard: %d' % len(proteins))
print(proteins)

model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout='sum').to(dev)
try:
    model, _, _, _, _ = load_checkpoint(A.ckpt, model)
except Exception:
    model.load_state_dict(torch.load(A.ckpt, map_location=dev))
model.eval()


def tload(p):
    return torch.load(p, map_location=dev, weights_only=False)


def load_emb(d):
    fs = sorted(glob.glob(os.path.join(d, 'prott5_embedding_*.pt')),
                key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    return torch.vstack([tload(f).to(dev) for f in fs])


def zero_(g, blk):
    if blk is None:
        return g
    blks = blk if isinstance(blk, tuple) else (blk,)
    g = g.clone()
    for b in blks:
        lo, hi = SL[b]
        g[:, :, lo:hi] = 0.0
    return g


rows = []
with torch.no_grad():
    for name in proteins:
        pdir = os.path.join(A.tensor_root, name)
        mpath = os.path.join(A.mut_root, name + '.csv')
        try:
            mut = pd.read_csv(mpath)
            mut = mut[~mut['mut_type'].str.contains('ins|del')].reset_index(drop=True)
            coords = tload(os.path.join(pdir, 'coords_tensor.pt')).squeeze() * NANO_TO_ANGSTROM
            dg = tload(os.path.join(pdir, 'deltaG.pt'))
            mask = tload(os.path.join(pdir, 'mask_tensor.pt')).squeeze()
            oh = tload(os.path.join(pdir, 'one_hot_encodings.pt'))
            emb = load_emb(os.path.join(pdir, 'prott5_embeddings'))
        except Exception as e:
            print('  skip %s: %s' % (name, e))
            continue

        idx = set(mut.index)
        idx -= set(mut[mut['ddG_ML'] == '-'].index)
        idx -= set(mut[~mut['name'].isin(tm_df['name'])].index)
        idx = sorted(idx)
        if len(idx) < 5:
            print('  skip %s: only %d rows' % (name, len(idx)))
            continue
        mutsel = mut.loc[idx]
        dgs = dg.reshape(-1)[idx]
        ohs = oh[idx]
        embs = emb[idx]
        if ohs.shape[-1] == OH_W + 1:
            ohs = ohs[..., :-1]

        n = len(idx)
        acc = {c[0]: [] for c in CONDS}
        Ef_base, Eu_base = [], []
        for j in range(0, n, A.mb):
            oj = ohs[j:j + A.mb].to(dev).float()
            ej = embs[j:j + A.mb].to(dev).float()
            fg = torch.stack([get_graph(coords, oj[k].squeeze(), ej[k].squeeze(), mask)
                              for k in range(oj.size(0))])
            ug = torch.stack([get_unfolded_graph(coords, oj[k].squeeze(), ej[k].squeeze(), mask)
                              for k in range(oj.size(0))])
            assert fg.shape[-1] == WIDTH, 'width %d != %d' % (fg.shape[-1], WIDTH)
            for tag, blk, place in CONDS:
                f2 = zero_(fg, blk) if place in ('f', 'b') else fg
                u2 = zero_(ug, blk) if place in ('u', 'b') else ug
                E = model(torch.cat([f2, u2], dim=0)).reshape(-1)
                h = E.size(0) // 2
                Ef, Eu = E[:h], E[h:]
                acc[tag].append((Eu - Ef).cpu().numpy())
                if tag == 'base':
                    Ef_base.append(Ef.cpu().numpy())
                    Eu_base.append(Eu.cpu().numpy())
            del fg, ug
            gc.collect()

        base_dg = dgs.cpu().numpy().astype(float)
        Efb = np.concatenate(Ef_base)
        Eub = np.concatenate(Eu_base)
        cat = {tag: np.concatenate(acc[tag]) for tag, _, _ in CONDS}
        for k in range(n):
            r = dict(protein=name, mut=mutsel['mut_type'].iloc[k],
                     deltaG=float(base_dg[k]), E_f=float(Efb[k]), E_u=float(Eub[k]))
            for tag, _, _ in CONDS:
                r['pred_' + tag] = float(cat[tag][k])
            rows.append(r)
        print('  %-10s n=%4d  E_f=%9.3f  E_u=%9.3f  pred=%8.3f true=%8.3f'
              % (name, n, Efb[0], Eub[0], cat['base'][0], base_dg[0]))
        sys.stdout.flush()

df = pd.DataFrame(rows)
od = os.path.dirname(A.out)
if od and not os.path.isdir(od):
    os.makedirs(od)
df.to_csv(A.out, index=False)
print('WROTE %s  rows=%d proteins=%d' % (A.out, len(df), df['protein'].nunique()))
