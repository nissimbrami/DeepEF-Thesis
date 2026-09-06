"""W0 -- unfolded-state channel ablation.  No training. Decides factor D of the factorial.

dG = E_unfolded - E_folded, and 77-88% of the across-protein variance in dG lives in
E_unfolded, with corr(E_u, wt_err) = 0.865.  So the per-protein offset b_p is manufactured
in the reference state.  Four channels feed it: D(16) and Fb(32) are geometry -- the coil
fixes those -- while one_hot(20) and emb(1024) are sequence content, which the coil does
NOT touch.  And corr(wt_err, length) <= 0.12, so the variance is not extensive.

Therefore the coil may repair the two channels that were probably not the cause.  This
script tests that before 12 of the 48 factorial cells are spent on it.

Conditions, all altering the UNFOLDED pass only (the folded pass is never touched):
  base   unfolded graph as-is
  noemb  ProtT5 block zeroed in the unfolded graph only
  noOH   one-hot block zeroed in the unfolded graph only
  coil   CFG.flory_unfolded = True
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse
import json
import sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'Megascale-fineTuning'))

from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph, load_checkpoint

COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5 = 'prott5_embeddings'

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', default='results/w0.json')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--limit', type=int, default=0, help='0 = all test proteins')
ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
A = ap.parse_args()
dev = torch.device(A.device)

# --- block layout, derived from CFG, never hardcoded ---
# get_graph: Fh = torch.cat([D, Fb, emb, one_hot], dim=1)
D_W, FB_W, OH_W = 16, 32, 20
EMB_W = int(CFG.emb_input_dim)
EMB_LO = D_W + FB_W
EMB_HI = EMB_LO + EMB_W
OH_LO = EMB_HI

import pandas as pd
tm = pd.read_csv(A.tm_path)['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
all_dirs = os.listdir(A.tensor_root)
test_proteins = [p for p in all_dirs if p in tm]
if A.limit:
    test_proteins = test_proteins[:A.limit]
print('test proteins: %d' % len(test_proteins))

model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
try:
    model, _, _, _, _ = load_checkpoint(A.ckpt, model)
except Exception:
    model.load_state_dict(torch.load(A.ckpt, map_location=dev))
model.eval()


def tload(path):
    """Tensors in this dataset were saved on CUDA; a login node has no GPU, so every
    load must be mapped explicitly. weights_only=True cannot deserialise a CUDA storage
    onto CPU, hence weights_only=False with map_location."""
    return torch.load(path, map_location=dev, weights_only=False)


def load_emb(protein_dir):
    d = os.path.join(protein_dir, PROTT5)
    if os.path.isdir(d):
        f = sorted(os.listdir(d))[0]
        return tload(os.path.join(d, f))
    return tload(d + '.pt')


rows = []
with torch.no_grad():
    for name in test_proteins:
        pdir = os.path.join(A.tensor_root, name)
        try:
            coords = tload(os.path.join(pdir, COORDS)).to(dev)
            dg = tload(os.path.join(pdir, DELTA_G))
            mask = tload(os.path.join(pdir, MASKS)).to(dev)
            oh = tload(os.path.join(pdir, ONE_HOT))
            emb = load_emb(pdir)
        except Exception as e:
            print('  skip %s: %s' % (name, e))
            continue
        try:
            coords = coords.squeeze()
            mask = mask.squeeze()
            oh0 = oh[0].squeeze().to(dev).float()
            emb0 = emb[0].squeeze().to(dev).float()
            true_wt_dg = float(dg.reshape(-1)[0])

            # folded: computed ONCE, never altered, reused across all conditions
            folded = get_graph(coords, oh0, emb0, mask).unsqueeze(0)
            E_f = float(model(folded).reshape(-1)[0])

            was = getattr(CFG, 'flory_unfolded', False)
            CFG.flory_unfolded = False
            base_u = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
            CFG.flory_unfolded = True
            coil_u = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
            CFG.flory_unfolded = was  # restore immediately, never leave it set

            # Post-hoc slice-zeroing of the ASSEMBLED graph. The unambiguous claim is
            # "the ProtT5 block in the unfolded graph is all zeros". Zeroing the input
            # emb before the call is a DIFFERENT operation, because get_unfolded_graph
            # runs F.normalize(emb, p=2, dim=0), and normalising zeros is not the same
            # as normalising then zeroing.
            noemb_u = base_u.clone()
            noemb_u[:, :, EMB_LO:EMB_HI] = 0.0
            noOH_u = base_u.clone()
            noOH_u[:, :, OH_LO:OH_LO + OH_W] = 0.0

            nv = float((mask > 0).float().sum())
            r = dict(protein=name, length=nv, true_wt_dg=true_wt_dg, E_f=E_f)
            for cond, g in (('base', base_u), ('noemb', noemb_u),
                            ('noOH', noOH_u), ('coil', coil_u)):
                E_u = float(model(g).reshape(-1)[0])
                r['E_u_' + cond] = E_u
                # reference config uses DG_LENGTH_NORM='none', so the divisor is 1.0
                r['wt_err_' + cond] = (E_u - E_f) - true_wt_dg
            rows.append(r)
            print('  %-10s N=%4d  E_f=%9.3f  base=%9.3f  noemb=%9.3f  noOH=%9.3f  coil=%9.3f'
                  % (name, nv, E_f, r['E_u_base'], r['E_u_noemb'],
                     r['E_u_noOH'], r['E_u_coil']))
        except Exception as e:
            print('  FAIL %s: %s' % (name, e))
            continue

if len(rows) < 20:
    print('')
    print('FATAL: only %d proteins produced a result (need >= 20).' % len(rows))
    print('A partial answer must not be read as a decision.')
    sys.exit(1)


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


L = [r['length'] for r in rows]
res = {}
var_base = None
for cond in ('base', 'noemb', 'noOH', 'coil'):
    Eu = [r['E_u_' + cond] for r in rows]
    we = [r['wt_err_' + cond] for r in rows]
    v = float(np.var(Eu))
    if cond == 'base':
        var_base = v
    res[cond] = dict(var_Eu=v,
                     r_vs_base=(v / var_base if var_base else float('nan')),
                     corr_Eu_wterr=corr(Eu, we),
                     corr_Eu_length=corr(Eu, L),
                     mean_Eu=float(np.mean(Eu)),
                     std_Eu=float(np.std(Eu)))

print('')
print('%-8s %14s %9s %18s %18s' % ('cond', 'var(E_u)', 'r_x',
                                   'corr(E_u,wt_err)', 'corr(E_u,length)'))
for c in ('base', 'noemb', 'noOH', 'coil'):
    d = res[c]
    print('%-8s %14.4f %9.4f %18.4f %18.4f'
          % (c, d['var_Eu'], d['r_vs_base'], d['corr_Eu_wterr'], d['corr_Eu_length']))

# --- D0 decision table, AUTOPILOT section D0 ---
r_noemb = res['noemb']['r_vs_base']
r_noOH = res['noOH']['r_vs_base']
r_coil = res['coil']['r_vs_base']
below = {}
for k, v in (('noemb', r_noemb), ('noOH', r_noOH), ('coil', r_coil)):
    if v < 0.5:
        below[k] = v
side = None
if len(below) >= 2:
    win = min(below, key=lambda k: below[k])
    side = [k for k in below if k != win]
    verdict = ('ENTANGLED: %s all collapse. Take the smaller r: %s (r=%.4f); '
               'log the other(s) as side arms.'
               % (', '.join(sorted(below)), win, below[win]))
    factor_d = '--unfolded_emb {full,zero}' if win == 'noemb' else '--flory_unfolded'
elif r_noemb < 0.5:
    verdict = 'ProtT5 is the offset channel (r_noemb=%.4f < 0.5).' % r_noemb
    factor_d = '--unfolded_emb {full,zero}  -- COIL DEMOTED to a 2-cell side arm'
elif r_coil < 0.5:
    verdict = 'Geometry is the channel (r_coil=%.4f < 0.5).' % r_coil
    factor_d = '--flory_unfolded  (as planned)'
elif r_noOH < 0.5:
    verdict = 'Composition + extensive sum (r_noOH=%.4f < 0.5).' % r_noOH
    factor_d = '--flory_unfolded, plus --dg_length_norm as a 2-cell side arm'
else:
    verdict = ('Nothing collapses (all r >= 0.5). The variance is in the learned weights, '
               'not the inputs. WEAKEST EVIDENCE -- log H4 for the write-up.')
    factor_d = '--flory_unfolded  (weakest evidence)'

print('')
print('=== D0 VERDICT ===')
print(verdict)
print('FACTOR D := %s' % factor_d)
if side:
    print('side arm(s): %s' % ', '.join(side))

out = dict(checkpoint=A.ckpt, n_proteins=len(rows), device=str(dev),
           block_layout=dict(D=D_W, Fb=FB_W, emb=EMB_W, one_hot=OH_W,
                             emb_slice=[EMB_LO, EMB_HI],
                             oh_slice=[OH_LO, OH_LO + OH_W]),
           conditions={'base': 'unfolded graph as-is',
                       'noemb': 'ProtT5 block zeroed in the unfolded graph only',
                       'noOH': 'one-hot block zeroed in the unfolded graph only',
                       'coil': 'CFG.flory_unfolded = True'},
           results=res, d0_verdict=verdict, factor_d=factor_d, side_arms=side,
           per_protein=rows)
outdir = os.path.dirname(A.out)
if outdir:
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
with open(A.out, 'w') as f:
    json.dump(out, f, indent=2)
print('')
print('wrote %s' % A.out)
