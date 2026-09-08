"""W0-dG: re-score the unfolded-state levers on the metric they were DESIGNED for.

WHY THIS EXISTS. The first W0 run scored every condition by corr(E_u, wt_err), where
wt_err is a ddG-side quantity. But the Flory coil replaces the distance map with
d(i,j) = b*|i-j|^nu -- a function of SEQUENCE SEPARATION ONLY. A point mutation changes
neither the chain length nor the positions, so that matrix is IDENTICAL for wild type
and mutant and cancels EXACTLY in output - output[0]. Scoring a dG lever by a ddG metric
measures the cancellation, not the lever.

So: score the SAME conditions against the ABSOLUTE wild-type dG instead.
  wt_dg_pred = E_u - E_f     vs     wt_dg_true
Report per condition:
  MAE  |pred - true|                 <- what the coil is supposed to reduce
  corr(pred, true) across proteins   <- can the model place proteins on one scale?
  std(err)                           <- this IS std(b_p), the offset spread
A lever that helps here helps ddG INDIRECTLY, by closing the pooled->PP gap.
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph, load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', default='results/w0_dg.json')
ap.add_argument('--device', default='cpu')
ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
# --- P1: the Flory nu / b sweep -------------------------------------------------
# nu is the coil scaling exponent in d(i,j) = b*|i-j|^nu. CFG.flory_nu defaults to
# 0.5 (theta / ideal chain); a denatured protein in water is a self-avoiding walk at
# nu ~= 0.588 (Kohn 2004 PNAS: Rg ~ N^0.598 over 28 proteins). NEVER swept before.
ap.add_argument('--nu', type=float, nargs='+', default=None,
                help='one or more coil exponents; each is run as its own condition. '
                     'Default None = no sweep (historical behaviour, CFG default 0.5).')
# b is the effective segment length, in ANGSTROM. train_utils stores it as a module
# constant _COIL_B_FIXED ALREADY multiplied by _COIL_COORD_SCALE (0.1), because the
# model is trained on coordinates in model units, not Angstrom. We therefore set that
# module constant, in the same units, rather than passing Angstrom into the graph.
ap.add_argument('--b', type=float, nargs='+', default=None,
                help='one or more segment lengths in ANGSTROM (5.82 = ours, 4.97 = Kohn). '
                     'Only meaningful with coil_b=fixed. Default None = leave at 5.82.')
ap.add_argument('--sweep_only', action='store_true',
                help='skip the five historical CONDS and run only the nu x b grid.')
ap.add_argument('--ref_fix', default='',
                help="optional 'PROT=value' override of the reference WT dG "
                     "(2K5H=4.805470 corrects the known bad reference row).")
A = ap.parse_args()
dev = torch.device(A.device)

def tload(p): return torch.load(p, map_location=dev, weights_only=False)

import pandas as pd
tm = pd.read_csv(A.tm_path)['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
proteins = [p for p in os.listdir(A.tensor_root) if p in tm]
print('test proteins: %d' % len(proteins))

model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
try: model, _, _, _, _ = load_checkpoint(A.ckpt, model)
except Exception: model.load_state_dict(torch.load(A.ckpt, map_location=dev))
model.eval()

import train_utils as _TU
_COORD_SCALE = getattr(_TU, '_COIL_COORD_SCALE', 0.1)
_B_DEFAULT_ANG = getattr(_TU, '_COIL_B_FIXED_ANGSTROM', 5.82)

CONDS = [] if A.sweep_only else ['base', 'coil', 'coil_fixed_b', 'coil_ca_only', 'noemb']
# SWEEP holds (name, nu, b_angstrom). Every entry is coil + coil_b=fixed + broadcast
# channels, i.e. EXACTLY the historical 'coil_fixed_b' condition with nu and b varied.
# The nu=0.5 / b=5.82 cell must therefore reproduce coil_fixed_b to machine precision:
# that equality is the sweep's own no-op gate.
SWEEP = []
if A.nu is not None:
    for _nu in A.nu:
        for _b in (A.b if A.b is not None else [_B_DEFAULT_ANG]):
            SWEEP.append(('nu%.3f_b%.2f' % (_nu, _b), float(_nu), float(_b)))
ALL = CONDS + [t[0] for t in SWEEP]
print('conditions: %s' % ', '.join(ALL))
_REF_FIX = {}
if A.ref_fix:
    _k, _v = A.ref_fix.split('=')
    _REF_FIX[_k] = float(_v)
    print('reference override: %s -> %s' % (_k, _REF_FIX[_k]))
rows = []
with torch.no_grad():
    for name in proteins:
        d = os.path.join(A.tensor_root, name)
        try:
            coords = tload(os.path.join(d,'coords_tensor.pt')).to(dev).squeeze()
            dg     = tload(os.path.join(d,'deltaG.pt'))
            mask   = tload(os.path.join(d,'mask_tensor.pt')).to(dev).squeeze()
            oh     = tload(os.path.join(d,'one_hot_encodings.pt'))
            ed     = os.path.join(d,'prott5_embeddings')
            emb    = tload(os.path.join(ed, sorted(os.listdir(ed))[0])) if os.path.isdir(ed) else tload(ed+'.pt')
        except Exception as e:
            print('  skip %s: %s' % (name, e)); continue
        try:
            oh0 = oh[0].squeeze().to(dev).float(); emb0 = emb[0].squeeze().to(dev).float()
            true = float(dg.reshape(-1)[0])
            if name in _REF_FIX:
                true = _REF_FIX[name]
            E_f = float(model(get_graph(coords, oh0, emb0, mask).unsqueeze(0)).reshape(-1)[0])
            r = dict(protein=name, true=true, E_f=E_f, length=float((mask>0).float().sum()))
            for c in CONDS:
                CFG.flory_unfolded = c.startswith('coil')
                CFG.coil_b = 'fixed' if c=='coil_fixed_b' else 'fitted'
                CFG.coil_channels = 'ca_only' if c=='coil_ca_only' else 'broadcast'
                CFG.unfolded_emb = 'zero' if c=='noemb' else 'full'
                g = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
                E_u = float(model(g).reshape(-1)[0])
                r['pred_'+c] = E_u - E_f
            for cname, cnu, cb in SWEEP:
                CFG.flory_unfolded = True
                CFG.coil_b = 'fixed'
                CFG.coil_channels = 'broadcast'
                CFG.unfolded_emb = 'full'
                CFG.flory_nu = cnu
                # b is a MODULE constant in model units, not a CFG field.
                _TU._COIL_B_FIXED = cb * _COORD_SCALE
                g = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
                E_u = float(model(g).reshape(-1)[0])
                r['pred_' + cname] = E_u - E_f
            CFG.flory_unfolded=False; CFG.coil_b='fitted'; CFG.coil_channels='broadcast'; CFG.unfolded_emb='full'
            CFG.flory_nu = 0.5
            _TU._COIL_B_FIXED = _B_DEFAULT_ANG * _COORD_SCALE
            rows.append(r)
        except Exception as e:
            print('  FAIL %s: %s' % (name, e)); continue

if len(rows) < 20:
    print('FATAL: only %d proteins' % len(rows)); sys.exit(1)

true = np.array([r['true'] for r in rows])
STD_TRUE = float(true.std())
# The degenerate attractor: a condition that DELETES the prediction sends pred -> const,
# hence b_p -> -true and std(b_p) -> std(true WT dG). A 'win' that lands there is void.
print('\nstd(true WT dG) = %.4f   <-- the degenerate attractor for std(b_p)' % STD_TRUE)
print('\n%-18s %10s %10s %12s %10s %12s %10s' %
      ('condition','MAE','corr','std(b_p)','bias','std(predWT)','n_under'))
res={}
for c in ALL:
    pred = np.array([r['pred_'+c] for r in rows])
    err = pred - true
    mae=float(np.abs(err).mean()); sd=float(err.std()); bias=float(err.mean())
    sp=float(pred.std())
    cr=float(np.corrcoef(pred,true)[0,1]) if pred.std()>0 else float('nan')
    cbp=float(np.corrcoef(err,true)[0,1]) if err.std()>0 else float('nan')
    res[c]=dict(mae=mae, corr=cr, std_err=sd, bias=bias, std_pred=sp,
                corr_bp_true=cbp, n_under=int((err<0).sum()), n=len(err))
    print('%-18s %10.4f %10.4f %12.4f %10.4f %12.4f %10d'
          % (c,mae,cr,sd,bias,sp,res[c]['n_under']))

if 'base' in res:
    b=res['base']
    print('\n--- change vs base (negative MAE / std = BETTER) ---')
    for c in ALL[1:]:
        print('  %-18s dMAE=%+.4f  dstd(b_p)=%+.4f  dcorr=%+.4f'
              % (c, res[c]['mae']-b['mae'], res[c]['std_err']-b['std_err'], res[c]['corr']-b['corr']))
print('\nNOTE: std(err) across proteins IS std(b_p), the offset spread the thesis targets.')
json.dump(dict(checkpoint=A.ckpt, n=len(rows), metric='ABSOLUTE wild-type dG',
               std_true_wt_dg=STD_TRUE, ref_fix=_REF_FIX,
               sweep=[dict(name=n_, nu=nu_, b_angstrom=b_) for n_, nu_, b_ in SWEEP],
               results=res, per_protein=rows), open(A.out,'w'), indent=2)
print('wrote %s' % A.out)
