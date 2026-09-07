"""LINE 7 -- unfolded reference-state sweep on a FROZEN checkpoint.

The Flory coil d(i,j) = b*|i-j|^nu is an INPUT transformation. Nothing about it is
learned, so every (nu, b) pair can be evaluated by re-running inference only. E_f
(folded pass) does not depend on the coil at all, so it is computed ONCE per protein
and cached; each grid cell then costs one unfolded forward pass per protein.

METRICS (all on ABSOLUTE wild-type dG, per the metric rule):
  bias      = mean(pred - true)            <- a CONSTANT; Pearson-invariant, cancels in ddG
  std_err   = std(pred - true)             <- this IS std(b_p), the offset SPREAD
  mae       = mean|pred - true|            <- confounds the two above
  mae_c     = mean|err - mean(err)|        <- MAE after removing the constant: honest spread
  corr      = corr(pred, true) over the test proteins
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse, json, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
import train_utils as TU

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', default='results/unfolded_sweep.json')
ap.add_argument('--device', default='cpu')
ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
ap.add_argument('--part', default='all')
ap.add_argument('--coord_scale', type=float, default=1.0,
                help='multiply loaded coords by this BEFORE any graph call. 1.0 = raw Angstrom '
                     '(what w0_dg.py did). 0.1 = what train.normalize_batch actually feeds the '
                     'model during training.')
A = ap.parse_args()
dev = torch.device(A.device)

def tload(p): return torch.load(p, map_location=dev, weights_only=False)

import pandas as pd
tm = pd.read_csv(A.tm_path)['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
proteins = sorted([p for p in os.listdir(A.tensor_root) if p in tm])
print('test proteins: %d' % len(proteins))

model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
try:
    model, _, _, _, _ = load_checkpoint(A.ckpt, model)
except Exception:
    model.load_state_dict(torch.load(A.ckpt, map_location=dev))
model.eval()

DATA = []
with torch.no_grad():
    for name in proteins:
        d = os.path.join(A.tensor_root, name)
        try:
            coords = tload(os.path.join(d, 'coords_tensor.pt')).to(dev).squeeze() * A.coord_scale
            dg = tload(os.path.join(d, 'deltaG.pt'))
            mask = tload(os.path.join(d, 'mask_tensor.pt')).to(dev).squeeze()
            oh = tload(os.path.join(d, 'one_hot_encodings.pt'))
            ed = os.path.join(d, 'prott5_embeddings')
            emb = tload(os.path.join(ed, sorted(os.listdir(ed))[0])) if os.path.isdir(ed) else tload(ed + '.pt')
            oh0 = oh[0].squeeze().to(dev).float()
            emb0 = emb[0].squeeze().to(dev).float()
            true = float(dg.reshape(-1)[0])
            E_f = float(model(get_graph(coords, oh0, emb0, mask).unsqueeze(0)).reshape(-1)[0])
            ca = coords[:, 1, :]
            b_fit = float(torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean())
            DATA.append(dict(name=name, coords=coords, oh=oh0, emb=emb0, mask=mask,
                             true=true, E_f=E_f, b_fit=b_fit,
                             L=float((mask > 0).float().sum())))
        except Exception as e:
            print('  skip %s: %s' % (name, e))
            continue
print('loaded %d proteins' % len(DATA))
if len(DATA) < 20:
    print('FATAL')
    sys.exit(1)
print('fitted b (model units): mean %.4f sd %.4f -> Angstrom mean %.3f'
      % (np.mean([d['b_fit'] for d in DATA]), np.std([d['b_fit'] for d in DATA]),
         np.mean([d['b_fit'] for d in DATA]) / A.coord_scale))

TRUE = np.array([d['true'] for d in DATA])


def evaluate(tag, setup):
    setup()
    preds = []
    with torch.no_grad():
        for d in DATA:
            g = get_unfolded_graph(d['coords'], d['oh'], d['emb'], d['mask']).unsqueeze(0)
            E_u = float(model(g).reshape(-1)[0])
            preds.append(E_u - d['E_f'])
    p = np.array(preds)
    e = p - TRUE
    return dict(tag=tag, mae=float(np.abs(e).mean()), bias=float(e.mean()),
                std_err=float(e.std(ddof=1)), mae_c=float(np.abs(e - e.mean()).mean()),
                corr=float(np.corrcoef(p, TRUE)[0, 1]) if p.std() > 1e-12 else float('nan'),
                pred_sd=float(p.std(ddof=1)), preds=[float(v) for v in p])


def reset():
    CFG.flory_unfolded = False
    CFG.coil_b = 'fitted'
    CFG.coil_channels = 'broadcast'
    CFG.unfolded_emb = 'full'
    CFG.flory_nu = 0.5
    TU._COIL_B_FIXED = TU._COIL_B_FIXED_ANGSTROM * A.coord_scale


def mk_base():
    def f():
        reset()
    return f


def mk_fitted(nu, chan='broadcast'):
    def f():
        reset()
        CFG.flory_unfolded = True
        CFG.flory_nu = nu
        CFG.coil_b = 'fitted'
        CFG.coil_channels = chan
    return f


def mk_fixed(nu, b_ang, chan='broadcast'):
    def f():
        reset()
        CFG.flory_unfolded = True
        CFG.flory_nu = nu
        CFG.coil_b = 'fixed'
        CFG.coil_channels = chan
        TU._COIL_B_FIXED = b_ang * A.coord_scale
    return f


RES = []
t0 = time.time()
RES.append(evaluate('BASE_tridiagonal', mk_base()))
r = RES[-1]
print('BASE  MAE %7.4f bias %+7.4f std_err %7.4f mae_c %7.4f corr %+.4f (%.1fs)'
      % (r['mae'], r['bias'], r['std_err'], r['mae_c'], r['corr'], time.time() - t0))

NUS = [0.33, 0.40, 0.45, 0.50, 0.55, 0.588, 0.62, 0.70, 0.80, 0.90, 1.00]
BS = [1.0, 2.0, 3.0, 3.8, 4.5, 4.9, 5.0, 5.82, 7.0, 9.0, 12.0]

if A.part in ('all', 'grid'):
    for nu in NUS:
        for b in BS:
            r = evaluate('nu%.3f_b%.2f' % (nu, b), mk_fixed(nu, b))
            r['nu'] = nu; r['b'] = b; r['arm'] = 'fixed_b'
            RES.append(r)
            print('nu=%.3f b=%5.2f  MAE %7.4f  bias %+7.4f  std_err %7.4f  mae_c %7.4f  corr %+.4f'
                  % (nu, b, r['mae'], r['bias'], r['std_err'], r['mae_c'], r['corr']))
    print('grid done %.1fs' % (time.time() - t0))

if A.part in ('all', 'extra'):
    for nu in NUS:
        r = evaluate('fittedb_nu%.3f' % nu, mk_fitted(nu))
        r['nu'] = nu; r['b'] = None; r['arm'] = 'fitted_b'
        RES.append(r)
        print('FITTEDb nu=%.3f  MAE %7.4f bias %+7.4f std_err %7.4f mae_c %7.4f corr %+.4f'
              % (nu, r['mae'], r['bias'], r['std_err'], r['mae_c'], r['corr']))
    for chan in ['ca_only', 'offset']:
        for nu in [0.5, 0.588]:
            for b in [4.9, 5.82]:
                r = evaluate('%s_nu%.3f_b%.2f' % (chan, nu, b), mk_fixed(nu, b, chan))
                r['nu'] = nu; r['b'] = b; r['arm'] = chan
                RES.append(r)
                print('%-7s nu=%.3f b=%.2f MAE %7.4f bias %+7.4f std_err %7.4f mae_c %7.4f corr %+.4f'
                      % (chan, nu, b, r['mae'], r['bias'], r['std_err'], r['mae_c'], r['corr']))

reset()
json.dump(dict(checkpoint=A.ckpt, n=len(DATA),
               proteins=[d['name'] for d in DATA],
               true=[d['true'] for d in DATA],
               b_fit_model_units=[d['b_fit'] for d in DATA],
               lengths=[d['L'] for d in DATA],
               coord_scale=A.coord_scale,
               coil_coord_scale_const=TU._COIL_COORD_SCALE,
               gaussian_coef=CFG.gaussian_coef,
               results=RES), open(A.out, 'w'), indent=1)
print('WROTE %s  (%d cells, %.1fs)' % (A.out, len(RES), time.time() - t0))
