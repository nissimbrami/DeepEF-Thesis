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

CONDS = ['base', 'coil', 'coil_fixed_b', 'coil_ca_only', 'noemb']
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
            CFG.flory_unfolded=False; CFG.coil_b='fitted'; CFG.coil_channels='broadcast'; CFG.unfolded_emb='full'
            rows.append(r)
        except Exception as e:
            print('  FAIL %s: %s' % (name, e)); continue

if len(rows) < 20:
    print('FATAL: only %d proteins' % len(rows)); sys.exit(1)

true = np.array([r['true'] for r in rows])
print('\n%-14s %10s %10s %10s %10s' % ('condition','MAE','corr','std(err)','bias'))
res={}
for c in CONDS:
    pred = np.array([r['pred_'+c] for r in rows])
    err = pred - true
    mae=float(np.abs(err).mean()); sd=float(err.std()); bias=float(err.mean())
    cr=float(np.corrcoef(pred,true)[0,1]) if pred.std()>0 else float('nan')
    res[c]=dict(mae=mae, corr=cr, std_err=sd, bias=bias)
    print('%-14s %10.4f %10.4f %10.4f %10.4f' % (c,mae,cr,sd,bias))

b=res['base']
print('\n--- change vs base (negative MAE / std = BETTER) ---')
for c in CONDS[1:]:
    print('  %-14s dMAE=%+.4f  dstd(err)=%+.4f  dcorr=%+.4f'
          % (c, res[c]['mae']-b['mae'], res[c]['std_err']-b['std_err'], res[c]['corr']-b['corr']))
print('\nNOTE: std(err) across proteins IS std(b_p), the offset spread the thesis targets.')
json.dump(dict(checkpoint=A.ckpt, n=len(rows), metric='ABSOLUTE wild-type dG',
               results=res, per_protein=rows), open(A.out,'w'), indent=2)
print('wrote %s' % A.out)
