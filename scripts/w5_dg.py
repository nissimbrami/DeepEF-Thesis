"""T12: score W5 burial on the metric it ACTS on.

gate_w5 passes 18/18 and every one of those assertions is mechanical -- shapes,
byte-identity when off, hydropathy ordering. Not one scores performance. Meanwhile W5's
own help text says burial is ZERO in the unfolded state and the folded-minus-unfolded
delta IS the hydrophobic driving force. So W5 acts on dG / b_p, yet its only scheduled
scoring is the S7 factorial whose EFFECT_MIN is defined on POOLED ddG. That is the coil
error one lever later.

THE TRAP THIS SCRIPT DOES NOT FALL INTO: W5 widens the vector 1092 -> 1095. A checkpoint
trained WITHOUT W5 has no weights for those 3 columns, so flipping the flag on and doing a
forward pass would feed untrained weights and measure noise, not the lever. That number
would be meaningless. So we do NOT do that.

What CAN be measured with no training, and is the honest b_p-side question:
  does the burial SIGNAL, computed on the 28 WT structures, predict the WT error b_p?
If burial predicts b_p, feeding burial to the model is motivated. If it does not, W5 is
not a b_p lever and the S7 ddG scoring is the least of its problems.
"""
import csv, io, json, math, os, sys
sys.path.insert(0, '/home/nissimb/DeepPEF')
import torch
from model.model_cfg import CFG
import train_utils as tu

EV = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/abl_calib_ctrl_repro2_e14.csv'

def pear(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def spear(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        for pos, i in enumerate(s): r[i] = pos
        return r
    return pear(rank(x), rank(y))

# b_p per protein = the WT error on absolute dG (WT is the row with ddG == 0)
rows = {}
with open(EV) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r['protein'], []).append(
            (float(r['deltaG']), float(r['pred_deltaG']), float(r['ddG'])))
bp = {}
for p, v in rows.items():
    wt = [t for t in v if t[2] == 0.0]
    if wt: bp[p] = wt[0][1] - wt[0][0]

# burial computed from the SAME per-protein tensors train.py loads
# (train.py:28 COORDS='coords_tensor.pt', one directory per protein)
TDIR = '/groups/keasar_group/casp15/meytav/protein_tensors'
feat = {}
missing = []
for p in bp:
    d = os.path.join(TDIR, p)
    cf, mf = os.path.join(d, 'coords_tensor.pt'), os.path.join(d, 'mask_tensor.pt')
    if not os.path.exists(cf):
        missing.append(p); continue
    x = torch.load(cf, map_location='cpu', weights_only=False).float()
    m = (torch.load(mf, map_location='cpu', weights_only=False).float()
         if os.path.exists(mf) else torch.ones(x.shape[0]))
    # MEASURED, not assumed: these stored tensors are already in ANGSTROM, and
    # compute_burial's cutoff is an Angstrom cutoff. Scaling by 0.1 here saturates
    # every neighbour count to exactly 1.0 (std 0.0) and every correlation becomes nan.
    # Verified on 2KVS/2BTH: angstrom -> mean 0.48/0.41 std 0.18/0.16; 0.1x -> 1.0/0.0.
    b = tu.compute_burial(x, m)
    v = b[m > 0].flatten()
    if v.numel() == 0: continue
    feat[p] = dict(mean_burial=float(v.mean()),
                   frac_buried=float((v > v.median()).float().mean()),
                   length=float(int(m.sum())))
if missing:
    print('no tensor dir for %d protein(s): %s' % (len(missing), ','.join(missing[:6])))
print('proteins with burial: %d / %d' % (len(feat), len(bp)))

if feat:
    ps = sorted(feat)
    out = {'eval_csv': EV, 'n': len(ps), 'correlations': {}}
    tgt = {'b_p': [bp[p] for p in ps], 'abs_b_p': [abs(bp[p]) for p in ps]}
    print('\n%-16s %-10s %8s %8s' % ('feature', 'target', 'pearson', 'spearman'))
    print('-'*46)
    for fn in ('mean_burial', 'frac_buried', 'length'):
        xs = [feat[p][fn] for p in ps]
        for tn, ys in tgt.items():
            r, rs = pear(xs, ys), spear(xs, ys)
            out['correlations']['%s~%s' % (fn, tn)] = {'pearson': r, 'spearman': rs, 'n': len(ps)}
            print('%-16s %-10s %8.3f %8.3f' % (fn, tn, r, rs))
    print('\nn=%d: |r| < %.3f is indistinguishable from zero at p=0.05'
          % (len(ps), 1.96/math.sqrt(len(ps)-3+1e-9)))
    json.dump(out, open('results/w5_on_dg.json', 'w'), indent=1)
    print('wrote results/w5_on_dg.json')
