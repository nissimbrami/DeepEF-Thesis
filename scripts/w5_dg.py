"""W5-dG: score the burial lever on the metric it actually acts on.

WHY THIS EXISTS. gate_w5.py passes 18/18 and not one assertion scores a performance
metric -- they are tensor shapes, byte-identity when off, and hydropathy ordering.
W5's only scheduled scoring is the S7 information factorial, whose decision rule
EFFECT_MIN is defined on POOLED ddG. But W5's own help text says burial is ZERO in
the unfolded state and the folded-minus-unfolded delta IS the hydrophobic driving
force: the lever acts on absolute dG and on b_p, not on ddG. THE METRIC RULE says
ddG cancels anything identical between WT and mutant, so a reference-state lever
judged on pooled ddG is read as the residual, not the lever. That is how the coil
was nearly discarded.

THREE SEPARABLE QUESTIONS, AND WHICH ONE EACH ARM ANSWERS
---------------------------------------------------------
Q1  SIGNAL (model-free, fully answerable on CPU today).
    Does burial, computed on the 28 WT structures, predict the per-protein WT dG
    error b_p? This is the b_p-side question and it needs no model at all: if the
    burial summary of a protein tracks the offset the model gets wrong on that
    protein, the information W5 injects is real and aimed at the right target.
    A null here would sink W5 before a single GPU-hour is spent.
    --> arm: --q1, scored across ALL 10 eval CSVs.

Q2  PLUMBING (zero-init, provable identity, measures NO effect).
    Load the W5-off checkpoint into a W5-on model, and zero the three new weight
    COLUMNS in fc1_gat/fc1_gcn. The forward pass is then provably identical to the
    W5-off model. This establishes that the block is wired where the model reads it
    and that nothing else moved. It deliberately measures ZERO performance change --
    that identity IS the result. It is not evidence for or against the lever.
    --> arm: --q2.

Q3  EFFECT (requires GPU training -- NOT measurable here, and this script refuses
    to fake it). See the trap below.

THE TRAP THIS SCRIPT REFUSES TO FALL INTO
-----------------------------------------
W5 widens the residue feature block 1092 -> 1095. You CANNOT switch the flag on and
forward-pass a pre-trained W5-off checkpoint: the new columns would hit weights that
were never trained, and the output would be noise attributable to the untrained
columns, not to burial. On this architecture it is even more clear-cut than that --
it does not merely produce noise, it HARD-FAILS:

    size mismatch for fc1_gcn.weight: checkpoint [64,52] vs model [64,55]
    size mismatch for fc1_gat.weight: checkpoint [64,36] vs model [64,39]

so there is no silent-noise path to fall into by accident. Note also that the
top-level fc1 is 1096-wide in BOTH configs (fc_in_dim = 2*(36+0) + 1024 = 1096 is
fixed, because fc2_gat/fc2_gcn project back to fixed internal widths). Only fc1_gat
and fc1_gcn grow. Any arm here that reported a dG number from untrained columns
would be the SIGNATURE FAILURE -- code runs, completes, reports a number, feature
never read. Q3 is therefore left to GPU and its sbatch line is written down, not
submitted.
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse
import glob
import json
import sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import (get_graph, get_unfolded_graph, compute_burial,
                         compute_hse, _KD_NORM)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='Megascale-fineTuning/models/'
                'PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/'
                'kf_all_epoch_14.pt')
ap.add_argument('--out', default='results/w5_dg.json')
ap.add_argument('--device', default='cpu')
ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
ap.add_argument('--eval_glob', default='eval_results/abl_*.csv')
ap.add_argument('--q1', action='store_true', help='burial signal vs WT error (model-free)')
ap.add_argument('--q2', action='store_true', help='zero-init plumbing identity')
A = ap.parse_args()
dev = torch.device(A.device)


def tload(p):
    return torch.load(p, map_location=dev, weights_only=False)


import pandas as pd
tm = pd.read_csv(A.tm_path)['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
proteins = sorted([p for p in os.listdir(A.tensor_root) if p in tm])
print('test proteins: %d' % len(proteins))

OUT = {}


def load_prot(name):
    """Row 0 of every variant tensor is the WILD TYPE.

    one_hot_encodings.pt is [V, L, 21], not [V, L, 20]. The trailing column is the
    pad/unknown slot and the model never sees it: Megascale-fineTuning/dataset.py:27
    does batch['one_hot'][..., :-1] before the graph is built. Slicing to :20 here is
    therefore matching the canonical pipeline, not a guess. Getting this wrong is not
    a loud failure -- it raises inside a per-protein try/except and every protein is
    silently skipped, leaving a full table of NaN that still prints and still writes
    JSON. That is the SIGNATURE FAILURE, and it happened on the first run of this
    script.
    """
    d = os.path.join(A.tensor_root, name)
    coords = tload(os.path.join(d, 'coords_tensor.pt')).to(dev).squeeze()
    mask = tload(os.path.join(d, 'mask_tensor.pt')).to(dev).squeeze()
    oh = tload(os.path.join(d, 'one_hot_encodings.pt'))
    ed = os.path.join(d, 'prott5_embeddings')
    if os.path.isdir(ed):
        emb = tload(os.path.join(ed, sorted(os.listdir(ed))[0]))
    else:
        emb = tload(ed + '.pt')
    dg = tload(os.path.join(d, 'deltaG.pt'))
    oh0 = oh[0].squeeze().to(dev).float()
    if oh0.shape[-1] == 21:
        oh0 = oh0[..., :-1]
    if oh0.shape[-1] != 20:
        raise ValueError('%s: one-hot width %d, expected 20 or 21' % (name, oh0.shape[-1]))
    return coords, mask, oh0, emb[0].squeeze().to(dev).float(), dg


def burial_summary(name):
    """Per-protein burial descriptors computed on the WT structure only.

    These are exactly the quantities the W5 block injects, reduced to one number per
    protein so they can be correlated with a per-protein offset b_p. mean_bur_hyd is
    the product column -- the hydrophobic driving-force term W5 exists to carry.
    """
    coords, mask, oh, emb, dg = load_prot(name)
    v = (mask > 0).float()
    n = int(v.sum().item())
    hyd = (oh @ _KD_NORM.to(oh.device).to(oh.dtype).unsqueeze(1)).squeeze(1)
    out = {'length': float(n)}
    for mode, fn in (('count', compute_burial), ('hse', compute_hse)):
        b = fn(coords, mask).squeeze(1)
        m = min(int(b.shape[0]), int(hyd.shape[0]), n)
        bv = b[:m]
        hv = hyd[:m]
        out['mean_bur_' + mode] = float(bv.mean())
        out['frac_bur_gt0.5_' + mode] = float((bv > 0.5).float().mean())
        out['mean_bur_hyd_' + mode] = float((bv * hv).mean())
    out['mean_hyd'] = float(hyd[:n].mean())
    return out


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan'), int(a.size)
    return float(np.corrcoef(a, b)[0, 1]), int(a.size)


def perm_p(a, b, r_obs, n_perm=2000, seed=0):
    """Two-sided permutation p for a correlation on ~28 points."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or not np.isfinite(r_obs):
        return float('nan')
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(a, rng.permutation(b))[0, 1]) >= abs(r_obs) - 1e-12:
            cnt += 1
    return (cnt + 1.0) / (n_perm + 1.0)


def wt_offsets(csv):
    """b_p per protein = mean WT dG error. The WT row is the one with ddG == 0."""
    df = pd.read_csv(csv)
    wt = df[df['ddG'].abs() < 1e-9] if 'ddG' in df.columns else df
    if len(wt) == 0:
        wt = df
    out = {}
    for p, d in wt.groupby('protein'):
        out[str(p)] = float((d['pred_deltaG'] - d['deltaG']).mean())
    return out


FEATS = ['mean_bur_count', 'mean_bur_hse', 'mean_bur_hyd_count',
         'mean_bur_hyd_hse', 'frac_bur_gt0.5_count', 'mean_hyd']

# ================================================================ Q1
if A.q1:
    print('')
    print('=' * 78)
    print('Q1  BURIAL SIGNAL vs WT dG ERROR  (model-free; the b_p-side question)')
    print('=' * 78)
    print('Does the burial summary of a protein predict the offset the model gets')
    print('wrong on that protein? No forward pass, no training, no new weights.')

    feats = {}
    skipped = []
    for name in proteins:
        try:
            feats[name] = burial_summary(name)
        except Exception as e:
            skipped.append((name, str(e)))
            print('  skip %s: %s' % (name, e))
    print('burial summaries computed for %d proteins (%d skipped)'
          % (len(feats), len(skipped)))
    # A silently-empty feature table still prints a full table of NaN and still
    # writes JSON. Refuse to report that as a measurement.
    if len(feats) < 20:
        print('FATAL: only %d/%d proteins produced a burial summary. A correlation '
              'table built on this would be NaN or meaningless -- refusing to report '
              'it as a result.' % (len(feats), len(proteins)))
        sys.exit(1)

    csvs = sorted(glob.glob(A.eval_glob))
    print('eval CSVs: %d' % len(csvs))

    per_csv = {}
    for c in csvs:
        g = wt_offsets(c)
        # The other silent-empty path: if the CSV protein names never match the
        # tensor dir names, every list below is empty and every r is NaN.
        matched = [p for p in g if p in feats]
        if len(matched) < 20:
            print('FATAL: %s -- only %d/%d CSV proteins matched a burial summary; '
                  'the join is broken, not the correlation'
                  % (os.path.basename(c), len(matched), len(g)))
            sys.exit(1)
        rec = {}
        for f in FEATS:
            xs, ys = [], []
            for p, err in g.items():
                if p in feats:
                    xs.append(feats[p][f])
                    ys.append(err)
            r, n = pearson(xs, ys)
            rec[f] = {'r': r, 'n': n}
        per_csv[os.path.basename(c)] = rec

    hdr = '  '.join('%8s' % os.path.basename(c)[4:12] for c in csvs)
    print('')
    print('%-24s %s' % ('feature', hdr))
    summary = {}
    for f in FEATS:
        rs = [per_csv[os.path.basename(c)][f]['r'] for c in csvs]
        arr = np.array(rs, float)
        pos = int((arr > 0).sum())
        neg = int((arr < 0).sum())
        consistent = max(pos, neg)
        mean_r = float(np.nanmean(arr))
        summary[f] = {'per_csv_r': rs, 'mean_r': mean_r,
                      'sign_consistent': '%d/%d' % (consistent, len(rs))}
        print('%-24s %s   mean=%+.3f  %d/%d' %
              (f, '  '.join('%+8.3f' % x for x in rs), mean_r, consistent, len(rs)))

    ref = [c for c in csvs if 'calib_ctrl_repro2' in c]
    if ref:
        g = wt_offsets(ref[0])
        print('')
        print('permutation p (2000 shuffles) on %s:' % os.path.basename(ref[0]))
        for f in FEATS:
            xs, ys = [], []
            for p, err in g.items():
                if p in feats:
                    xs.append(feats[p][f])
                    ys.append(err)
            r, n = pearson(xs, ys)
            pv = perm_p(xs, ys, r)
            summary[f]['ref_r'] = r
            summary[f]['ref_perm_p'] = pv
            print('  %-24s r=%+.3f  n=%d  p=%.4f' % (f, r, n, pv))

    OUT['Q1'] = {'question': 'does WT burial predict per-protein WT dG error b_p (model-free)',
                 'per_csv': per_csv, 'summary': summary,
                 'n_csvs': len(csvs), 'n_proteins': len(feats),
                 'burial_summaries': feats}

# ================================================================ Q2
if A.q2:
    print('')
    print('=' * 78)
    print('Q2  ZERO-INIT PLUMBING IDENTITY  (measures NO effect, by construction)')
    print('=' * 78)
    print('Loads the W5-off checkpoint into a W5-ON model and zeroes the three new')
    print('weight COLUMNS. The forward pass must then be identical to W5-off. This')
    print('proves the block is wired where the model reads it. It is NOT evidence')
    print('that burial helps -- a zeroed column cannot help.')

    sd = torch.load(A.ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    CFG.burial_features = True
    CFG.burial_mode = 'count'
    m_on = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
               dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
    try:
        m_on.load_state_dict(sd)
        naive_err = ['UNEXPECTED: strict load succeeded']
    except RuntimeError as e:
        naive_err = [s.strip() for s in str(e).split('\n')[1:] if s.strip()]
    print('')
    print('Naive "just switch the flag on" strict load -> RuntimeError:')
    for line in naive_err:
        print('   ' + line)

    sd2 = dict(sd)
    for key, nblk in (('fc1_gat.weight', 16), ('fc1_gcn.weight', 32)):
        W = sd[key]
        z = torch.zeros(W.shape[0], 3, dtype=W.dtype)
        sd2[key] = torch.cat([W[:, :nblk], z, W[:, nblk:]], dim=1)
    miss, unexp = m_on.load_state_dict(sd2, strict=False)
    print('')
    print('zero-init load: missing=%d unexpected=%d' % (len(miss), len(unexp)))
    m_on.eval()

    CFG.burial_features = False
    m_off = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
    m_off.load_state_dict(sd)
    m_off.eval()

    max_abs = 0.0
    n_ok = 0
    rows = []
    with torch.no_grad():
        for name in proteins:
            try:
                coords, mask, oh, emb, dg = load_prot(name)
            except Exception:
                continue
            CFG.burial_features = False
            f0 = get_graph(coords, oh, emb, mask).unsqueeze(0)
            u0 = get_unfolded_graph(coords, oh, emb, mask).unsqueeze(0)
            d0 = float(m_off(u0).reshape(-1)[0] - m_off(f0).reshape(-1)[0])
            CFG.burial_features = True
            f1 = get_graph(coords, oh, emb, mask).unsqueeze(0)
            u1 = get_unfolded_graph(coords, oh, emb, mask).unsqueeze(0)
            d1 = float(m_on(u1).reshape(-1)[0] - m_on(f1).reshape(-1)[0])
            CFG.burial_features = False
            max_abs = max(max_abs, abs(d1 - d0))
            n_ok += 1
            rows.append({'protein': name, 'dG_off': d0, 'dG_zeroinit': d1,
                         'diff': d1 - d0})
    print('')
    print('proteins compared: %d' % n_ok)
    print('max |dG(zero-init W5 on) - dG(W5 off)| = %.3e' % max_abs)
    ident = max_abs < 1e-4
    if ident:
        print('PLUMBING IDENTITY: PASS -- the block is inert when zeroed, so it is '
              'wired where the model reads it')
    else:
        print('PLUMBING IDENTITY: FAIL -- the block is NOT where fc1_gat/fc1_gcn read it')
    print('REMINDER: this is a ZERO by construction. It says nothing about whether')
    print('burial improves dG. That question needs training (Q3, GPU).')
    OUT['Q2'] = {'question': 'is the W5 block wired where the model reads it '
                             '(zero-init identity; measures no effect)',
                 'max_abs_diff': max_abs, 'identical': bool(ident),
                 'n_proteins': n_ok, 'naive_strict_load_error': naive_err,
                 'per_protein': rows}
    CFG.burial_features = False

OUT['Q3'] = {'question': 'does W5 reduce MAE / std(err) on absolute WT dG',
             'status': 'NOT MEASURABLE WITHOUT TRAINING',
             'why': 'W5 widens fc1_gat 36->39 and fc1_gcn 52->55. A W5-off checkpoint '
                    'has no weights for those columns; loading it into a W5-on model '
                    'raises a size mismatch. Any number produced by forcing it would '
                    'be attributable to untrained columns, not to burial.'}

if OUT:
    d = os.path.dirname(A.out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    json.dump(OUT, open(A.out, 'w'), indent=2)
    print('')
    print('wrote %s' % A.out)
