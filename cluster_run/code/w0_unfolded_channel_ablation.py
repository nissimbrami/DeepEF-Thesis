"""W0 — unfolded-state channel ablation.  AUTOPILOT S1, decision node D0.

No training.  For each test protein, wild type only, compute E_unfolded under four
ablations of the UNFOLDED pass only, and report how much each one collapses the
across-protein variance of E_u.

Why this exists
---------------
dG = E_unfolded - E_folded, and 77-88% of the across-protein variance in dG lives in
E_unfolded, with corr(E_u, wt_err) = 0.865.  So the per-protein offset b_p is
manufactured in the reference state.  Four channels feed the unfolded pass:

    D distance (16) | Fb bonded (32) | emb ProtT5 (1024) | one_hot (20)

The Flory coil (lever D) rewrites D and Fb.  It touches neither sequence channel.
But corr(wt_err, length) <= 0.12, so the variance is not extensive -- it is sequence
content, which only one_hot and emb carry.  Hence the coil may repair the two channels
that were not the cause.  This script decides that before 12 of the 48 factorial cells
are spent on it.

Conditions, all applied to the unfolded graph only; the folded pass is never touched,
so E_f is computed once per protein and reused:

    base   unfolded graph as-is, CFG.flory_unfolded = False
    noemb  the ProtT5 block of the unfolded graph zeroed
    noOH   the one-hot block of the unfolded graph zeroed
    coil   CFG.flory_unfolded = True

Ablation is applied to the ASSEMBLED graph tensor, not to the inputs.  get_unfolded_graph
runs F.normalize(emb, p=2, dim=0) before concatenating, and normalising a zero tensor is
not the same as normalising and then zeroing.  Zeroing the assembled block is the
unambiguous operation: "the ProtT5 block of the unfolded graph is all zeros".

Usage
-----
    python w0_unfolded_channel_ablation.py --ckpt <path.pt> --out results/w0.json
run from /home/nissimb/DeepPEF.
"""

import os

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_DISABLED', 'true')

import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd
import torch

# The repo root holds model/ and train_utils.py; train.py lives in Megascale-fineTuning.
# sys.path[0] is this script's own directory, so both must be added explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (_ROOT, os.path.join(_ROOT, 'Megascale-fineTuning')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# train.py parses sys.argv at import time; give it nothing to chew on.
_argv_backup = sys.argv
sys.argv = [sys.argv[0]]
from model.hydro_net import PEM                     # noqa: E402
from model.model_cfg import CFG                     # noqa: E402
from train_utils import (get_graph, get_unfolded_graph,   # noqa: E402
                         load_checkpoint)
import train as T                                   # noqa: E402
sys.argv = _argv_backup

CONDITIONS = ['base', 'noemb', 'noOH', 'coil']


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def block_bounds(emb_dim):
    """Column ranges of the assembled graph: [D(16) | Fb(32) | emb(E) | one_hot(20)]."""
    return {'emb': (48, 48 + emb_dim), 'one_hot': (48 + emb_dim, 48 + emb_dim + 20)}


def unfolded_graph_for(cond, coords, one_hot_wt, emb_wt, mask, bounds):
    """Build the unfolded graph for one condition.  CFG.flory_unfolded is read at call
    time via getattr, so it is set immediately before the call and restored after."""
    prev = getattr(CFG, 'flory_unfolded', False)
    try:
        CFG.flory_unfolded = (cond == 'coil')
        g = get_unfolded_graph(coords, one_hot_wt, emb_wt, mask)
    finally:
        CFG.flory_unfolded = prev

    if cond == 'noemb':
        lo, hi = bounds['emb']
        g = g.clone()
        g[:, lo:hi] = 0.0
    elif cond == 'noOH':
        lo, hi = bounds['one_hot']
        g = g.clone()
        g[:, lo:hi] = 0.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint .pt to probe')
    ap.add_argument('--out', default='results/w0.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='0 = all test proteins')
    ap.add_argument('--min_proteins', type=int, default=20,
                    help='fail if fewer than this many proteins produced a result')
    # train.py defines these inside its training functions, not at module level.
    ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
    ap.add_argument('--mut_root', default='./data/Processed_K50_dG_datasets/mutation_datasets')
    args = ap.parse_args()

    dev = torch.device(args.device)
    emb_dim = int(CFG.emb_input_dim)
    bounds = block_bounds(emb_dim)
    print('[w0] device=%s  emb_dim=%d  blocks=%s' % (dev, emb_dim, bounds))

    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate,
                light_attention=T.LIGHT_ATTENTION, readout=T.READOUT).to(dev)
    # Checkpoints come in two shapes here: the wrapped dict load_checkpoint expects,
    # and a bare state_dict.  train.py handles both with a try/except; do the same,
    # but report which one was found so a silently-unloaded model is impossible.
    try:
        load_checkpoint(args.ckpt, model, device=str(dev))
        print('[w0] checkpoint format: wrapped (model_state_dict)')
    except KeyError:
        sd = torch.load(args.ckpt, map_location=str(dev), weights_only=False)
        target = model.module if hasattr(model, 'module') else model
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print('[w0] checkpoint format: bare state_dict '
              '(missing=%d, unexpected=%d)' % (len(missing), len(unexpected)))
        if missing or unexpected:
            print('[w0] FATAL: state_dict does not match the model. missing=%s unexpected=%s'
                  % (list(missing)[:5], list(unexpected)[:5]))
            sys.exit(2)
    model.eval()

    ds = T.AllProteinValidationDataset(tensor_root_dir=args.tensor_root,
                                       mutations_root_dir=args.mut_root,
                                       train=False)
    n_total = len(ds)
    n_use = n_total if args.limit <= 0 else min(args.limit, n_total)
    print('[w0] test proteins: %d (using %d)' % (n_total, n_use))

    rows = []
    skipped = []
    for i in range(n_use):
        try:
            d = ds[i]
        except Exception as exc:                      # noqa: BLE001
            skipped.append((i, 'load: %s' % exc))
            print('[w0] SKIP %d (load): %r' % (i, exc))
            continue
        try:
            coords = d['coords'].squeeze().to(dev)
            mask = d['masks'].squeeze().to(dev)
            one_hot_wt = d['one_hot'][0].squeeze().to(dev)
            emb_wt = d['prott5'][0].squeeze().to(dev)
            true_dg = float(np.asarray(d['delta_g'][0]).reshape(-1)[0])
            n_valid = int((mask > 0).sum().item())

            with torch.no_grad():
                folded = get_graph(coords, one_hot_wt, emb_wt, mask).unsqueeze(0)
                e_f = float(model(folded).reshape(-1)[0].item())

                energies = {}
                for cond in CONDITIONS:
                    g = unfolded_graph_for(cond, coords, one_hot_wt, emb_wt, mask, bounds)
                    energies[cond] = float(model(g.unsqueeze(0)).reshape(-1)[0].item())

            row = {'name': d['name'], 'length': n_valid, 'E_f': e_f,
                   'true_wt_dg': true_dg, 'n_variants': int(len(d['mutations']))}
            for cond in CONDITIONS:
                row['E_u_' + cond] = energies[cond]
                # dg_length_norm is 'none' in the reference configuration, so the
                # divisor is 1.0; recorded explicitly so this is checkable.
                row['pred_dg_' + cond] = energies[cond] - e_f
                row['wt_err_' + cond] = (energies[cond] - e_f) - true_dg
            rows.append(row)
            print('[w0] %2d/%d %-22s N=%4d  E_f=%9.3f  base=%9.3f  coil=%9.3f'
                  % (i + 1, n_use, row['name'], n_valid, e_f,
                     energies['base'], energies['coil']))
        except Exception as exc:                      # noqa: BLE001
            skipped.append((i, 'compute: %s' % exc))
            print('[w0] SKIP %d (compute): %r' % (i, exc))

    if len(rows) < args.min_proteins:
        print('[w0] FATAL: only %d proteins produced a result (need >= %d). '
              'A partial answer must not be read as a decision.'
              % (len(rows), args.min_proteins))
        sys.exit(2)

    df = pd.DataFrame(rows)
    lengths = df['length'].values

    stats = {}
    for cond in CONDITIONS:
        eu = df['E_u_' + cond].values
        stats[cond] = {
            'var_Eu': float(np.var(eu, ddof=1)),
            'mean_Eu': float(np.mean(eu)),
            'std_Eu': float(np.std(eu, ddof=1)),
            'corr_Eu_wt_err': pearson(eu, df['wt_err_' + cond].values),
            'corr_Eu_length': pearson(eu, lengths),
            'mean_abs_wt_err': float(np.mean(np.abs(df['wt_err_' + cond].values))),
        }
    base_var = stats['base']['var_Eu']
    for cond in CONDITIONS:
        stats[cond]['r_vs_base'] = (float(stats[cond]['var_Eu'] / base_var)
                                    if base_var > 0 else float('nan'))

    r = {c: stats[c]['r_vs_base'] for c in CONDITIONS}
    verdict, factor_d, side_arm = decide_d0(r)

    print('\n%-8s %12s %8s %14s %14s' % ('cond', 'var(E_u)', 'r/base',
                                         'corr(E_u,err)', 'corr(E_u,len)'))
    for cond in CONDITIONS:
        s = stats[cond]
        print('%-8s %12.4f %8.3f %14.4f %14.4f'
              % (cond, s['var_Eu'], s['r_vs_base'],
                 s['corr_Eu_wt_err'], s['corr_Eu_length']))
    print('\n[D0] %s' % verdict)
    print('[D0] factor_D := %s' % factor_d)
    if side_arm:
        print('[D0] side arm  := %s' % side_arm)

    out = {
        'schema': 'w0.v1',
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'checkpoint': os.path.abspath(args.ckpt),
        'device': str(dev),
        'n_proteins': int(len(rows)),
        'n_test_proteins_total': int(n_total),
        'skipped': skipped,
        'emb_dim': emb_dim,
        'block_bounds': bounds,
        'dg_length_norm': T.DG_LENGTH_NORM,
        'tensor_root': args.tensor_root,
        'mut_root': args.mut_root,
        'conditions': {
            'base': 'unfolded graph as-is, CFG.flory_unfolded=False',
            'noemb': 'ProtT5 block of the assembled unfolded graph zeroed',
            'noOH': 'one-hot block of the assembled unfolded graph zeroed',
            'coil': 'CFG.flory_unfolded=True (Flory random coil)',
        },
        'stats': stats,
        'r_vs_base': r,
        'd0': {'verdict': verdict, 'factor_D': factor_d, 'side_arm': side_arm},
        'per_protein': rows,
    }
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n[w0] wrote %s' % args.out)


def decide_d0(r):
    """AUTOPILOT D0.  Let r_x = var(E_u | x) / var(E_u | base).

    r_noemb < 0.5                      -> ProtT5 is the offset channel; factor D becomes
                                          --unfolded_emb {full,zero}, coil demoted
    r_coil  < 0.5 and r_noemb >= 0.5   -> geometry; factor D := --flory_unfolded
    r_noOH  < 0.5 and both others >= 0.5 -> composition; coil stays, add --dg_length_norm
    all >= 0.5                         -> variance is in learned weights; coil on the
                                          weakest evidence, log H4 for the write-up
    two below 0.5                      -> entangled; take the smaller r, log the other
    """
    below = [c for c in ('noemb', 'noOH', 'coil') if r[c] < 0.5]

    if len(below) >= 2:
        winner = min(below, key=lambda c: r[c])
        others = [c for c in below if c != winner]
        factor = ('--unfolded_emb {full,zero}' if winner == 'noemb'
                  else '--flory_unfolded')
        return ('Channels are entangled: %s all collapse var(E_u) below 0.5 '
                '(%s). Taking the smaller r.'
                % (', '.join(below), ', '.join('%s=%.3f' % (c, r[c]) for c in below)),
                factor, 'side arm: ' + ', '.join(others))

    if r['noemb'] < 0.5:
        return ('ProtT5 is the offset channel: r_noemb=%.3f < 0.5. The coil repairs D and '
                'Fb, which are not where the variance is.' % r['noemb'],
                '--unfolded_emb {full,zero}', 'coil demoted to a 2-cell side arm')

    if r['coil'] < 0.5:
        return ('Geometry is the offset channel: r_coil=%.3f < 0.5, r_noemb=%.3f >= 0.5.'
                % (r['coil'], r['noemb']),
                '--flory_unfolded', None)

    if r['noOH'] < 0.5:
        return ('Composition and the extensive sum: r_noOH=%.3f < 0.5, others >= 0.5.'
                % r['noOH'],
                '--flory_unfolded', '--dg_length_norm as a 2-cell side arm')

    return ('Nothing collapses var(E_u) (noemb=%.3f, noOH=%.3f, coil=%.3f). The variance '
            'is in the learned weights, not the inputs. Factor D proceeds on the weakest '
            'evidence and H4 is logged for the write-up.'
            % (r['noemb'], r['noOH'], r['coil']),
            '--flory_unfolded', 'log H4: the offset story needs re-examination')


if __name__ == '__main__':
    main()
