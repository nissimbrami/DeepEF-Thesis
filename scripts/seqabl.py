"""Ofir Ezrielev sequence-ablation analogue (thesis 3.1.1 p.24 permutation, 3.1.2 p.25 collapse)
applied to a FROZEN DeepEF checkpoint. Read-only: patches get_graph in memory, edits nothing.

Ofir permuted the AA order of the sequence while holding composition fixed. Our analogue holds
the 3D coords + mask fixed (they define the fold) and permutes the RESIDUE AXIS of the identity
channels (one_hot + prott5). Composition is preserved exactly; the sequence<->structure
correspondence is destroyed. If ddG survives this, the model is reading composition, not
structure-in-context.

Modes:
  none      identity (must reproduce the shipped CSV bit-for-bit -- this is the gate)
  perm      random permutation of the residue axis of one_hot AND prott5 (same perm for both)
  perm_emb  permute prott5 only (one_hot kept aligned to coords)
  perm_oh   permute one_hot only (prott5 kept aligned)
  collapse  Ofir 3.1.2: replace prott5 per-residue vector with the sequence MEAN (broadcast)
  collapse_oh  replace one_hot per-residue with the mean composition vector (broadcast)

The permutation is drawn ONCE per protein per seed and reused for every mutant of that protein,
so mutant and WT see the SAME scramble -- otherwise ddG would just be permutation noise.
"""
import os, sys, argparse
sys.path.append('./')
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
a, rest = ap.parse_known_args()

# evaluate.py parses argv itself -> hand it only what it understands
sys.argv = ['evaluate.py', '--trained_model_path', a.ckpt, '--model_name', a.out]

import train_utils
_G, _U = train_utils.get_graph, train_utils.get_unfolded_graph
STATE = {'perm': None, 'n': None}

def _mk(n, dev):
    """One permutation per protein (keyed by N), fixed across all mutants of that protein."""
    if STATE['n'] != n:
        g = np.random.RandomState(a.seed + n)      # deterministic, N-keyed
        STATE['perm'] = torch.as_tensor(g.permutation(n), device=dev, dtype=torch.long)
        STATE['n'] = n
    return STATE['perm']

def _mut(one_hot, emb):
    m = a.mode
    if m == 'none':
        return one_hot, emb
    n = one_hot.shape[0]
    if m in ('perm', 'perm_emb', 'perm_oh'):
        p = _mk(n, one_hot.device)
        if m == 'perm':      return one_hot[p], emb[p]
        if m == 'perm_emb':  return one_hot,    emb[p]
        if m == 'perm_oh':   return one_hot[p], emb
    if m == 'collapse':
        return one_hot, emb.mean(dim=0, keepdim=True).expand_as(emb).contiguous()
    if m == 'collapse_oh':
        return one_hot.mean(dim=0, keepdim=True).expand_as(one_hot).contiguous(), emb
    raise SystemExit('bad mode ' + m)

CALLS = {'n': 0}
def gp(x, one_hot, emb, mask, **kw):
    CALLS['n'] += 1
    oh, em = _mut(one_hot, emb)
    return _G(x, oh, em, mask, **kw)
def up(x, one_hot, emb, mask, **kw):
    oh, em = _mut(one_hot, emb)
    return _U(x, oh, em, mask, **kw)

train_utils.get_graph, train_utils.get_unfolded_graph = gp, up

import importlib
ev = importlib.import_module('Megascale-fineTuning.evaluate') if False else None
# evaluate.py lives in a dir with a dash -> load by path
import importlib.util
spec = importlib.util.spec_from_file_location('ev', 'Megascale-fineTuning/evaluate.py')
ev = importlib.util.module_from_spec(spec)
sys.modules['ev'] = ev
spec.loader.exec_module(ev)
# evaluate.py binds get_graph at import time -> rebind in ITS namespace too
ev.get_graph, ev.get_unfolded_graph = gp, up

ev.tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
ev.mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
ev.CFG.dropout_rate = ev.DROP_OUT
ev.run_validation_metrics()
print('PATCH_CALLS', CALLS['n'], 'mode', a.mode, 'seed', a.seed)
assert CALLS['n'] > 0, 'get_graph patch was NEVER CALLED -- ablation did not happen'
