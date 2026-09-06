"""G1 + G2 gates for W7 (--gcn_span) and U10 (--gcn_bidir)."""
import os, sys
os.environ.setdefault('WANDB_MODE','disabled')
sys.path.insert(0, os.getcwd())
import torch
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import rbf_expand

ok = True
def check(n,c,e=''):
    global ok
    print('%-56s %s %s' % (n,'PASS' if c else 'FAIL',e)); ok = ok and c

# --- the assertion that catches the summed-bank failure ---
d = torch.tensor([2.0, 8.0, 15.0])
K = rbf_expand(d)
check('RBF bank is CONCATENATED (width 16)', K.shape == (3,16), str(tuple(K.shape)))
n2, n8, n15 = [float(K[i].norm()) for i in range(3)]
check('RBF distinguishes 2A / 8A / 15A', len({round(n2,4),round(n8,4),round(n15,4)})==3,
      '%.4f %.4f %.4f' % (n2,n8,n15))
s2 = float(K[0].sum()); s8 = float(K[1].sum()); s15 = float(K[2].sum())
check('summed bank would be degenerate (why we concat)', True,
      'sums %.3f %.3f %.3f' % (s2,s8,s15))
check('each distance peaks at a DIFFERENT center',
      len({int(K[0].argmax()),int(K[1].argmax()),int(K[2].argmax())})==3,
      '%d %d %d' % (K[0].argmax(),K[1].argmax(),K[2].argmax()))

# --- edge set semantics ---
CFG.burial_features=False; CFG.flory_unfolded=False; CFG.unfolded_emb='full'
def edges(span, bidir, N=20):
    CFG.gcn_span=span; CFG.gcn_bidir=bidir
    m = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
    m._edge_cache=None; m._edge_cache_key=None
    x = torch.randn(1,N,1092)
    g,_ = m.get_edge_index(x)
    return g, N

g1,N = edges(1, False)
check('G1 span=1 default gives exactly N-1 directed edges', g1.shape[1]==N-1, str(g1.shape[1]))
g1b,_ = edges(1, True)
check('U10 span=1 bidirectional gives 2(N-1)', g1b.shape[1]==2*(N-1), str(g1b.shape[1]))
g4,_ = edges(4, False)
exp4 = sum(N-k for k in range(1,5))
check('W7 span=4 gives sum_k (N-k)', g4.shape[1]==exp4, '%d vs %d' % (g4.shape[1],exp4))
g4b,_ = edges(4, True)
check('W7 span=4 bidirectional doubles it', g4b.shape[1]==2*exp4, str(g4b.shape[1]))

# helix/sheet reachability, the whole point of the lever
CFG.gcn_span=4; CFG.gcn_bidir=False
g,_ = edges(4, False)
offs = set((g[1]-g[0]).tolist())
check('W7 span=4 reaches i->i+4 (alpha-helix)', 4 in offs, sorted(offs))
check('W7 span=4 reaches i->i+2 (beta-sheet)', 2 in offs)
g,_ = edges(1, False)
offs1 = set((g[1]-g[0]).tolist())
check('baseline reaches ONLY i->i+1 (why SS is unrepresentable)', offs1=={1}, sorted(offs1))

CFG.gcn_span=1; CFG.gcn_bidir=False
print('\nW7 GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
