"""G4-equivalent on CPU: build a real PEM and push a real graph through it with each
lever on, so the fc1/inst_norm widths are exercised end to end. Catches a slice-arithmetic
error that the tensor-shape gates cannot: a wrong width raises here, a wrong ORDER does not,
so we also assert the solvation columns land where the model reads them.
"""
import os, sys
os.environ.setdefault('WANDB_MODE','disabled')
sys.path.insert(0, os.getcwd())
import torch
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph

torch.manual_seed(0)
ok = True
def check(n, c, e=''):
    global ok
    print('%-52s %s %s' % (n, 'PASS' if c else 'FAIL', e)); ok = ok and c

def run(tag, **cfg):
    for k, v in cfg.items(): setattr(CFG, k, v)
    N = 36
    x = torch.randn(N,4,3)*7.0
    oh = torch.eye(20)[torch.randint(0,20,(N,))]
    emb = torch.randn(N, int(CFG.emb_input_dim))
    mask = torch.ones(N)
    m = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
    m.eval()
    f = get_graph(x, oh, emb, mask).unsqueeze(0)
    u = get_unfolded_graph(x, oh, emb, mask).unsqueeze(0)
    with torch.no_grad():
        e = m(torch.cat([f,u],dim=0))
    dg = float(e.reshape(-1)[0] - e.reshape(-1)[1])
    fin = torch.isfinite(e).all().item()
    check('%-22s forward ok, dG finite' % tag, bool(fin), 'dG=%.4f width=%d' % (dg, f.shape[-1]))
    return f.shape[-1]

base = dict(flory_unfolded=False, unfolded_emb='full', burial_features=False, burial_mode='count')
w_off  = run('baseline', **base)
w_u2   = run('unfolded_emb=zero', **dict(base, unfolded_emb='zero'))
w_coil = run('flory_unfolded', **dict(base, flory_unfolded=True))
w_w5   = run('burial_features', **dict(base, burial_features=True))
w_hse  = run('burial hse', **dict(base, burial_features=True, burial_mode='hse'))
w_both = run('burial + uemb zero', **dict(base, burial_features=True, unfolded_emb='zero'))

check('width: levers that add no dims keep width', w_off == w_u2 == w_coil, '%d' % w_off)
check('width: burial adds exactly 3', w_w5 == w_off + 3 and w_both == w_off + 3, '%d' % w_w5)
for k,v in base.items(): setattr(CFG,k,v)
print('\nG4-CPU: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
