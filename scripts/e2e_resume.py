# -*- coding: utf-8 -*-
"""End-to-end CPU proof of --resume, using the REAL patched code from train.py.

No GPU, no sbatch, no dataset. We lift the ACTUAL patched source of Trainer.train()'s
resume prologue and the epoch-end save block out of train.py and run them against a real
torch model + real Adam + real ReduceLROnPlateau, so what is exercised is the code that
will run on the cluster, not a paraphrase of it.

Three runs on identical seeds/data:
  A) uninterrupted 3 epochs
  B) 1 epoch, process "killed", then --resume for 2 more  (WITH optimizer+scheduler state)
  C) same as B but resuming from a LEGACY bare state_dict (optimizer/scheduler LOST)

Then compares final weights A vs B and A vs C.
"""
from __future__ import print_function
import io, os, re, sys, shutil, tempfile
import torch, torch.nn as nn, torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'Megascale-fineTuning', 'train.py')
src = io.open(TRAIN, encoding='utf-8').read()

# ---- lift the REAL patched fragments -------------------------------------------------
hs, he = src.index('def _find_resume_ckpt'), src.index('class Trainer')
HELPERS = src[hs:he]

PROLOGUE = src[src.index('        start_epoch = 0\n'):
               src.index('        for epoch in range(start_epoch, epochs):')]
SAVE = src[src.index('                _ck = {'):
           src.index('                os.replace(_tmp, _dst)') + len('                os.replace(_tmp, _dst)')]
SCHEDFIX = src[src.index('            if not DEBUG:\n                try:\n                    _fix'):
               src.index("checkpoint for epoch {epoch}: {_e}', flush=True)")
               + len("checkpoint for epoch {epoch}: {_e}', flush=True)")]
print('lifted resume prologue (%d chars), save block (%d chars), sched fixup (%d chars) '
      'from the real train.py' % (len(PROLOGUE), len(SAVE), len(SCHEDFIX)))

def _dedent(block):
    """Strip the common leading indentation from a lifted source block."""
    import textwrap
    return textwrap.dedent(block)


TMP = tempfile.mkdtemp(prefix='e2e_resume_')


def make_data(n=64, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = X @ w + 0.1 * torch.randn(n, 1, generator=g)
    return X, y


class Net(nn.Module):
    def __init__(self, d=8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Harness(object):
    """Stands in for Trainer: same attribute names the lifted code touches."""

    def __init__(self, model_dir, kf='all'):
        torch.manual_seed(1234)
        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=0)
        self.device = 'cpu'
        self.criterion = nn.L1Loss()
        self.kf = kf
        self.model_dir = model_dir

    def validate(self, epoch, run=None):
        return 0.5, 0.5


def run(tag, epochs, model_dir, resume, allow_partial=False, stop_after=None,
        kf='all', quiet=False):
    """Runs the REAL lifted prologue + REAL lifted save block around a real train loop."""
    os.makedirs(os.path.join(TMP, model_dir), exist_ok=True)
    ns = {
        'os': os, 'torch': torch, 're': re, '_re_resume': re,
        'MODEL_PATH': TMP, 'MODEL_NAME': model_dir,
        'RESUME': resume, 'RESUME_ALLOW_PARTIAL': allow_partial,
        'FREEZE_LAYERS': True, 'LR': 1e-2, 'DEBUG': False,
        'print': (lambda *a, **k: None) if quiet else print,
    }
    exec(compile(HELPERS, 'helpers', 'exec'), ns)

    h = Harness(model_dir, kf)
    X, y = make_data()
    ns.update({'self': h, 'kf': kf, 'epochs': epochs, 'run': None, 'running_loss': 0.0})

    # ---- the REAL resume prologue -----------------------------------------------
    # It contains a `return` (the "already finished, validate only" branch), so it is
    # wrapped in a function exactly as it sits inside Trainer.train().
    # The prologue's own early-return yields (self.model, pc); ours yields ('START', n).
    wrapper = ('def _prologue(self, kf, epochs, run):\n'
               + PROLOGUE
               + "        return 'START', start_epoch\n")
    exec(compile(wrapper, 'prologue', 'exec'), ns)
    _res = ns['_prologue'](h, kf, epochs, None)
    if _res[0] != 'START':           # the "nothing left to train" branch fired
        return h, epochs - 1
    start_epoch = _res[1]
    ns['start_epoch'] = start_epoch

    # ---- the training loop -------------------------------------------------------
    torch.manual_seed(4321 + start_epoch * 0)  # data order fixed; seed not epoch-dependent
    for epoch in range(start_epoch, epochs):
        h.model.train()
        for i in range(0, X.size(0), 16):
            h.optimizer.zero_grad()
            loss = h.criterion(h.model(X[i:i + 16]), y[i:i + 16])
            loss.backward()
            h.optimizer.step()
        # ---- the REAL save block --------------------------------------------------
        ns.update({'epoch': epoch, 'running_loss': float(loss.item())})
        exec(compile(_dedent(SAVE), 'save', 'exec'), ns)
        pc, vl = h.validate(epoch)
        h.scheduler.step(pc)
        # ---- the REAL scheduler-state fixup (RESUME_SCHED_FIX_V1) -----------------
        ns.update({'current_lr': h.optimizer.param_groups[0]['lr'],
                   'pc_corr': pc, 'val_loss': vl})
        exec(compile(_dedent(SCHEDFIX), 'schedfix', 'exec'), ns)
        if stop_after is not None and epoch >= stop_after:
            if not quiet:
                print('  [%s] *** SIMULATED KILL after epoch %d ***' % (tag, epoch))
            return h, epoch
    return h, epochs - 1


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


print('\n' + '=' * 78)
print('RUN A: uninterrupted 3 epochs')
print('=' * 78)
A, _ = run('A', 3, 'runA', resume=False)

print('\n' + '=' * 78)
print('RUN B: 1 epoch -> KILL -> --resume for 2 more (optimizer+scheduler RESTORED)')
print('=' * 78)
run('B', 3, 'runB', resume=False, stop_after=0)
print('  --- restarting the process with --resume ---')
B, _ = run('B', 3, 'runB', resume=True)

print('\n' + '=' * 78)
print('RUN C: same, but resuming from a LEGACY bare state_dict (opt/sched LOST)')
print('=' * 78)
run('C', 3, 'runC', resume=False, stop_after=0)
# downgrade the epoch-0 checkpoint to the old bare format, as the 361 files on disk are
p0 = os.path.join(TMP, 'runC', 'kf_all_epoch_0.pt')
torch.save(torch.load(p0, map_location='cpu', weights_only=False)['model_state_dict'], p0)
print('  (rewrote kf_all_epoch_0.pt as a legacy bare state_dict)')
try:
    run('C', 3, 'runC', resume=True)
    print('  !! resumed WITHOUT --resume_allow_partial_state -- guard did not fire')
except RuntimeError as e:
    print('  GUARD FIRED as designed:\n    %s' % str(e)[:220].replace('\n', '\n    '))
C, _ = run('C', 3, 'runC', resume=True, allow_partial=True)

# ---------------------------------------------------------------- comparison
a, b, c = flat(A.model), flat(B.model), flat(C.model)
print('\n' + '=' * 78)
print('FINAL WEIGHT COMPARISON  (%d parameters)' % a.numel())
print('=' * 78)


def cmp(name, x, yv):
    same = bool(torch.equal(x, yv))
    md = float((x - yv).abs().max())
    rel = md / float(x.abs().max())
    print('%-46s bit-identical=%-5s  max|d|=%.3e  rel=%.3e' % (name, same, md, rel))
    return same, md


sB, dB = cmp('A (uninterrupted)  vs B (resume, full state)', a, b)
sC, dC = cmp('A (uninterrupted)  vs C (resume, LEGACY bare)', a, c)

print('\nVERDICT')
print('  B: %s' % ('BIT-IDENTICAL to the uninterrupted run' if sB else
                   'NOT bit-identical (max|d|=%.3e)' % dB))
print('  C: %s' % ('bit-identical (unexpected)' if sC else
                   'DIVERGES (max|d|=%.3e) -- optimizer/scheduler state was lost' % dC))
print('\n  Ratio of C-divergence to B-divergence: %s'
      % ('inf (B is exact)' if dB == 0 else '%.1fx' % (dC / dB)))
shutil.rmtree(TMP, ignore_errors=True)
