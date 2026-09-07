# -*- coding: utf-8 -*-
"""gate_resume.py -- guards for the --resume crash/preemption-safety feature in
Megascale-fineTuning/train.py.

Asserts, without a GPU and without submitting anything:
  G1  --resume OFF leaves the epoch loop byte-identical to the pre-resume script
      (the ONLY difference in the loop header is `range(start_epoch, epochs)` with
      start_epoch provably 0 when RESUME is False).
  G2  the resume scan picks the HIGHEST epoch, NUMERICALLY (epoch 10 beats epoch 9).
  G3  a fresh/absent dir yields no checkpoint => the run starts at epoch 0.
  G4  OLD bare-state_dict checkpoints (361 on disk) still load through _load_ckpt_any.
  G5  NEW wrapper checkpoints carry model+optimizer+scheduler+epoch AND every key
      train_utils.load_checkpoint() indexes, so evaluate.py keeps reading them.
  G6  the epoch printed in the [resume] log is the epoch actually resumed into.
  G7  the kf key is matched exactly, so stage-1 (integer fold) and stage-2 (kf='full')
      checkpoints are never confused inside one MODEL_NAME dir.
  G8  a zero-byte (mid-write) checkpoint is skipped rather than resumed from.

Run:  python scripts/gate_resume.py
"""
from __future__ import print_function
import io, os, re, sys, shutil, tempfile, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'Megascale-fineTuning', 'train.py')
BAK = TRAIN + '.bak_resume'

FAIL = []
PASS = []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('%-6s %s%s' % ('[ OK ]' if cond else '[FAIL]', name, ('  -- ' + detail) if detail else ''))


src = io.open(TRAIN, encoding='utf-8').read()

# ---------------------------------------------------------------- G1: OFF is inert
check('G1a --resume flag exists, default OFF (store_true)',
      "_p.add_argument('--resume', action='store_true'" in src)

check('G1b loop is range(start_epoch, epochs)',
      'for epoch in range(start_epoch, epochs):' in src and
      src.count('for epoch in range(epochs):') == 0)

# start_epoch must be initialised to 0 BEFORE the `if RESUME:` block, and every write to
# start_epoch must be inside that block. That is what makes OFF inert.
m = re.search(r'\n(\s*)start_epoch = 0\n(\s*)if RESUME:\n', src)
check('G1c start_epoch = 0 immediately precedes `if RESUME:`', m is not None)
assigns = re.findall(r'^\s*start_epoch\s*=\s*(.+)$', src, re.M)
check('G1d start_epoch assigned exactly twice (0, then _re_epoch+1)',
      assigns == ['0', '_re_epoch + 1'], repr(assigns))

# The reassignment must sit inside the `if RESUME:` block.
blk = src[src.index('        if RESUME:'):src.index('        for epoch in range(start_epoch, epochs):')]
check('G1e the only start_epoch reassignment is inside `if RESUME:`',
      'start_epoch = _re_epoch + 1' in blk)

if os.path.exists(BAK):
    old = io.open(BAK, encoding='utf-8').read()
    check('G1f backup of the pre-resume script exists', True, BAK)
    check('G1g pre-resume script had NO resume support',
          'RESUME_MARKER_V1' not in old and old.count('--resume') == 0)
    # Everything outside the injected blocks must be untouched: strip the added regions
    # from the new file and compare to the original.
    stripped = src
    for pat in [
        re.compile(r"_p\.add_argument\('--resume'.*?\n_p\.add_argument\('--resume_allow_partial_state'.*?\n", re.S),
        re.compile(r'RESUME = _a\.resume\n.*?asserted against this process.*?\n', re.S),
        re.compile(r'\n# --- RESUME_MARKER_V1 helpers.*?return obj, None\n\n', re.S),
        re.compile(r'        # --- RESUME_MARKER_V1 ---.*?\n(?=        for epoch in range\(start_epoch, epochs\):)', re.S),
        re.compile(r'\n            # RESUME_SCHED_FIX_V1:.*?flush=True\)(?=\n)', re.S),
    ]:
        stripped = pat.sub('', stripped, count=1)
    stripped = stripped.replace('for epoch in range(start_epoch, epochs):',
                                'for epoch in range(epochs):')
    # the save block: replace the new wrapper save with the old one-liner
    stripped = re.sub(
        r'                # RESUME_MARKER_V1: save model.*?os\.replace\(_tmp, _dst\)',
        "                torch.save(self.model.state_dict(), "
        "os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt'))",
        stripped, count=1, flags=re.S)
    check('G1h stripping the injected blocks reproduces the ORIGINAL file exactly',
          stripped == old,
          'differs' if stripped != old else '')
else:
    check('G1f backup of the pre-resume script exists', False, BAK + ' missing')

# ---------------------------------------------------------------- runtime probes
# Exercise the real helpers without importing train.py (which needs torch_geometric,
# a GPU default device, and the dataset). We re-exec the helper source with a stub
# module namespace -- the functions only touch os/glob/re/torch.
import torch

ns = {'os': os, 'torch': torch, 'print': print}
hs = src.index('def _find_resume_ckpt')
he = src.index('class Trainer')
helper_src = src[hs:he]
tmp = tempfile.mkdtemp(prefix='gate_resume_')
ns['MODEL_PATH'] = tmp
ns['MODEL_NAME'] = 'run'
ns['_re_resume'] = re
exec(compile(helper_src, 'helpers', 'exec'), ns)
_find = ns['_find_resume_ckpt']
_load = ns['_load_ckpt_any']

d = os.path.join(tmp, 'run')

# G3: fresh dir -> nothing
check('G3a absent dir => (None, None) => start at epoch 0', _find('all') == (None, None))
os.makedirs(d)
check('G3b empty dir  => (None, None) => start at epoch 0', _find('all') == (None, None))

# G2: numeric, not lexicographic
for e in [0, 1, 2, 9, 10, 11]:
    torch.save({'model_state_dict': {'w': torch.zeros(1)}, 'optimizer_state_dict': {},
                'scheduler_state_dict': {}, 'epoch': e, 'loss': 0.0, 'valid_loss': None,
                'freeze_layers': True, 'format': 'RESUME_MARKER_V1'},
               os.path.join(d, 'kf_all_epoch_%d.pt' % e))
best, bp = _find('all')
check('G2 highest epoch chosen NUMERICALLY (11 > 9, not "9" > "11")',
      best == 11 and bp.endswith('kf_all_epoch_11.pt'), 'got epoch=%s' % best)

# G7: kf key matched exactly; stage-1 folds must not be seen by a kf='full' scan
torch.save({'model_state_dict': {}, 'epoch': 99, 'loss': 0.0, 'valid_loss': None,
            'freeze_layers': True}, os.path.join(d, 'kf_0_epoch_99.pt'))
torch.save({'model_state_dict': {}, 'epoch': 98, 'loss': 0.0, 'valid_loss': None,
            'freeze_layers': False}, os.path.join(d, 'kf_full_epoch_98.pt'))
check('G7a kf="all" scan ignores kf_0_* and kf_full_*', _find('all')[0] == 11)
check('G7b kf=0 scan sees only its own fold', _find(0)[0] == 99)
check('G7c kf="full" scan sees only the stage-2 files', _find('full')[0] == 98)

# G8: zero-byte mid-write checkpoint is skipped
io.open(os.path.join(d, 'kf_all_epoch_12.pt'), 'wb').close()
best2, _ = _find('all')
check('G8 zero-byte (mid-write) checkpoint skipped, falls back to 11', best2 == 11,
      'got %s' % best2)
os.remove(os.path.join(d, 'kf_all_epoch_12.pt'))

# G4: OLD bare state_dict still loads
legacy = os.path.join(d, 'legacy.pt')
torch.save({'fc1.weight': torch.ones(2, 2), 'fc1.bias': torch.zeros(2)}, legacy)
sd, extras = _load(legacy, 'cpu')
check('G4a legacy bare state_dict loads (extras is None)',
      extras is None and 'fc1.weight' in sd)
check('G4b legacy tensors survive intact', bool(torch.equal(sd['fc1.weight'], torch.ones(2, 2))))

# G5: NEW wrapper round-trips and carries every key evaluate.py's loader indexes
wrap = os.path.join(d, 'kf_all_epoch_11.pt')
obj = torch.load(wrap, map_location='cpu', weights_only=False)
for k in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch'):
    check('G5a wrapper carries %r' % k, k in obj)
# train_utils.load_checkpoint indexes model_state_dict/epoch/loss/valid_loss -- all four
# must exist or evaluate.py raises KeyError and its bare `except:` fallback then feeds a
# dict to load_state_dict and dies.
for k in ('model_state_dict', 'epoch', 'loss', 'valid_loss'):
    check('G5b load_checkpoint() key %r present (evaluate.py compat)' % k, k in obj)
sd2, ex2 = _load(wrap, 'cpu')
check('G5c wrapper unwraps to the model state_dict', ex2 is not None and 'w' in sd2)

# the saver in train.py must actually emit those keys
save_blk = src[src.index("_ck = {"):src.index("os.replace(_tmp, _dst)")]
for k in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
          'epoch', 'loss', 'valid_loss'):
    check('G5d saver emits %r' % k, ("'%s'" % k) in save_blk)
check('G5e save is atomic (tmp + os.replace)',
      "_tmp = _dst + '.tmp'" in src and 'os.replace(_tmp, _dst)' in src)

# ---------------------------------------------------------------- G6: log matches resume
# The printed "CONTINUING AT EPOCH %d" must be start_epoch, and start_epoch == best+1.
check('G6a log prints the epoch it resumes INTO',
      'CONTINUING AT EPOCH %d of %d' in src and
      '% (_re_path, _re_epoch, start_epoch, epochs' in src)
check('G6b start_epoch == highest_completed + 1', 'start_epoch = _re_epoch + 1' in src)
check('G6c fresh start is logged loudly too', 'STARTING FRESH FROM EPOCH 0' in src)

# ---------------------------------------------------------------- optimizer/scheduler guard
check('G9a legacy checkpoint REFUSED under --resume unless explicitly allowed',
      '_extras is None and not RESUME_ALLOW_PARTIAL' in src and
      'RuntimeError' in blk)
check('G9b two-stage mismatch REFUSED (freeze_layers asserted)',
      'STAGE MISMATCH' in src and "_extras.get('freeze_layers')" in src)

# --------------------------------------------------- G10: scheduler-state ORDERING
# train.py saves the checkpoint BEFORE self.scheduler.step(pc_corr), so the blob written
# in the save block is the PRE-step ReduceLROnPlateau state for that epoch. Resuming with
# it replays the next epoch against a stale scheduler. The fixup after scheduler.step()
# rewrites just that blob. Without this, run B is NOT bit-identical (measured 3.5e-2).
check('G10a scheduler-state fixup present after scheduler.step()',
      'RESUME_SCHED_FIX_V1' in src)
_i_save = src.index('os.replace(_tmp, _dst)')
_i_step = src.index('self.scheduler.step(pc_corr)')
_i_fix = src.index('RESUME_SCHED_FIX_V1')
check('G10b fixup runs AFTER scheduler.step(), which runs AFTER the save',
      _i_save < _i_step < _i_fix)
check('G10c fixup rewrites scheduler_state_dict only (model/optimizer untouched)',
      "_o['scheduler_state_dict'] = self.scheduler.state_dict()" in src and
      "_o['model_state_dict'] =" not in src and
      "_o['optimizer_state_dict'] =" not in src)
check('G10d fixup rewrite is atomic and failure-tolerant',
      'os.replace(_t, _fix)' in src and 'could not update scheduler state' in src)

shutil.rmtree(tmp, ignore_errors=True)

print('\n%d passed, %d failed' % (len(PASS), len(FAIL)))
if FAIL:
    print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1)
print('GATE_RESUME OK')
