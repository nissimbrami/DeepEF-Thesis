# -*- coding: utf-8 -*-
"""Add --resume (default OFF, byte-identical when absent) to Megascale-fineTuning/train.py.

Idempotent: refuses to double-apply. Writes a .bak_resume next to the file.
"""
import io, os, sys, shutil

P = '/home/nissimb/DeepPEF/Megascale-fineTuning/train.py'
src = io.open(P, encoding='utf-8').read()

if 'RESUME_MARKER_V1' in src:
    print('ALREADY APPLIED'); sys.exit(0)

orig = src

# ---------------------------------------------------------------- 1. argparse
anchor = "_p.add_argument('--no_freeze', action='store_true',"
assert src.count(anchor) == 1, 'no_freeze anchor'
add = (
    "_p.add_argument('--resume', action='store_true', help=\"CRASH/PREEMPTION SAFETY "
    "(RESUME_MARKER_V1). Default OFF => byte-identical to the pre-resume script. When set, scan "
    "THIS run's own model dir for kf_{kf}_epoch_*.pt, take the HIGHEST completed epoch N, restore "
    "model (+optimizer+scheduler when the checkpoint carries them) and continue the loop at N+1 "
    "instead of 0. Without it a requeued job silently retrains from epoch 0.\")\n"
    "_p.add_argument('--resume_allow_partial_state', action='store_true', help=\"Permit --resume "
    "from an OLD bare-state_dict checkpoint (no optimizer/scheduler inside). Adam moments and the "
    "ReduceLROnPlateau state reset, so the run is NOT numerically equivalent to an uninterrupted "
    "one. Off => --resume REFUSES such a checkpoint rather than silently producing a different "
    "trajectory.\")\n"
)
src = src.replace(anchor, add + anchor, 1)

# ---------------------------------------------------------------- 2. globals + two-stage note
g_anchor = "UNFREEZE_EPOCHS = _a.unfreeze_epochs\n"
assert src.count(g_anchor) == 1, 'unfreeze_epochs anchor'
guard = g_anchor + '''RESUME = _a.resume
RESUME_ALLOW_PARTIAL = _a.resume_allow_partial_state
# --- RESUME_MARKER_V1: two-stage safety -------------------------------------
# The freeze/unfreeze schedule is NOT one loop with a stage switch inside it: it is TWO
# SEPARATE process launches. Stage 1 is the freeze phase (FREEZE_LAYERS=True, LR=1e-4,
# saves kf_{fold}_epoch_*.pt). Stage 2 is a second launch carrying --unfreeze_from <stage-1
# checkpoint>, which sets FREEZE_LAYERS=False, LR=UNFREEZE_LR and writes kf_full_epoch_*.pt.
# The stage is therefore fully determined by argv, and SLURM requeue re-runs the SAME argv,
# so a resumed job re-enters the stage it was killed in; it cannot restart stage 1.
# The residual hazard is stage 2 writing into the SAME MODEL_NAME dir as stage 1 (same
# --run_tag). The kf key is part of the filename -- stage 2 uses kf='full', stage 1 uses
# integer folds -- so the globs are disjoint by construction. We do not trust that: the
# resume scan matches the kf key EXACTLY, and the loaded checkpoint's freeze_layers flag is
# asserted against this process's FREEZE_LAYERS before any resume proceeds.
'''
src = src.replace(g_anchor, guard, 1)

# ---------------------------------------------------------------- 3. helpers before Trainer
h_anchor = "\nclass Trainer"
assert h_anchor in src, 'Trainer anchor'
helpers = '''
# --- RESUME_MARKER_V1 helpers ------------------------------------------------
import re as _re_resume


def _find_resume_ckpt(kf):
    """Highest COMPLETED epoch for THIS run dir and THIS kf key, or (None, None).

    Numeric sort, not lexicographic: 'kf_all_epoch_10.pt' must beat 'kf_all_epoch_9.pt'.
    The kf key is matched EXACTLY so stage-1 (integer fold) and stage-2 (kf='full')
    checkpoints can never be confused even when they share a MODEL_NAME dir.
    """
    import glob as _glob
    d = os.path.join(MODEL_PATH, MODEL_NAME)
    if not os.path.isdir(d):
        return None, None
    pat = _re_resume.compile(r'^kf_' + _re_resume.escape(str(kf)) + r'_epoch_(\\d+)\\.pt$')
    best, best_p = None, None
    for p in _glob.glob(os.path.join(d, 'kf_%s_epoch_*.pt' % kf)):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        e = int(m.group(1))
        # A zero-byte file is a checkpoint that was being written when the job died. Treat it
        # as NOT completed rather than resuming from a truncated tensor blob.
        try:
            if os.path.getsize(p) <= 0:
                print('[resume] skipping empty/truncated checkpoint %s' % p)
                continue
        except OSError:
            continue
        if best is None or e > best:
            best, best_p = e, p
    return best, best_p


def _load_ckpt_any(path, map_location):
    """Load either a NEW wrapper dict or an OLD bare state_dict. Returns (sd, extras).

    extras is None for the legacy bare-state_dict files already on disk (361 of them); a dict
    with optimizer/scheduler/epoch for wrapper checkpoints. Backward compatibility is required:
    the running factorial reads those legacy files.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        return obj['model_state_dict'], obj
    return obj, None

'''
src = src.replace(h_anchor, helpers + h_anchor, 1)

# ---------------------------------------------------------------- 4. save wrapper
save_old = ("                torch.save(self.model.state_dict(), "
            "os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt'))")
assert src.count(save_old) == 1, 'save anchor'
save_new = '''                # RESUME_MARKER_V1: save model + optimizer + scheduler + epoch so a resumed run
                # restores Adam's moments and the ReduceLROnPlateau state.
                #
                # READER COMPATIBILITY (this shape is not free-form). The factorial scores these
                # files with `evaluate.py --trained_model_path <kf_..._epoch_N.pt>`, which calls
                # train_utils.load_checkpoint() inside a bare `except:` that falls back to
                # `model.load_state_dict(torch.load(path))`. load_checkpoint() indexes
                # model_dict['model_state_dict'], ['epoch'], ['loss'] and ['valid_loss'] -- so a
                # wrapper MISSING 'loss'/'valid_loss' would raise KeyError, hit the fallback, and
                # the fallback would then hand a dict to load_state_dict and die. We therefore
                # emit ALL FIVE keys load_checkpoint() reads. Old bare-state_dict files keep
                # working through the same fallback path, untouched.
                _ck = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': float(running_loss),        # required by train_utils.load_checkpoint
                    'valid_loss': None,                 # required by train_utils.load_checkpoint
                    'kf': kf,
                    'model_name': MODEL_NAME,
                    'freeze_layers': FREEZE_LAYERS,
                    'lr': LR,
                    'format': 'RESUME_MARKER_V1',
                }
                _dst = os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt')
                # Atomic: write a temp name in the SAME dir then rename. A job killed mid-save
                # must not leave a half-written file that a later --resume would load.
                _tmp = _dst + '.tmp'
                torch.save(_ck, _tmp)
                os.replace(_tmp, _dst)'''
src = src.replace(save_old, save_new, 1)

# ---------------------------------------------------------------- 5. the loop
loop_old = """        running_loss = 0
        wandb_step = 0
        for epoch in range(epochs):"""
assert src.count(loop_old) == 1, 'loop anchor'
loop_new = '''        running_loss = 0
        wandb_step = 0
        # --- RESUME_MARKER_V1 ---------------------------------------------------
        # start_epoch stays 0 unless --resume is passed AND a checkpoint for THIS run dir and
        # THIS kf exists. With --resume absent this block does nothing and leaves start_epoch=0,
        # so range(start_epoch, epochs) == range(epochs) exactly as before.
        start_epoch = 0
        if RESUME:
            _re_epoch, _re_path = _find_resume_ckpt(kf)
            if _re_path is None:
                print('[resume] ENABLED but NO checkpoint found in %s for kf=%s -- '
                      'STARTING FRESH FROM EPOCH 0 (%d epochs to run)'
                      % (os.path.join(MODEL_PATH, MODEL_NAME), kf, epochs), flush=True)
            else:
                _sd, _extras = _load_ckpt_any(_re_path, self.device)
                if _extras is None and not RESUME_ALLOW_PARTIAL:
                    raise RuntimeError(
                        "--resume found a LEGACY bare-state_dict checkpoint with no optimizer/"
                        "scheduler state: %s (epoch %d). Resuming from it would reset Adam's "
                        "moment estimates and the ReduceLROnPlateau state, so the run would NOT "
                        "be numerically equivalent to an uninterrupted one and the resulting "
                        "model would be silently incomparable to the rest of the factorial. Pass "
                        "--resume_allow_partial_state to accept that explicitly, or move the "
                        "legacy checkpoints aside and let this run start fresh."
                        % (_re_path, _re_epoch))
                self.model.load_state_dict(_sd)
                _opt_ok = _sch_ok = False
                if _extras is not None:
                    # A checkpoint written in the OTHER stage must never be resumed into this one.
                    _ck_fl = _extras.get('freeze_layers')
                    if _ck_fl is not None and bool(_ck_fl) != bool(FREEZE_LAYERS):
                        raise RuntimeError(
                            '--resume STAGE MISMATCH: checkpoint %s was written with '
                            'freeze_layers=%s but this process has freeze_layers=%s. Resuming '
                            'would continue the wrong stage of the two-stage schedule. Refusing.'
                            % (_re_path, bool(_ck_fl), bool(FREEZE_LAYERS)))
                    if _extras.get('optimizer_state_dict') is not None:
                        self.optimizer.load_state_dict(_extras['optimizer_state_dict'])
                        _opt_ok = True
                    if _extras.get('scheduler_state_dict') is not None:
                        self.scheduler.load_state_dict(_extras['scheduler_state_dict'])
                        _sch_ok = True
                start_epoch = _re_epoch + 1
                print('[resume] RESUMED from %s\\n'
                      '[resume]   completed epoch = %d  ->  CONTINUING AT EPOCH %d of %d\\n'
                      '[resume]   optimizer state restored = %s   scheduler state restored = %s\\n'
                      '[resume]   lr now = %s   freeze_layers = %s   kf = %s'
                      % (_re_path, _re_epoch, start_epoch, epochs, _opt_ok, _sch_ok,
                         self.optimizer.param_groups[0]['lr'], FREEZE_LAYERS, kf), flush=True)
                if start_epoch >= epochs:
                    print('[resume] epoch %d is the LAST of %d; nothing left to train. '
                          'Running a final validation only.' % (_re_epoch, epochs), flush=True)
                    _pc, _vl = self.validate(_re_epoch, run)
                    return self.model, _pc
        for epoch in range(start_epoch, epochs):'''
src = src.replace(loop_old, loop_new, 1)

assert src != orig
shutil.copyfile(P, P + '.bak_resume')
io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('PATCH APPLIED')
