# -*- coding: utf-8 -*-
"""Fix the scheduler-state ORDERING bug in the resume checkpoint.

train.py's epoch end is, in this order:
    save checkpoint            <-- scheduler state here is PRE-step for this epoch
    pc_corr = self.validate(epoch)
    self.scheduler.step(pc_corr)

So a checkpoint written at epoch N carries the ReduceLROnPlateau state from BEFORE
epoch N's plateau step. Resuming at N+1 with that state replays epoch N+1 against a
stale scheduler (wrong num_bad_epochs / best / possibly wrong LR), which is exactly
the "not numerically equivalent" failure this work exists to prevent.

Fix: after scheduler.step(pc_corr), patch the just-written checkpoint's
'scheduler_state_dict' (and record the post-step LR) in place, so the file describes
the state at the true epoch boundary. The model/optimizer tensors are NOT re-saved and
NOT changed -- only the scheduler blob and lr field are updated, atomically.

Idempotent.
"""
import io, sys, shutil

P = '/home/nissimb/DeepPEF/Megascale-fineTuning/train.py'
src = io.open(P, encoding='utf-8').read()

if 'RESUME_SCHED_FIX_V1' in src:
    print('ALREADY APPLIED'); sys.exit(0)

old = """            pc_corr, val_loss = self.validate(epoch,run)
            self.model.train()
            # update the learning rate
            self.scheduler.step(pc_corr)
            current_lr = self.optimizer.param_groups[0]['lr']"""
assert src.count(old) == 1, 'epoch-end anchor'

new = """            pc_corr, val_loss = self.validate(epoch,run)
            self.model.train()
            # update the learning rate
            self.scheduler.step(pc_corr)
            current_lr = self.optimizer.param_groups[0]['lr']
            # RESUME_SCHED_FIX_V1: the checkpoint above was written BEFORE this
            # scheduler.step(), so its scheduler blob is the PRE-step state for this
            # epoch. A resume at epoch+1 would then run with a stale ReduceLROnPlateau
            # (wrong num_bad_epochs/best, possibly a pre-decay LR). Rewrite just the
            # scheduler blob + lr in the file we already wrote, so the checkpoint
            # describes the true epoch boundary. Model/optimizer tensors are untouched.
            if not DEBUG:
                try:
                    _fix = os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt')
                    _o = torch.load(_fix, map_location='cpu', weights_only=False)
                    if isinstance(_o, dict) and 'scheduler_state_dict' in _o:
                        _o['scheduler_state_dict'] = self.scheduler.state_dict()
                        _o['lr'] = current_lr
                        _o['val_pc_corr'] = float(pc_corr)
                        _o['valid_loss'] = float(val_loss)
                        _t = _fix + '.tmp'
                        torch.save(_o, _t)
                        os.replace(_t, _fix)
                except Exception as _e:
                    print(f'[resume] WARNING: could not update scheduler state in '
                          f'checkpoint for epoch {epoch}: {_e}', flush=True)"""

src = src.replace(old, new, 1)
shutil.copyfile(P, P + '.bak_sched')
io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('SCHED FIX APPLIED')
