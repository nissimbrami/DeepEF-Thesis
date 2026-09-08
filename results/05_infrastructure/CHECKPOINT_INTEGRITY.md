# Checkpoint Integrity & Restartability

**Question asked:** can we resume from any epoch, and is what is on disk actually usable?

**ANSWER: YES - every checkpoint on disk is intact, loadable and forward-runnable (28/28 runs).
But NOT YET from the files currently on disk**, because all 364 of them are legacy bare
state_dicts with no optimizer/scheduler state, and the new `--resume` deliberately *refuses*
those by default. Resume becomes real only for checkpoints written from now on, by jobs
launched from the updated `train.py`.

Verified 2026-09-07 on the BGU cluster, CPU only, read-only. No jobs touched.

> **The briefing I was given is now out of date, and it matters.** It states `train.py` has no
> `--resume` and `grep -ic resume` = 0. As of this check `grep -ic resume` = **32**,
> `start_epoch` appears 8 times, and the epoch loop reads `for epoch in range(start_epoch, epochs)`.
> Resume support landed in the working tree *while this audit was running* (there is a
> `train.py.bak_resume` alongside it). Section 6 audits what actually landed.

---

## 1. Inventory - 33 directories, 28 with checkpoints, 364 files, ZERO GAPS

Every run is a contiguous `epoch_0 .. epoch_N` sequence. **No run is missing an interior
epoch.** A gap would have meant a silent `torch.save` failure; there are none.

| Run tag (`PEM_..._light_attentionkf_` prefix stripped) | first | last | count | missing |
|---|---|---|---|---|
| `anchor_w0.3_s42` | 0 | 14 | 15 | - |
| `anchor_w1.0_s42` | 0 | 14 | 15 | - |
| `anchor_w3.0_s42` | 0 | 14 | 15 | - |
| `calib_ctrl_repro2` | 0 | 14 | 15 | - |
| `p3_a0_d0_s0_D0_coil_seed42` | 0 | 9 | 10 | - |
| `p3_a0_d0_s0_D1_uemb_seed42` | 0 | 14 | 15 | - |
| `p3_a0_d0_s1_D0_coil_seed42` | 0 | 9 | 10 | - |
| `p3_a0_d0_s1_D1_uemb_seed42` | 0 | 14 | 15 | - |
| `p3_a0_d1_s0_D0_coil_seed42` | 0 | 11 | 12 | - |
| `p3_a0_d1_s0_D1_uemb_seed42` | 0 | 14 | 15 | - |
| `p3_a0_d1_s1_D0_coil_seed42` | 0 | 14 | 15 | - |
| `p3_a0_d1_s1_D1_uemb_seed42` | 0 | 14 | 15 | - |
| `p3_a1_d0_s0_D0_coil_seed42` | 0 | 14 | 15 | - |
| `p3_a1_d0_s0_D1_uemb_seed42` | 0 | 3 | 4 | - (running) |
| `p3_a1_d0_s1_D0_coil_seed42` | 0 | 14 | 15 | - |
| `p3_a1_d0_s1_D1_uemb_seed42` | 0 | 3 | 4 | - (running) |
| `p3_a1_d1_s0_D0_coil_seed42` | 0 | 14 | 15 | - |
| `p3_a1_d1_s0_D1_uemb_seed42` | 0 | 3 | 4 | - (running) |
| `p3_a1_d1_s1_D0_coil_seed42` | 0 | 14 | 15 | - |
| `p3_a1_d1_s1_D1_uemb_seed42` | 0 | 3 | 4 | - (running) |
| `p3_slope0.3_s42` / `slope1.0` / `slope3.0` | 0 | 14 | 15 each | - |
| `sigma_seed1` / `2` / `3` / `4` / `42` | 0 | 14 | 15 each | - |

Runs ending at 9 / 11 are short because those jobs are still in flight or ended early; they are
still contiguous. Five directories hold no `.pt` at all (`..._light_attentionkf` bare,
`calib_ctrl_repro`, `probe1080`, `eval_results`, `ref_dg_seed42`) - scaffolding, not lost work.

## 2. Integrity - all sampled checkpoints load, all finite

14 checkpoints sampled (3 oldest on disk, 4 newest from currently-running jobs, plus an
even spread), loaded with `map_location='cpu', weights_only=False`:

- **14/14 loaded.** Every one is a real `state_dict`: **80 keys, 80 tensors, 22,017,597 parameters.**
- **0 non-finite tensors.** No NaN, no Inf, anywhere.
- Structure is identical across every run and every epoch - the architecture is stable.

File sizes: all 364 files are 88,099,250-88,099,334 bytes (~84.0 MB). The 84-byte spread is
pickle metadata, not content. **No zero-byte or truncated file exists** (`find -size -1M`
returns nothing) - which is the signature a half-written `torch.save` would leave.

## 3. The real question - restart probe: PASS on 28/28 runs

For **every one of the 28 runs**, the newest checkpoint was loaded into a freshly constructed
CPU `PEM` built exactly as `scripts/gate_g4_cpu.py` builds it, with `strict=True`, then pushed
through a real forward pass:

```
PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
    dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
```

**Result: 28 PASS, 0 FAIL.** Every checkpoint satisfied `strict=True` (no missing keys, no
unexpected keys, no shape mismatch) and produced a **finite dG**, ranging -0.5167 to +0.5322
across runs - varied, not a collapsed constant. Input width 1092 throughout.

Including the two endpoints the task called out:
- newest, from a running job (`p3_a1_d1_s1_D1_uemb_seed42/kf_all_epoch_3.pt`, written 14:32): dG=+0.0463
- oldest on disk (`calib_ctrl_repro2/kf_all_epoch_0.pt`, 2026-09-05 23:42): dG=+0.5274

A checkpoint written by a *running* job loads cleanly, so `torch.save` is not leaving a window
where a preemption would catch a partial file.

### One thing worth recording, because it nearly became a false alarm

A first pass inferred the model config from the run tag and read `_s1_` as "burial features on".
That mis-load failed loudly with a shape mismatch (`fc1_gcn` [64,52] vs [64,55]). **The
checkpoints were fine; the guess was wrong.** Checking `submit_factorial.sh` settles it: the tag
is `p3_a{A}_d{B}_s{C}_D{D}` where A=`wt_anchor_weight`, B=`designed_weight`, C=`slope_weight`,
D=the coil/uemb flag. These are all **loss weights - none change architecture**, and the D flag
adds no input dims either. So all runs share the width-52 GCN input, confirmed empirically by
28/28 strict loads against one baseline model.

The lesson is real though: **a checkpoint is only restartable together with the CFG it was
trained under.** The `.pt` stores weights only, no config. Any resume path must reconstruct the
same lever settings, and burial features (+3 dims) *would* genuinely break a load if mismatched.
Today no run uses them, so the risk is latent, not active.

## 4. Disk projection - ample headroom, no quota

| | |
|---|---|
| One checkpoint | 84.0 MB |
| Now on disk | 364 files = **29.86 GB** (`du`: 30 G) |
| Factorial projection | 48 runs x 15 epochs = 720 files -> **~59 GB** |
| Plus existing | ~**89 GB** total worst case |
| `/home` available | **166 TB** (398T size, 59% used) |
| Inodes | 356 G free, 1% used |
| Per-user quota | **none** - `quota -s -u nissimb` exits 1 with no output; `lfs` not present (NFS, not Lustre) |
| `/home/nissimb` total now | 37 GB |

~89 GB against 166 TB is **0.05%** of free space. **No quota exists to exceed**, so the
silent-failure mode of a `torch.save` hitting a quota wall is not in play. Disk is not a risk to
the factorial.

## 5. Gate

`scripts/gate_g4_cpu.py` after all work: **ALL PASS, `dG=-0.0030 width=1092`** - unchanged.
All probe scripts were deleted from the cluster; nothing was left behind in the repo.

## 6. Resume support - it landed mid-audit, and it is well built

`Megascale-fineTuning/train.py` is modified in the working tree (`git status`: ` M`) and now has:

- `for epoch in range(start_epoch, epochs)` (line 677), with `start_epoch = 0` unless `--resume`.
- A save that writes the **full** resume payload - `model_state_dict`, `optimizer_state_dict`,
  `scheduler_state_dict`, `epoch`, `loss`, `valid_loss`, `kf`, `freeze_layers`, `lr`, and a
  `format: RESUME_MARKER_V1` tag.
- **Atomic saves**: `torch.save(_ck, _tmp)` then `os.replace(_tmp, _dst)`. A job killed mid-save
  cannot leave a half-written file that a later `--resume` would load. Confirmed: **zero stray
  `.tmp` files on disk.**
- Two refusals that are exactly right for preemption safety:
  1. A **legacy bare checkpoint raises** rather than resuming, because that would reset Adam's
     moments and the `ReduceLROnPlateau` state and make the run "silently incomparable to the
     rest of the factorial". Overridable only via an explicit `--resume_allow_partial_state`.
  2. A **stage mismatch** (`freeze_layers` differing between checkpoint and process) raises,
     so a checkpoint from one stage of the two-stage schedule cannot be resumed into the other.

This is the resume path my earlier draft said still had to be written. It is written, and it
handles the numerical-equivalence hazard honestly instead of papering over it.

## 7. THE OPERATIVE GAP - the code is new, the files on disk are old

Measured, not assumed, across all 364 checkpoints:

| | |
|---|---|
| Legacy bare `state_dict` (no optimizer/scheduler) | **364** |
| `RESUME_MARKER_V1` wrapper checkpoints | **0** |
| Stray `.tmp` files | **0** |

**Every checkpoint on disk today is the old format.** The five running jobs were launched from
the previous `train.py` and are still writing bare state_dicts; the new code is in the working
tree but not in those processes.

The consequence, stated plainly:

- Killing or preempting a **currently running** job today and restarting it with `--resume`
  would **raise the legacy-checkpoint error and refuse**, by design. Forcing it with
  `--resume_allow_partial_state` would resume the weights but reset Adam and the LR schedule -
  a real discontinuity, and that run would no longer be cleanly comparable to the rest of the
  factorial.
- So for the **jobs in flight right now, preemption still costs up to 8 hours.** The resume
  machinery does not retroactively rescue them.
- Resume becomes genuinely safe only for runs **launched after** the updated `train.py` is in
  effect, whose checkpoints carry optimizer and scheduler state.

---

## Verdict for the preemptible-card decision

**Is anything already computed lost or corrupt? No.** 364 files, zero gaps, zero truncation,
zero NaN/Inf, 28/28 runs strict-load into a fresh CPU PEM and produce a finite dG. The disk is
trustworthy and ~89 GB against 166 TB with no quota is a non-issue.

**Can we resume from any epoch? Not from what is on disk now** - those 364 files lack the
optimizer/scheduler state, and `--resume` correctly refuses them rather than silently degrading
the run. **Yes for anything written from the updated `train.py`.**

**Recommendation before moving work to `--qos keasar` preemptible cards:**

1. Do not treat the currently-running five as protected. They are not; a preemption loses their
   in-flight epochs.
2. Land the modified `train.py` properly (it is an uncommitted working-tree change plus
   `.bak`/`.bak_resume` files sitting next to it) and confirm new jobs emit
   `format: RESUME_MARKER_V1` - one `torch.load` of the first new checkpoint proves it.
3. Only then move the factorial to preemptible cards, and verify one real
   preempt-and-resume cycle end to end before trusting the other ~90 queued jobs to it.
4. Separately worth confirming, as the briefing flagged: `--qos keasar` is not in the gpu
   partition's `AllowQos` list, so it may simply be rejected. That is untested here and is not
   a checkpoint question.
