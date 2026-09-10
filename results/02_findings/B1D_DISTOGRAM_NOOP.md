# B1d — the distogram flag is a FIFTH silent no-op. Four GPU arms are training a plain baseline.

**Date:** 2026-09-10. Checklist item **B1d**: *"distogram head on the unfolded state, weight 0.1"*.

## What we had learned before this item

`FLORY_QUARTER_TESTED.md` established that our Flory implementation swapped the unfolded
**distance map** only, while IFUM additionally **train the network to predict the unfolded
ensemble**. That learned half is the part of the idea that was never tested.

**What was not good:** I checked B1c (block magnitude) first — a side measurement — instead of the
item that actually matters. B1c was not what was asked.

**The direction:** build the half IFUM actually do — a predicted unfolded ensemble — at weight
0.1, not IFUM's 100.

## PLAN

**Q:** Is the distogram head wired into the live model, and does `--distogram_weight` do anything?
**Falsifier:** if `DistogramHead` does not appear in the model and the weight is never read by a
loss, then the flag is inert and the four running arms are baselines.

## EXECUTE + VERIFY — the falsifier FIRED

**1. `DistogramHead` is not in the live model.**

```
grep -c "DistogramHead"  model/hydro_net.py         -> 0
                         Megascale-fineTuning/train.py -> 0
                         train_utils.py              -> 0
```

It exists only in `scripts_new/distogram_head.py`, whose own header says it is a *proposal*, with
the integration code **commented out** (lines 145-151).

**2. `--distogram_weight` is parsed and stored, never used.**

```
train.py:99   _p.add_argument('--distogram_weight', ...)
train.py:147  CFG.distogram_weight = _a.distogram_weight
train.py:305  'distogram_weight': _a.distogram_weight,      <- logging only
```

**Three occurrences, all bookkeeping. No loss term reads it.**

**3. The arms confirm it empirically.** Three weights spanning 100x give the same trajectory:

| arm | val RMSE, epochs 0-5 |
|---|---|
| disto0.01_s42 | 2.173 1.985 1.868 1.860 1.855 1.862 |
| disto0.1_s42 | 2.129 1.942 1.835 1.791 1.782 1.694 |
| disto1.0_s42 | 2.180 1.961 1.838 1.818 1.816 1.741 |

**A weight sweep from 0.01 to 1.0 changes nothing beyond seed noise, because the weight is never
applied.** `disto_w12_s42` makes four arms.

## The finding

**This is the fifth silent no-op of the project's signature type: a flag that parses, logs, and
does nothing, while the run completes and reports a number.** It joins the four in
`FINDINGS` §5.5, the three-part scoring bug, the evaluate.py graph flags, and the W6 cancellation.

**Cost: four GPU arms of ~8h each — roughly 32 GPU-hours training a baseline under four different
names.** Had they been scored, they would have been written up as "the distogram lever is null".

**They are RUNNING and I am not cancelling them** (hard rule). They will complete as extra
seed-42 baselines, which is not worthless — but they must never be reported as distogram results.

## What B1d actually requires

The head must be **built and wired**, not enabled:

1. Instantiate `DistogramHead` in `PEM.__init__` when `distogram_weight > 0`.
2. In the trainer at `train.py:711/721`, after the forward pass, call it with
   `f_type='features'` (the model already exposes this at `hydro_net.py:536`) and add
   `distogram_weight * dloss` to the loss.
3. **Apply it to the UNFOLDED state** — the point of the item. The unfolded graph is built at
   `train.py:978`, so its coordinates are available at the loss site.
4. **Weight 0.1.** IFUM's 100 is calibrated to their loss scale; here CE ~ln(32)=3.5 against MSE
   ~1 would swamp the primary objective.

**A caveat that must be resolved first:** the help text argues an unfolded distogram teaches
nothing "because the coil map is an analytic function of |i-j| alone". **That is true of our
analytic coil and false of a sampled ensemble.** So the unfolded target must be a **sampled**
conformation set, not our own coil map — otherwise the head would be trained to predict a formula
it could compute directly, and the objection would be correct.

**Confidence: 99%** — three independent confirmations: absent from the model, never read by a
loss, and a 100x weight sweep with no effect.
